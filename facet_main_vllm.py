"""
vLLM version of facet_main.py: the same FACET pipeline (narrative extraction +
citation-relation classification for every in-corpus reference), served by
CitationPipelinePredictorVLLM (prediction/predict_main_vllm.py).

Differences from facet_main.py:
  - Main papers are processed in CHUNKS (--chunk-size): Layer 1 for all M and
    reference abstracts of a chunk goes in ONE vLLM call, then Layer 3 for all
    citation contexts of the chunk in a second call. vLLM serves both LoRA
    adapters per request, so there is no set_adapter() race and no need to run
    papers one at a time.
  - S3 / citation-context preparation (CPU + network) for the NEXT chunk runs
    in the background while the GPU works on the current chunk, in
    --prep-workers separate PROCESSES (it is CPU-bound Python, so threads
    would share one core); each main paper and its references are fetched
    once (see prepare_paper).
  - Only the paper-index entries of the main papers being processed are kept
    in memory (the full index is several GB; references come from S3).
  - --skip-processed resumes a run from the processed-ids file (spot restarts).

Output per paper is the same dict as facet_main.facet_analyse():
    {m_id, m_title, m_abstract, m_aspects, references}

Usage (as a library):
    predictor = CitationPipelinePredictorVLLM(...)
    paper_index = load_paper_index(...)
    citation_ctx_extractor = CitationContextExtractor()
    result = await facet_analyse("2308.01689", paper_index, predictor, citation_ctx_extractor)

Usage (CLI):
    python -m facet_main_vllm <main_paper_id> [--save]
    python -m facet_main_vllm --ids-file data/M_test_set.txt [--save] [--chunk-size 32] [--skip-processed]
"""

import asyncio
import json
import multiprocessing as mp
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import boto3
from botocore.config import Config

import utils.seku_crd_corpus_utils as corpus
from utils.citation_context import CitationContextExtractor
# CitationPipelinePredictorVLLM (vLLM + torch) is imported only in __main__, so
# the spawned preparation workers, which import this module, stay small.
from utils.seku_crd_corpus_utils import get_paper_from_s3, get_title_and_abstract

# One S3 client and one fetch pool for the whole process. boto3 clients are
# thread-safe once created, but creating them concurrently from the default
# session is not -- so create it once, under a lock.
S3_FETCH_WORKERS = 16
_s3_client = None
_s3_lock = threading.Lock()
_fetch_pool = ThreadPoolExecutor(max_workers=S3_FETCH_WORKERS)


def _get_s3_client():
    global _s3_client
    with _s3_lock:
        if _s3_client is None:
            _s3_client = boto3.client(
                "s3",
                aws_access_key_id=corpus.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=corpus.AWS_SECRET_ACCESS_KEY,
                region_name=corpus.AWS_REGION,
                config=Config(max_pool_connections=S3_FETCH_WORKERS * 2, retries={"max_attempts": 3}),
            )
        return _s3_client


def _slim_paper(paper):
    """Keep only what the predictor reads from a reference paper (title,
    abstract) so full texts are not held in memory. get_title_and_abstract()
    returns the same values for the slim and the full record."""
    if paper is None:
        return None
    title, abstract = get_title_and_abstract(paper)
    return {"metadata": {"title": title}, "abstract": abstract}


def _fetch_slim(paper_id):
    return _slim_paper(get_paper_from_s3(paper_id, s3_client=_get_s3_client()))


def prepare_paper(main_paper_id, paper_index, citation_ctx_extractor, max_references=32):
    """CPU/network part of facet_analyse() for one main paper (no GPU).

    Produces the same ref_contexts as utils.citation_context.find_citation_context(),
    but faster:
      - the main paper is fetched once and passed to the extractor as a dict
        (find_citation_context writes it to ./temp_paper_text.json and the
        extractor re-reads that file once per reference);
      - reference papers are fetched once, concurrently (find_citation_context
        fetches them one by one, and facet_analyse fetches them again), through
        one shared S3 client, and reduced to title + abstract immediately so
        full texts never accumulate in memory.
    No shared temp file, so several papers can be prepared in parallel threads.

    Returns:
        dict job for CitationPipelinePredictorVLLM.predict_papers():
        {m_id, m_title, m_abstract, ref_contexts, ref_papers_data}
    """
    m_data = paper_index[main_paper_id]
    return _prepare_core(main_paper_id, m_data["title"], m_data["in_corpus_references"],
                         citation_ctx_extractor, max_references)


def _prepare_core(main_paper_id, m_title, in_corpus_refs_ids, citation_ctx_extractor,
                  max_references=32):
    """prepare_paper() without the paper-index lookup (runs in prep workers)."""
    main_paper_data = get_paper_from_s3(main_paper_id, s3_client=_get_s3_client())
    if main_paper_data is None:
        raise RuntimeError(f"main paper {main_paper_id!r} not found in S3")
    _, m_abstract = get_title_and_abstract(main_paper_data)

    if max_references != -1:
        in_corpus_refs_ids = in_corpus_refs_ids[:max_references]
    # Fetched concurrently on the shared pool and slimmed right away: only
    # title + abstract of each reference is kept.
    ref_papers_data = dict(zip(in_corpus_refs_ids, _fetch_pool.map(_fetch_slim, in_corpus_refs_ids)))

    # Same per-reference logic as find_citation_context(): keep references
    # with at least one citation context; errors are per reference.
    ref_contexts = {}
    for ref_paper_id in in_corpus_refs_ids:
        try:
            result = citation_ctx_extractor.extract(main_paper_data, ref_paper_id)
            if result:
                _, ref_abstract = get_title_and_abstract(ref_papers_data.get(ref_paper_id))
                ref_contexts[ref_paper_id] = {"citation_context": result, "abstract": ref_abstract}
        except Exception as e:
            print(f"  ref {ref_paper_id}: {e}")
    print(f"Extracted context for {main_paper_id} success [{len(ref_contexts)}/{len(in_corpus_refs_ids)}]")
    del main_paper_data  # full text no longer needed

    return {"m_id": main_paper_id, "m_title": m_title, "m_abstract": m_abstract,
            "ref_contexts": ref_contexts, "ref_papers_data": ref_papers_data}


_worker_extractors = {}


def _prep_task(task):
    """Process-pool entry point: (main_paper_id, m_title, refs, max_references,
    context_window) -> job or None. Errors are logged, not raised."""
    main_paper_id, m_title, refs, max_references, context_window = task
    try:
        extractor = _worker_extractors.get(context_window)
        if extractor is None:
            extractor = _worker_extractors[context_window] = CitationContextExtractor(S=context_window)
        return _prepare_core(main_paper_id, m_title, refs, extractor, max_references)
    except Exception:
        print(f"can not prepare {main_paper_id}, issue {traceback.format_exc()}", flush=True)
        return None


def make_prep_executor(prep_workers):
    """Process pool for preparation ('spawn': the parent holds CUDA/vLLM state,
    which must not be forked). None -> prepare in the calling thread."""
    if prep_workers <= 0:
        return None
    return ProcessPoolExecutor(max_workers=prep_workers, mp_context=mp.get_context("spawn"))


def prepare_chunk(main_paper_ids, paper_index, citation_ctx_extractor, max_references=32,
                  executor=None):
    """Prepare a chunk of main papers, on `executor` (a make_prep_executor()
    pool) if given; a failure on one paper is logged and skipped.

    Returns:
        list of jobs (successful papers only, input order).
    """
    tasks = []
    for main_paper_id in main_paper_ids:
        m_data = paper_index.get(main_paper_id)
        if m_data is None:
            print(f"can not prepare {main_paper_id}, issue: not in the paper index")
            continue
        tasks.append((main_paper_id, m_data["title"], m_data["in_corpus_references"],
                      max_references, citation_ctx_extractor.S))
    if executor is None:
        jobs = [_prep_task(t) for t in tasks]
    else:
        jobs = list(executor.map(_prep_task, tasks))
    return [job for job in jobs if job is not None]


def load_index_subset(wanted_ids, max_files=-1, paper_index_file=None):
    """Read the paper index JSONL but keep only the records of wanted_ids.

    Same file and max_files semantics as utils.seku_crd_corpus_utils.load_paper_index()
    (-1 = whole file), without holding every record in memory.

    Returns:
        {paper_id: record} for the wanted ids found.
    """
    wanted = set(wanted_ids)
    limit = 200000 if max_files == -1 else max_files
    path = paper_index_file or corpus.PAPER_INDEX_FILE
    subset = {}
    with open(path, "r") as f:
        for n, line in enumerate(f, 1):
            rec = json.loads(line)
            if rec["paper_id"] in wanted:
                subset[rec["paper_id"]] = rec
            if n >= limit or len(subset) == len(wanted):
                break
    print(f"Paper index: kept {len(subset)}/{len(wanted)} wanted record(s) from {path}")
    return subset


def _to_result(job, pred):
    return {
        "m_id": job["m_id"],
        "m_title": job["m_title"],
        "m_abstract": job["m_abstract"],
        "m_aspects": pred["m_aspects"],
        "references": pred["references"],
    }


def _predict_jobs(jobs, predictor, narrative_max_new_tokens, relation_max_new_tokens):
    """predict_papers() over a chunk; if the chunk call fails, retry paper by
    paper so one bad paper does not lose the whole chunk.

    Returns:
        list of (job, result_or_None).
    """
    if not jobs:
        return []
    try:
        preds = predictor.predict_papers(
            jobs, narrative_max_new_tokens=narrative_max_new_tokens,
            relation_max_new_tokens=relation_max_new_tokens)
        return [(job, _to_result(job, pred)) for job, pred in zip(jobs, preds)]
    except Exception:
        print(f"chunk prediction failed, retrying per paper; issue {traceback.format_exc()}")
    out = []
    for job in jobs:
        try:
            pred = predictor.predict_papers(
                [job], narrative_max_new_tokens=narrative_max_new_tokens,
                relation_max_new_tokens=relation_max_new_tokens)[0]
            out.append((job, _to_result(job, pred)))
        except Exception:
            print(f"can not process {job['m_id']}, issue {traceback.format_exc()}")
            out.append((job, None))
    return out


async def facet_analyse(main_paper_id, paper_index, predictor, citation_ctx_extractor,
                        max_references=32, narrative_max_new_tokens=2048,
                        relation_max_new_tokens=512):
    """Same as facet_main.facet_analyse(), for one main paper, on vLLM.

    Returns:
        dict: {m_id, m_title, m_abstract, m_aspects, references}
    """
    job = await asyncio.to_thread(
        prepare_paper, main_paper_id, paper_index, citation_ctx_extractor, max_references)
    pred = predictor.predict_papers(
        [job], narrative_max_new_tokens=narrative_max_new_tokens,
        relation_max_new_tokens=relation_max_new_tokens)[0]
    return _to_result(job, pred)


async def facet_analyse_many(main_paper_ids, paper_index, predictor, citation_ctx_extractor,
                             max_references=32, narrative_max_new_tokens=2048,
                             relation_max_new_tokens=512, chunk_size=16, prep_workers=3,
                             on_result=None):
    """Run the pipeline for many main papers in chunks of chunk_size.

    Data preparation for chunk k+1 runs in the background (prep_workers
    worker processes) while the GPU processes chunk k. A failure on one paper (preparation or
    prediction) is logged and does not stop the rest.

    Args:
        on_result: optional callback(main_paper_id, result), invoked right
            after each paper's result is ready (e.g. to persist it at once).
            An exception from on_result propagates.

    Returns:
        dict: {main_paper_id: result}, one entry per successful id.
    """
    chunks = [main_paper_ids[i:i + chunk_size] for i in range(0, len(main_paper_ids), chunk_size)]
    results = {}
    if not chunks:
        return results

    done = 0
    t0 = time.time()
    executor = make_prep_executor(prep_workers)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare_chunk, chunks[0], paper_index, citation_ctx_extractor,
                                 max_references, executor)
            for k in range(len(chunks)):
                jobs = await asyncio.wrap_future(future)
                if k + 1 < len(chunks):
                    future = pool.submit(prepare_chunk, chunks[k + 1], paper_index,
                                         citation_ctx_extractor, max_references, executor)
                for job, result in _predict_jobs(jobs, predictor, narrative_max_new_tokens,
                                                 relation_max_new_tokens):
                    if result is None:
                        continue
                    results[job["m_id"]] = result
                    if on_result is not None:
                        on_result(job["m_id"], result)
                done += len(chunks[k])
                elapsed = time.time() - t0
                print(f"[chunk {k + 1}/{len(chunks)}] ids done {done}/{len(main_paper_ids)}, "
                      f"succeeded {len(results)}, {elapsed / 60:.1f} min elapsed, "
                      f"{3600 * done / max(elapsed, 1e-9):.0f} papers/h", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    return results


if __name__ == "__main__":
    # python -m facet_main_vllm <main_paper_id> [--save]
    # python -m facet_main_vllm --ids-file data/M_test_set.txt [--save]
    import argparse
    import json
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    from prediction.predict_main_vllm import CitationPipelinePredictorVLLM

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=True)

    ap = argparse.ArgumentParser(description="Run the FACET citation-relevance pipeline on vLLM for one or many main papers")
    ap.add_argument("main_paper_id", nargs="?", default=None,
                    help="Corpus id of a single main paper to analyse (mutually exclusive with --ids-file)")
    ap.add_argument("--ids-file", default=None,
                    help="Text file of main paper ids, one per line, to analyse in bulk "
                         "(mutually exclusive with main_paper_id)")
    ap.add_argument("--max-references", type=int, default=32, help="Cap on references processed (-1 for all)")

    def _max_index_files(value):
        if value.strip().lower() == "all":
            return -1
        return int(value)

    ap.add_argument("--max-index-files", type=_max_index_files, default=4096,
                    help="Lines of the paper index to scan (same as load_paper_index's max_files). "
                         "Pass an int, or 'All' to scan the full index; only the requested main "
                         "papers' records are kept in memory")
    ap.add_argument("--context-checkpoint-kind", choices=["final", "train-best", "eval-best"], default="train-best",
                    help="Which context-classifier W&B artifact to load: 'final', 'train-best', or 'eval-best'")
    ap.add_argument("--narrative-max-new-tokens", type=int, default=2048,
                    help="Max tokens generated per narrative call (outputs average ~270 tokens)")
    ap.add_argument("--relation-max-new-tokens", type=int, default=512,
                    help="Max tokens generated per relation-classification call")
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="Main papers per vLLM chunk (all their prompts are submitted together)")
    ap.add_argument("--prep-workers", type=int, default=3,
                    help="Worker processes preparing main papers (S3 fetch + citation-context "
                         "extraction) in parallel; 0 = prepare in the main process")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="Fraction of GPU memory vLLM may use")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="Max prompt + generated tokens per request")
    ap.add_argument("--max-num-seqs", type=int, default=128,
                    help="Max sequences vLLM decodes concurrently")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graphs (saves memory, slower)")
    ap.add_argument("--nb-ids", type=int, default=None,
                    help="Process only the first N ids from --ids-file (default: all)")
    ap.add_argument("--skip-processed", action="store_true",
                    help="Skip ids already listed in --processed-file (resume an interrupted run)")
    ap.add_argument("--processed-file", default=str(project_root / "data" / "M_test_set_processed_vllm.txt"),
                    help="File that successfully processed ids are appended to")
    ap.add_argument("--save", action="store_true", help="Write result JSON file(s) to --output-dir instead of printing them")
    ap.add_argument("--output-dir", default=str(project_root / "data" / "facet_predictions_vllm"))
    args = ap.parse_args()

    if bool(args.main_paper_id) == bool(args.ids_file):
        ap.error("pass exactly one of main_paper_id or --ids-file")

    narrative_local_dir = os.environ.get(
        "NARRATIVE_MODEL_LOCAL_DIR", str(project_root / "model" / "narrative"))
    context_classifier_local_dir = os.environ.get(
        "CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR", str(project_root / "model" / "context_classifier"))

    def load_models():
        print("Loading models (vLLM)...")
        predictor = CitationPipelinePredictorVLLM(
            narrative_local_dir=narrative_local_dir,
            context_classifier_local_dir=context_classifier_local_dir,
            narrative_checkpoint_kind="train-best",
            narrative_wandb_version="latest",
            narrative_version="v2",
            context_checkpoint_kind=None if args.context_checkpoint_kind == "final" else args.context_checkpoint_kind,
            context_wandb_version="latest",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
        )
        return predictor, CitationContextExtractor()

    if args.ids_file:
        with open(args.ids_file) as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # Entries look like "2305.09630:Physics" -- take the id, drop the domain.
        main_paper_ids = []
        skipped_slash_ids = []
        for line in raw_lines:
            paper_id = line.split(":")[0].strip()
            if "/" in paper_id:
                # e.g. "hep-th/0404065" would create a nested output subdirectory.
                skipped_slash_ids.append(paper_id)
                continue
            main_paper_ids.append(paper_id)

        print(f"Loaded {len(main_paper_ids)} main paper id(s) from {args.ids_file}")
        if skipped_slash_ids:
            print(f"Skipped {len(skipped_slash_ids)} id(s) containing '/' "
                  f"(would break output path construction): {skipped_slash_ids}")

        processed_file = Path(args.processed_file)
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_processed and processed_file.exists():
            with open(processed_file) as f:
                already = {line.strip() for line in f if line.strip()}
            before = len(main_paper_ids)
            main_paper_ids = [pid for pid in main_paper_ids if pid not in already]
            print(f"Skipped {before - len(main_paper_ids)} already processed id(s) listed in {processed_file}")

        if args.nb_ids:
            main_paper_ids = main_paper_ids[:args.nb_ids]
            print(f"Capped to the first {len(main_paper_ids)} id(s) via --nb-ids {args.nb_ids}")

        print("Loading paper index (requested main papers only)...")
        paper_index = load_index_subset(main_paper_ids, max_files=args.max_index_files)
        predictor, citation_ctx_extractor = load_models()

        if args.save:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        def on_result(main_paper_id, result):
            # Write THIS paper's result immediately so an interrupted run keeps
            # everything already computed.
            if args.save:
                out_path = out_dir / f"{main_paper_id}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=3)
                print(f"Wrote -> {out_path}")
            with open(processed_file, "a") as f:
                f.write(f"{main_paper_id}\n")

        results = asyncio.run(facet_analyse_many(
            main_paper_ids, paper_index, predictor, citation_ctx_extractor,
            max_references=args.max_references,
            narrative_max_new_tokens=args.narrative_max_new_tokens,
            relation_max_new_tokens=args.relation_max_new_tokens,
            chunk_size=args.chunk_size,
            prep_workers=args.prep_workers,
            on_result=on_result,
        ))
        print(f"Succeeded: {len(results)}/{len(main_paper_ids)}")

        if not args.save:
            print(json.dumps(results, indent=2))
    else:
        print("Loading paper index (requested main paper only)...")
        paper_index = load_index_subset([args.main_paper_id], max_files=args.max_index_files)
        predictor, citation_ctx_extractor = load_models()

        result = asyncio.run(facet_analyse(
            args.main_paper_id, paper_index, predictor, citation_ctx_extractor,
            max_references=args.max_references,
            narrative_max_new_tokens=args.narrative_max_new_tokens,
            relation_max_new_tokens=args.relation_max_new_tokens,
        ))

        if args.save:
            out_path = Path(args.output_dir) / f"{args.main_paper_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=3)
            print(f"Wrote -> {out_path}")
        else:
            print(json.dumps(result, indent=2))
