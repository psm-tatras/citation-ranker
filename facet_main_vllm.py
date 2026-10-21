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
    in a background thread while the GPU works on the current chunk.
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
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from prediction.predict_main_vllm import CitationPipelinePredictorVLLM
from utils.citation_context import CitationContextExtractor, find_citation_context
from utils.seku_crd_corpus_utils import get_paper_from_s3, get_papers_from_s3, get_title_and_abstract


def prepare_paper(main_paper_id, paper_index, citation_ctx_extractor, max_references=32):
    """CPU/network part of facet_analyse() for one main paper (no GPU).

    Same data steps as facet_main.facet_analyse(). Not thread-safe across
    papers (find_citation_context writes ./temp_paper_text.json), so call it
    from one thread at a time.

    Returns:
        dict job for CitationPipelinePredictorVLLM.predict_papers():
        {m_id, m_title, m_abstract, ref_contexts, ref_papers_data}
    """
    m_data = paper_index[main_paper_id]
    m_title = m_data["title"]
    main_paper_data = get_paper_from_s3(main_paper_id)
    _, m_abstract = get_title_and_abstract(main_paper_data)

    status, cite_ctx = find_citation_context(
        main_paper_id, bulk_paper_index=paper_index,
        citation_context_extractor=citation_ctx_extractor, max_refs=max_references)
    if not status:
        raise RuntimeError(f"find_citation_context failed for main_paper_id={main_paper_id!r}")
    ref_contexts = cite_ctx[main_paper_id]["references"]

    in_corpus_refs_ids = m_data["in_corpus_references"]
    if max_references != -1:
        in_corpus_refs_ids = in_corpus_refs_ids[:max_references]
    ref_papers_data = asyncio.run(get_papers_from_s3(in_corpus_refs_ids))

    return {"m_id": main_paper_id, "m_title": m_title, "m_abstract": m_abstract,
            "ref_contexts": ref_contexts, "ref_papers_data": ref_papers_data}


def prepare_chunk(main_paper_ids, paper_index, citation_ctx_extractor, max_references=32):
    """prepare_paper() over a chunk; a failure on one paper is logged and skipped.

    Returns:
        list of jobs (successful papers only, input order).
    """
    jobs = []
    for main_paper_id in main_paper_ids:
        try:
            jobs.append(prepare_paper(main_paper_id, paper_index, citation_ctx_extractor, max_references))
        except Exception:
            print(f"can not prepare {main_paper_id}, issue {traceback.format_exc()}")
    return jobs


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
                             relation_max_new_tokens=512, chunk_size=32, on_result=None):
    """Run the pipeline for many main papers in chunks of chunk_size.

    Data preparation for chunk k+1 runs in a single background thread while
    the GPU processes chunk k. A failure on one paper (preparation or
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
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare_chunk, chunks[0], paper_index, citation_ctx_extractor, max_references)
        for k in range(len(chunks)):
            jobs = await asyncio.wrap_future(future)
            if k + 1 < len(chunks):
                future = pool.submit(prepare_chunk, chunks[k + 1], paper_index,
                                     citation_ctx_extractor, max_references)
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
                  f"{3600 * done / max(elapsed, 1e-9):.0f} papers/h")
    return results


if __name__ == "__main__":
    # python -m facet_main_vllm <main_paper_id> [--save]
    # python -m facet_main_vllm --ids-file data/M_test_set.txt [--save]
    import argparse
    import json
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    from utils.seku_crd_corpus_utils import load_paper_index

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
                    help="load_paper_index's max_files cap. Pass an int, or 'All' to load the full index")
    ap.add_argument("--context-checkpoint-kind", choices=["final", "train-best", "eval-best"], default="train-best",
                    help="Which context-classifier W&B artifact to load: 'final', 'train-best', or 'eval-best'")
    ap.add_argument("--narrative-max-new-tokens", type=int, default=2048,
                    help="Max tokens generated per narrative call (outputs average ~270 tokens)")
    ap.add_argument("--relation-max-new-tokens", type=int, default=512,
                    help="Max tokens generated per relation-classification call")
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="Main papers per vLLM chunk (all their prompts are submitted together)")
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

    print("Loading paper index...")
    paper_index = load_paper_index(max_files=args.max_index_files)

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
    citation_ctx_extractor = CitationContextExtractor()

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
            on_result=on_result,
        ))
        print(f"Succeeded: {len(results)}/{len(main_paper_ids)}")

        if not args.save:
            print(json.dumps(results, indent=2))
    else:
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
