"""
End-to-end FACET pipeline for one main paper: narrative extraction +
citation-relation classification for every in-corpus reference, as a single
callable (facet_analyse) -- ports playground/facet_main.ipynb into a
reusable module. Reference processing uses the batched/async path
(CitationPipelinePredictor.predict_references), not a per-reference
sequential loop.

Usage (as a library) -- model/index loading is the CALLER's responsibility,
so a long-running process loads them once and reuses them across calls:
    predictor = CitationPipelinePredictor(...)
    paper_index = load_paper_index(...)
    citation_ctx_extractor = CitationContextExtractor()
    facet_relevance = await facet_analyse(
        "2308.01689", paper_index, predictor, citation_ctx_extractor)

Usage (CLI, one-shot -- loads everything fresh, prints or saves the result):
    python -m facet_main <main_paper_id> [--save]
    python -m facet_main --ids-file data/M_test_set.txt [--save]
"""

import asyncio
import gc
import traceback

import torch

from prediction.predict_main import CitationPipelinePredictor
from utils.citation_context import CitationContextExtractor, find_citation_context
from utils.seku_crd_corpus_utils import get_paper_from_s3, get_papers_from_s3, get_title_and_abstract


async def facet_analyse(main_paper_id, paper_index, predictor, citation_ctx_extractor,
                         max_references=32, narrative_max_new_tokens=2048,
                         relation_max_new_tokens=512, narrative_batch_size=8,
                         relation_batch_size=8):
    """
    Args:
        main_paper_id: corpus id of the main paper to analyse.
        paper_index: {paper_id: record}, e.g. from
            utils.seku_crd_corpus_utils.load_paper_index().
        predictor: a loaded CitationPipelinePredictor.
        citation_ctx_extractor: a CitationContextExtractor instance.
        max_references: cap on in-corpus references processed (-1 for all).
        narrative_max_new_tokens, relation_max_new_tokens,
        narrative_batch_size, relation_batch_size: passed through to
            CitationPipelinePredictor.predict_references() -- separate batch
            sizes since Layer 3 (relation) has one item per citing sentence,
            often several per reference, so it can need a smaller batch
            than Layer 1 (narrative, one item per reference) on the same GPU.

    Returns:
        dict: {m_id, m_title, m_abstract, m_aspects, references}
    """
    try:
        m_data = paper_index[main_paper_id]
        m_title = m_data["title"]
        main_paper_data = get_paper_from_s3(main_paper_id)
        _, m_abstract = get_title_and_abstract(main_paper_data)

        m_narrative = predictor.predict_narrative(m_abstract, max_new_tokens=narrative_max_new_tokens)

        status, cite_ctx = find_citation_context(
            main_paper_id, bulk_paper_index=paper_index,
            citation_context_extractor=citation_ctx_extractor, max_refs=max_references)
        if not status:
            raise RuntimeError(f"find_citation_context failed for main_paper_id={main_paper_id!r}")
        ref_contexts = cite_ctx[main_paper_id]["references"]

        in_corpus_refs_ids = m_data["in_corpus_references"]
        if max_references != -1:
            in_corpus_refs_ids = in_corpus_refs_ids[:max_references]
        ref_papers_data = await get_papers_from_s3(in_corpus_refs_ids)

        references = await predictor.predict_references(
            m_title=m_title, m_narrative=m_narrative,
            ref_contexts=ref_contexts, ref_papers_data=ref_papers_data,
            narrative_batch_size=narrative_batch_size, relation_batch_size=relation_batch_size,
            narrative_max_new_tokens=narrative_max_new_tokens,
            relation_max_new_tokens=relation_max_new_tokens,
        )
    finally:
        # Release this paper's reserved KV-cache/activation memory before the
        # caller (or facet_analyse_many's next iteration) starts the next
        # one -- on success AND on failure, so one paper's leftovers never
        # stack onto the next paper's peak.
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "m_id": main_paper_id,
        "m_title": m_title,
        "m_abstract": m_abstract,
        "m_aspects": m_narrative,
        "references": references,
    }


async def facet_analyse_many(main_paper_ids, paper_index, predictor, citation_ctx_extractor,
                              max_references=32, narrative_max_new_tokens=2048,
                              relation_max_new_tokens=512, narrative_batch_size=8,
                              relation_batch_size=8, on_result=None):
    """
    Run facet_analyse() for each id in main_paper_ids, IN SEQUENCE -- not
    concurrently. predictor shares one base model with both LoRA adapters
    attached (see CitationPipelinePredictor), switching the active adapter
    via set_adapter() before each generate() call; interleaving multiple
    facet_analyse() calls concurrently against that one shared model would
    race on which adapter is active mid-generation. Batching already
    happens WITHIN each paper's own reference processing
    (predict_references), which is where the real GPU parallelism comes from.

    A failure on one paper is caught, logged, and does not stop the rest.
    GPU memory is released after EVERY paper, success or failure -- see
    facet_analyse()'s own try/finally, which this loop relies on so one
    paper's leftover reserved KV-cache/activation memory never stacks onto
    the next paper's peak over a long --ids-file run.

    Args:
        on_result: optional callback(main_paper_id, result), invoked right
            after EACH paper succeeds -- e.g. to persist it to disk
            immediately rather than waiting for the whole id list to finish.
            Without this, the only copy of every result lives in the
            returned dict until this function returns, so a crash/OOM/kill
            partway through a long run loses everything computed so far.
            A raised exception from on_result propagates (a save failure is
            not swallowed the way a paper's own processing failure is).

    Returns:
        dict: {main_paper_id: facet_analyse() result}, one entry per id that
        succeeded. Failed ids are omitted -- see printed output for the error.
    """
    results = {}
    for main_paper_id in main_paper_ids:
        try:
            result = await facet_analyse(
                main_paper_id, paper_index, predictor, citation_ctx_extractor,
                max_references=max_references,
                narrative_max_new_tokens=narrative_max_new_tokens,
                relation_max_new_tokens=relation_max_new_tokens,
                narrative_batch_size=narrative_batch_size,
                relation_batch_size=relation_batch_size,
            )
        except Exception:
            print(f"can not process {main_paper_id}, issue {traceback.format_exc()}")
            continue
        results[main_paper_id] = result
        if on_result is not None:
            on_result(main_paper_id, result)
    return results


if __name__ == "__main__":
    # python -m facet_main <main_paper_id> [--save]
    # python -m facet_main --ids-file data/M_test_set.txt [--save]
    import argparse
    import json
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    from utils.seku_crd_corpus_utils import load_paper_index

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=True)

    ap = argparse.ArgumentParser(description="Run the FACET citation-relevance pipeline for one or many main papers")
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
                     help="Which context-classifier W&B artifact to load: 'final' (default full-run adapter), "
                          "'train-best', or 'eval-best'")
    ap.add_argument("--batch-size", type=int, default=8,
                     help="Batch size for the Layer 1 (narrative) batched generation calls. Lower this "
                          "if you hit CUDA OOM (default 8 can peak close to a 12GB GPU's limit)")
    ap.add_argument("--relation-batch-size", type=int, default=8,
                     help="Batch size for the Layer 3 (relation classification) batched generation calls. "
                          "Separate from --batch-size since Layer 3 has one item per CITING SENTENCE "
                          "(often several per reference), so it can OOM even when Layer 1 doesn't -- "
                          "lower this independently if the OOM happens during relation classification")
    ap.add_argument("--narrative-max-new-tokens", type=int, default=2048,
                     help="Max tokens generated per narrative call. Default 2048 is a large safety margin; "
                          "real narrative JSON output measured around ~270 tokens, so lowering this "
                          "(e.g. to 400-600) is the single biggest GPU memory lever if you hit OOM -- a batch "
                          "pads/grows its KV cache toward this cap if even one sequence doesn't stop cleanly")
    ap.add_argument("--nb-ids", type=int, default=None,
                     help="Process only the first N ids from --ids-file (default: all)")
    ap.add_argument("--save", action="store_true", help="Write result JSON file(s) to --output-dir instead of printing them")
    ap.add_argument("--output-dir", default=str(project_root / "data" / "facet_predictions"))
    args = ap.parse_args()

    if bool(args.main_paper_id) == bool(args.ids_file):
        ap.error("pass exactly one of main_paper_id or --ids-file")

    narrative_local_dir = os.environ.get(
        "NARRATIVE_MODEL_LOCAL_DIR", str(project_root / "model" / "narrative"))
    context_classifier_local_dir = os.environ.get(
        "CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR", str(project_root / "model" / "context_classifier"))

    print("Loading paper index...")
    paper_index = load_paper_index(max_files=args.max_index_files)

    print("Loading models...")
    predictor = CitationPipelinePredictor(
        narrative_local_dir=narrative_local_dir,
        context_classifier_local_dir=context_classifier_local_dir,
        narrative_checkpoint_kind="train-best",
        narrative_wandb_version="latest",
        narrative_version="v2",
        context_checkpoint_kind=None if args.context_checkpoint_kind == "final" else args.context_checkpoint_kind,
        context_wandb_version="latest",
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
                # e.g. "hep-th/0404065" -- building an output path from this
                # (out_dir / f"{paper_id}.json") would create a nested
                # "hep-th/" subdirectory instead of a flat filename. Skip
                # rather than special-case the path construction.
                skipped_slash_ids.append(paper_id)
                continue
            main_paper_ids.append(paper_id)

        print(f"Loaded {len(main_paper_ids)} main paper id(s) from {args.ids_file}")
        if skipped_slash_ids:
            print(f"Skipped {len(skipped_slash_ids)} id(s) containing '/' "
                  f"(would break output path construction): {skipped_slash_ids}")

        if args.nb_ids:
            main_paper_ids = main_paper_ids[:args.nb_ids]
            print(f"Capped to the first {len(main_paper_ids)} id(s) via --nb-ids {args.nb_ids}")

        if args.save:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        processed_file = project_root / "data" / "M_test_set_processed.txt"

        def on_result(main_paper_id, result):
            # Write THIS paper's result immediately, not after the whole id
            # list finishes -- otherwise a crash/OOM/kill partway through a
            # long run (thousands of papers) loses every result computed so
            # far, even the ones already done.
            if args.save:
                out_path = out_dir / f"{main_paper_id}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=3)
                print(f"Wrote -> {out_path}")
            # Mark as processed so a future --ids-file run can skip it (see
            # the companion snippet for filtering a candidate list against
            # this file).
            with open(processed_file, "a") as f:
                f.write(f"{main_paper_id}\n")
            print(f"Marked {main_paper_id} as processed -> {processed_file}")

        results = asyncio.run(facet_analyse_many(
            main_paper_ids, paper_index, predictor, citation_ctx_extractor,
            max_references=args.max_references,
            narrative_max_new_tokens=args.narrative_max_new_tokens,
            narrative_batch_size=args.batch_size,
            relation_batch_size=args.relation_batch_size,
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
            narrative_batch_size=args.batch_size,
            relation_batch_size=args.relation_batch_size,
        ))

        if args.save:
            out_path = Path(args.output_dir) / f"{args.main_paper_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=3)
            print(f"Wrote -> {out_path}")
        else:
            print(json.dumps(result, indent=2))
