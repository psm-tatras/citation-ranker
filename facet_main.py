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
import traceback

from prediction.predict_main import CitationPipelinePredictor
from utils.citation_context import CitationContextExtractor, find_citation_context
from utils.seku_crd_corpus_utils import get_paper_from_s3, get_papers_from_s3, get_title_and_abstract


async def facet_analyse(main_paper_id, paper_index, predictor, citation_ctx_extractor,
                         max_references=32, narrative_max_new_tokens=2048,
                         relation_max_new_tokens=512, batch_size=8):
    """
    Args:
        main_paper_id: corpus id of the main paper to analyse.
        paper_index: {paper_id: record}, e.g. from
            utils.seku_crd_corpus_utils.load_paper_index().
        predictor: a loaded CitationPipelinePredictor.
        citation_ctx_extractor: a CitationContextExtractor instance.
        max_references: cap on in-corpus references processed (-1 for all).
        narrative_max_new_tokens, relation_max_new_tokens, batch_size: passed
            through to CitationPipelinePredictor.predict_references().

    Returns:
        dict: {m_id, m_title, m_abstract, m_aspects, references}
    """
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
        batch_size=batch_size, narrative_max_new_tokens=narrative_max_new_tokens,
        relation_max_new_tokens=relation_max_new_tokens,
    )

    return {
        "m_id": main_paper_id,
        "m_title": m_title,
        "m_abstract": m_abstract,
        "m_aspects": m_narrative,
        "references": references,
    }


async def facet_analyse_many(main_paper_ids, paper_index, predictor, citation_ctx_extractor,
                              max_references=32, narrative_max_new_tokens=2048,
                              relation_max_new_tokens=512, batch_size=8):
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

    Returns:
        dict: {main_paper_id: facet_analyse() result}, one entry per id that
        succeeded. Failed ids are omitted -- see printed output for the error.
    """
    results = {}
    for main_paper_id in main_paper_ids:
        try:
            results[main_paper_id] = await facet_analyse(
                main_paper_id, paper_index, predictor, citation_ctx_extractor,
                max_references=max_references,
                narrative_max_new_tokens=narrative_max_new_tokens,
                relation_max_new_tokens=relation_max_new_tokens,
                batch_size=batch_size,
            )
        except Exception:
            print(f"can not process {main_paper_id}, issue {traceback.format_exc()}")
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
    ap.add_argument("--max-index-files", type=int, default=4096, help="load_paper_index's max_files cap (-1 for all)")
    ap.add_argument("--context-checkpoint-kind", choices=["final", "train-best", "eval-best"], default="train-best",
                     help="Which context-classifier W&B artifact to load: 'final' (default full-run adapter), "
                          "'train-best', or 'eval-best'")
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
            main_paper_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(main_paper_ids)} main paper id(s) from {args.ids_file}")

        results = asyncio.run(facet_analyse_many(
            main_paper_ids, paper_index, predictor, citation_ctx_extractor,
            max_references=args.max_references,
        ))
        print(f"Succeeded: {len(results)}/{len(main_paper_ids)}")

        if args.save:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for main_paper_id, result in results.items():
                with open(out_dir / f"{main_paper_id}.json", "w") as f:
                    json.dump(result, f, indent=3)
            print(f"Wrote {len(results)} file(s) -> {out_dir}")
        else:
            print(json.dumps(results, indent=2))
    else:
        result = asyncio.run(facet_analyse(
            args.main_paper_id, paper_index, predictor, citation_ctx_extractor,
            max_references=args.max_references,
        ))

        if args.save:
            out_path = Path(args.output_dir) / f"{args.main_paper_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=3)
            print(f"Wrote -> {out_path}")
        else:
            print(json.dumps(result, indent=2))
