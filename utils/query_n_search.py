"""
query_n_search.py

Utilities for aspect-level citation analysis:

  aspect_query - rank cited papers by relevance_score for a given aspect and
                 dump the main paper data + sorted references to a JSON file.
  aspect_score - compute A (span-to-span embedding similarity) and C
                 (keyword-embedding overlap) for a given aspect, between the
                 main paper and each of its references.
  compute_C    - keyword-overlap bonus via global max over all
                 (main_keyword, ref_keyword) pairs.

Usage
-----
    python -m utils.query_n_search \
        --filename data/.../results/2310.02903.json \
        --aspect CONTRIBUTION \
        --top_n 10 \
        --output_dir data/.../results/aspect_queries
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from utils.embedding_utils import *


def _filter_aspect(d, aspect):
    """Return a dict containing only the given aspect key, if present."""
    return {aspect: d[aspect]} if aspect in d else {}


def aspect_query(filename, aspect, top_n, output_dir, aspect_only=False):
    """
    Read a citation JSON file, rank cited papers by relevance_score for the
    given aspect, and dump the main paper data + sorted references to a file.

    Parameters
    ----------
    filename : str
        Path to the JSON file (single main paper with references).
    aspect : str
        Aspect name to rank by (e.g. "PROBLEM", "CONTRIBUTION").
    top_n : int
        Number of top cited papers to keep.
    output_dir : str
        Directory to write the output file into. Output filename is the
        input filename's stem, suffixed with the aspect (e.g.
        "2310_02903_PROBLEM.json").
    aspect_only : bool
        If True, the output file's "nes", "relevance_score", and
        "aspect_type" / "main_aspect_type" fields (main paper and each
        reference) are restricted to only the given aspect — no other
        aspect or its score is written.

    Returns
    -------
    List[dict]
        Cited paper entries, exactly as in the input JSON, sorted descending
        by relevance_score[aspect]. References missing this aspect score are
        excluded.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    main_id = next(iter(data))
    main_data = data[main_id]
    references = main_data.get("references", {})

    scored = [
        ref for ref in references.values()
        if aspect in ref.get("relevance_score", {})
    ]

    scored.sort(key=lambda ref: ref["relevance_score"][aspect], reverse=True)

    top_refs = scored[:top_n]

    # === Build main data (everything except "references") ===
    output = {k: v for k, v in main_data.items() if k != "references"}

    # === Build references to write (filtered if aspect_only) ===
    if aspect_only:
        if "nes" in output:
            output["nes"] = _filter_aspect(output["nes"], aspect)
        if "main_aspect_type" in output:
            output["main_aspect_type"] = _filter_aspect(output["main_aspect_type"], aspect)

        refs_to_write = []
        for ref in top_refs:
            filtered_ref = {k: v for k, v in ref.items() if k not in ("nes", "relevance_score", "aspect_type")}
            filtered_ref["nes"] = _filter_aspect(ref.get("nes", {}), aspect)
            filtered_ref["relevance_score"] = _filter_aspect(ref.get("relevance_score", {}), aspect)
            if "aspect_type" in ref:
                filtered_ref["aspect_type"] = _filter_aspect(ref["aspect_type"], aspect)
            refs_to_write.append(filtered_ref)
    else:
        refs_to_write = top_refs

    output["references"] = refs_to_write

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(filename).stem
    output_path = out_dir / f"{input_stem}_{aspect}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return top_refs


# ============================================================================
# Existing scoring primitive (as provided)
# ============================================================================

def reference_relevance_computation(base_text, ref_dict, embedding_model):
    """Cosine-similarity relevance of each reference text to base_text.

    base_text : the main paper's text.
    ref_dict  : {paper_id: text} for the reference papers to score.
    embedding_model : a loaded EmbeddingModel (load_embedding_model() already
                       called).

    Returns {paper_id: cosine similarity with base_text}.
    """
    ref_embeddings = embedding_model.get_embedding_batch(ref_dict)
    base_embedding = embedding_model.get_embedding_batch({"__base__": base_text})["__base__"]

    base_norm = np.linalg.norm(base_embedding)
    similarities = {}
    for paper_id, vec in ref_embeddings.items():
        denom = base_norm * np.linalg.norm(vec)
        similarities[paper_id] = float(np.dot(base_embedding, vec) / denom) if denom else 0.0
    return similarities


# ============================================================================
# Keyword extraction helper
# ============================================================================

_KEYWORD_PATTERN = re.compile(r"<([^<>]+)>")


def _extract_keywords(text):
    """Extract <...>-tagged keyword phrases from a span. Returns a list of strings."""
    return _KEYWORD_PATTERN.findall(text or "")


# ============================================================================
# compute_C
# ============================================================================

def _cosine(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom else 0.0


def compute_C(main_keywords, ref_keywords_dict, embedding_model):
    """
    Keyword-overlap bonus via global max over all (main_keyword, ref_keyword)
    pairs. Each keyword is embedded individually (not joined into one
    string), so a single strong match isn't diluted by unrelated keywords
    bundled alongside it on either side.

    Parameters
    ----------
    main_keywords : List[str]
        Keyword phrases extracted from the main paper's aspect span.
    ref_keywords_dict : Dict[str, List[str]]
        {paper_id: [keyword phrases]} extracted from each reference's aspect
        span.
    embedding_model : EmbeddingModel
        A loaded embedding model (load_embedding_model() already called).

    Returns
    -------
    Dict[str, float]
        {paper_id: max cosine similarity across all (main_keyword,
        ref_keyword) pairs}. Papers with no extracted keywords are omitted.
    """
    if not main_keywords:
        return {}

    main_vecs = embedding_model.get_embedding_batch(
        {f"main_{i}": kw for i, kw in enumerate(main_keywords)}
    )

    flat_ref_dict = {}
    ref_index_map = {}
    for paper_id, kws in ref_keywords_dict.items():
        keys = [f"{paper_id}__{j}" for j in range(len(kws))]
        for key, kw in zip(keys, kws):
            flat_ref_dict[key] = kw
        if keys:
            ref_index_map[paper_id] = keys

    if not flat_ref_dict:
        return {}

    ref_vecs = embedding_model.get_embedding_batch(flat_ref_dict)

    C_scores = {}
    for paper_id, keys in ref_index_map.items():
        best = 0.0
        for m_vec in main_vecs.values():
            for key in keys:
                sim = _cosine(m_vec, ref_vecs[key])
                if sim > best:
                    best = sim
        C_scores[paper_id] = best

    return C_scores


# ============================================================================
# aspect_score
# ============================================================================

def aspect_score(filename, aspect_name, embedding_model):
    """
    Compute A (span-to-span) and C (keyword-embedding overlap, via
    compute_C) for each reference of a main paper, restricted to one aspect.

    Parameters
    ----------
    filename : str
        Path to the JSON file (single main paper with references).
    aspect_name : str
        Aspect to score (e.g. "PROBLEM", "FINDING", "CONTRIBUTION").
    embedding_model : EmbeddingModel
        A loaded embedding model (load_embedding_model() already called).

    Returns
    -------
    Dict[str, Dict[str, float]]
        {paper_id: {"A": float, "C": float}} for references that have the
        given aspect extracted. If the main paper has no span for this
        aspect, returns {}.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    main_id = next(iter(data))
    main_data = data[main_id]
    main_nes = main_data.get("nes", {})

    if aspect_name not in main_nes:
        return {}

    base_text = main_nes[aspect_name]
    references = main_data.get("references", {})

    # === Span-level texts for A ===
    ref_span_dict = {
        paper_id: ref["nes"][aspect_name]
        for paper_id, ref in references.items()
        if aspect_name in ref.get("nes", {})
    }

    if not ref_span_dict:
        return {}

    A_scores = reference_relevance_computation(base_text, ref_span_dict, embedding_model)

    # === Keyword-level max-pair overlap for C ===
    main_keywords = _extract_keywords(base_text)
    ref_keywords_dict = {
        paper_id: _extract_keywords(span_text)
        for paper_id, span_text in ref_span_dict.items()
    }
    C_scores = compute_C(main_keywords, ref_keywords_dict, embedding_model)

    # === Combine ===
    return {
        paper_id: {"A": A_scores.get(paper_id, 0.0), "C": C_scores.get(paper_id, 0.0)}
        for paper_id in ref_span_dict
    }


# ============================================================================
# aspect_query_v2
# ============================================================================

def aspect_query_v2(filename, aspect, top_n, output_dir, embedding_model, aspect_only=False):
    """
    Like aspect_query, but ranks cited papers using aspect_score() (A, C)
    instead of a precomputed relevance_score. Sorted by A (span-to-span
    embedding similarity); C (keyword-embedding overlap) is attached
    alongside each kept reference under "aspect_score".

    Parameters
    ----------
    filename : str
        Path to the JSON file (single main paper with references).
    aspect : str
        Aspect name to rank by (e.g. "PROBLEM", "CONTRIBUTION").
    top_n : int
        Number of top cited papers to keep.
    output_dir : str
        Directory to write the output file into. Output filename is the
        input filename's stem, suffixed with the aspect and "_v2" (e.g.
        "2310_02903_PROBLEM_v2.json").
    embedding_model : EmbeddingModel
        A loaded embedding model (load_embedding_model() already called).
    aspect_only : bool
        If True, the output file's "nes", "relevance_score", and
        "aspect_type" / "main_aspect_type" fields (main paper and each
        reference) are restricted to only the given aspect.

    Returns
    -------
    List[dict]
        Cited paper entries (with an added "aspect_score": {aspect: {"A":..,
        "C":..}} key), sorted descending by A. References missing this
        aspect are excluded.
    """
    scores = aspect_score(filename, aspect, embedding_model)

    if not scores:
        return []

    with open(filename, "r") as f:
        data = json.load(f)

    main_id = next(iter(data))
    main_data = data[main_id]
    references = main_data.get("references", {})

    scored = [
        (paper_id, ref, scores[paper_id])
        for paper_id, ref in references.items()
        if paper_id in scores
    ]

    scored.sort(key=lambda item: item[2]["A"], reverse=True)

    top_items = scored[:top_n]

    top_refs = []
    for paper_id, ref, score in top_items:
        ref_copy = dict(ref)
        ref_copy["aspect_score"] = {aspect: score}
        top_refs.append(ref_copy)

    # === Build main data (everything except "references") ===
    output = {k: v for k, v in main_data.items() if k != "references"}

    # === Build references to write (filtered if aspect_only) ===
    if aspect_only:
        if "nes" in output:
            output["nes"] = _filter_aspect(output["nes"], aspect)
        if "main_aspect_type" in output:
            output["main_aspect_type"] = _filter_aspect(output["main_aspect_type"], aspect)

        refs_to_write = []
        for ref in top_refs:
            filtered_ref = {
                k: v for k, v in ref.items()
                if k not in ("nes", "relevance_score", "aspect_type", "aspect_score")
            }
            filtered_ref["nes"] = _filter_aspect(ref.get("nes", {}), aspect)
            if "relevance_score" in ref:
                filtered_ref["relevance_score"] = _filter_aspect(ref["relevance_score"], aspect)
            if "aspect_type" in ref:
                filtered_ref["aspect_type"] = _filter_aspect(ref["aspect_type"], aspect)
            filtered_ref["aspect_score"] = ref["aspect_score"]  # already aspect-restricted
            refs_to_write.append(filtered_ref)
    else:
        refs_to_write = top_refs

    output["references"] = refs_to_write

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(filename).stem
    output_path = out_dir / f"{input_stem}_{aspect}_v2.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return top_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filename", required=True,
                     help="Path to the citation JSON file (single main paper with references).")
    ap.add_argument("--aspect", required=True,
                     help="Aspect to rank references by (e.g. 'PROBLEM', 'CONTRIBUTION').")
    ap.add_argument("--top_n", type=int, default=10,
                     help="Number of top cited papers to keep (default: 10).")
    ap.add_argument("--output_dir", required=True,
                     help="Directory to write the output file into.")
    ap.add_argument("--aspect_only", action="store_true",
                     help="Restrict the output's nes/relevance_score/aspect_type "
                          "fields to just the queried aspect.")
    args = ap.parse_args()

    # === aspect_query (precomputed relevance_score) ===
    # top_refs = aspect_query(
    #     filename=args.filename,
    #     aspect=args.aspect,
    #     top_n=args.top_n,
    #     output_dir=args.output_dir,
    #     aspect_only=args.aspect_only,
    # )
    # print(f"Found {len(top_refs)} reference(s) with a '{args.aspect}' relevance score")
    # print(f"Written to {args.output_dir}")

    # === aspect_query_v2 (fresh A/C via aspect_score) ===
    embedding_model = EmbeddingModel()
    embedding_model.load_embedding_model()

    top_refs = aspect_query_v2(
        filename=args.filename,
        aspect=args.aspect,
        top_n=args.top_n,
        output_dir=args.output_dir,
        embedding_model=embedding_model,
        aspect_only=args.aspect_only,
    )
    print(f"Found {len(top_refs)} reference(s) with a '{args.aspect}' aspect score")
    print(f"Written to {args.output_dir}")


if __name__ == "__main__":
    main()