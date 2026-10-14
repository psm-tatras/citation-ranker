"""
Generate a Markdown report from a facet-prediction JSON file (facet_main.ipynb
output), written next to it with a .md extension.

Usage:
    python -m utils.generate_report data/facet_predictions/2302.08982.json
"""

import argparse
import json
import os

ASPECT_KEYS = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD")
SIMILARITY_ASPECT_KEYS = ("TASK", "CONTRIBUTION", "METHOD", "RESULT")
FACET_SCORE_ASPECT_KEYS = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")


def format_aspects(aspects):
    lines = ["### Aspects", "--------", ""]
    for key in ASPECT_KEYS:
        lines.append(f"**{key}:** {aspects.get(key) or '(not stated)'}")
        lines.append("")
    return lines


def format_aspect_similarity(aspect_similarity):
    """Render prediction.qwen_embedding's M-vs-R cosine similarity, if present."""
    lines = ["### Aspect Similarity (cosine, M vs R)", "--------", ""]
    for key in SIMILARITY_ASPECT_KEYS:
        value = aspect_similarity.get(key)
        rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "(not available)"
        lines.append(f"**{key}:** {rendered}")
        lines.append("")
    return lines


def format_facet_score(facet_score):
    """Render utils.facet_scoring's per-reference importance score, if present."""
    lines = ["### Facet Score", "--------", ""]
    lines.append(f"**Score:** {facet_score.get('score')}")
    lines.append("")
    lines.append(f"**Strength:** {facet_score.get('strength')}")
    lines.append("")
    lines.append(f"**Contexts Used:** {facet_score.get('n_contexts_used')}")
    lines.append("")
    for sub_key, label in (("facet_similarity", "Facet Similarity"), ("mask", "Mask"), ("percentile", "Percentile")):
        sub = facet_score.get(sub_key) or {}
        lines.append(f"**{label}:**")
        lines.append("")
        for key in FACET_SCORE_ASPECT_KEYS:
            value = sub.get(key)
            rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "(not available)"
            lines.append(f"- **{key}:** {rendered}")
        lines.append("")
    return lines


def format_facet_ranking(ranking):
    """Render utils.facet_scoring's top-level M-paper ranking of its references, if present."""
    lines = ["# Facet Ranking", "-----------", ""]
    for i, item in enumerate(ranking, start=1):
        lines.append(f"{i}. **{item.get('r_id')}** -- score: {item.get('score')}")
        lines.append("")
    return lines


def format_context(entry, relation, index):
    text = " ".join(part for part in (
        " ".join(entry.get("before") or []),
        entry.get("center", ""),
        " ".join(entry.get("after") or []),
    ) if part)
    return [
        f"#### Context {index}", "",
        f"**Main Paper Text:** {text}", "",
        "**Classified Relation:**", "",
        "```json",
        json.dumps(relation, indent=2),
        "```",
    ]


def format_reference(r_id, r_data):
    lines = [
        f"## ID: {r_id}", "",
        f"**Title:** {r_data.get('title', '')}", "",
        f"**Abstract:** {r_data.get('abstract', '')}", "",
    ]
    lines += format_aspects(r_data.get("aspect") or {})
    if r_data.get("aspect_similarity"):
        lines += format_aspect_similarity(r_data["aspect_similarity"])
    if r_data.get("facet_score"):
        lines += format_facet_score(r_data["facet_score"])
    lines += ["### Context", "--------"]
    contexts = r_data.get("m_context") or []
    relations = r_data.get("facet_relevance") or []
    for i, (ctx, relation) in enumerate(zip(contexts, relations), start=1):
        lines.append("")
        lines += format_context(ctx, relation, i)
    return lines


def generate_report(data):
    """Render a facet-prediction dict as Markdown."""
    lines = [
        "# Main Paper Section", "--------------------", "",
        f"**ID:** {data.get('m_id', '')}", "",
        f"**Title:** {data.get('m_title', '')}", "",
        f"**Abstract:** {data.get('m_abstract', '')}", "",
    ]
    lines += format_aspects(data.get("m_aspects") or {})
    if data.get("facet_ranking"):
        lines += format_facet_ranking(data["facet_ranking"])
    lines += ["# References Section", "-----------", ""]

    for r_id, r_entry in (data.get("references") or {}).items():
        r_data = r_entry[0][r_id]
        lines += format_reference(r_id, r_data)
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Generate a readable Markdown report from a facet-prediction JSON file")
    ap.add_argument("input_json", help="Path to a facet-prediction *.json file (facet_main.ipynb output)")
    args = ap.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    out_path = os.path.splitext(args.input_json)[0] + ".md"
    with open(out_path, "w") as f:
        f.write(generate_report(data))
    print(f"Wrote report -> {out_path}")


if __name__ == "__main__":
    main()
