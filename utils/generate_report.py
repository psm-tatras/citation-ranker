"""
Generate Markdown report(s) from facet-prediction JSON file(s) (facet_main.py
output). Takes either one JSON file or a directory of them; without
--output_dir, each report is written next to its input JSON (.md extension,
same base name), otherwise all reports go into --output_dir instead.

Usage:
    python -m utils.generate_report data/facet_predictions/2302.08982.json
    python -m utils.generate_report data/facet_predictions/final_output/121026-physics \\
        --output_dir data/facet_predictions/final_output/121026-physics/reports
"""

import argparse
import json
from pathlib import Path

from utils.prompts import ASPECT_KEYS


def format_aspects(aspects):
    lines = ["### Aspects", "--------", ""]
    for key in ASPECT_KEYS:
        lines.append(f"**{key}:** {aspects.get(key) or '(not stated)'}")
        lines.append("")
    return lines


def format_aspect_similarity(aspect_similarity):
    """Render prediction.qwen_embedding's M-vs-R cosine similarity, if present."""
    lines = ["### Aspect Similarity (cosine, M vs R)", "--------", ""]
    for key in ASPECT_KEYS:
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
        for key in ASPECT_KEYS:
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
        description="Generate readable Markdown report(s) from facet-prediction JSON file(s)")
    ap.add_argument("input_path", help="A facet-prediction *.json file, or a directory of them")
    ap.add_argument("--output_dir", default=None,
                     help="Write all .md reports here (same base names). Without it, each "
                          "report is written next to its own input JSON")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern when input_path is a directory")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    files = [input_path] if input_path.is_file() else sorted(input_path.glob(args.pattern))
    if not files:
        raise SystemExit(f"No JSON files found at {args.input_path!r}")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = failed = 0
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            report = generate_report(data)
        except Exception as exc:
            failed += 1
            print(f"{path.name}: {exc}")
            continue
        out_path = (out_dir / f"{path.stem}.md") if out_dir else path.with_suffix(".md")
        with open(out_path, "w") as f:
            f.write(report)
        written += 1
        print(f"Wrote report -> {out_path}")

    if len(files) > 1:
        print(f"files={len(files)} written={written} failed={failed}")


if __name__ == "__main__":
    main()
