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


def format_aspects(aspects):
    lines = ["### Aspects", "--------", ""]
    for key in ASPECT_KEYS:
        lines.append(f"**{key}:** {aspects.get(key) or '(not stated)'}")
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
