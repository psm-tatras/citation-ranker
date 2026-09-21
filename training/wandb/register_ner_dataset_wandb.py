#!/usr/bin/env python3
"""
Register the NER dataset (ner_spans.jsonl, ner_bio.jsonl, ner_labels.json) as
a versioned W&B dataset artifact. Does nothing else -- no model training, no
S3 upload. Run this whenever the dataset files change to create a new
artifact version.

Usage:
    python3 register_ner_dataset_wandb.py \
        --data-dir "/Work/Experiments/citation-processor-ranker/data/SEKU/processed/ner_data_3k" \
        --project "PSM-IISER-NER"
"""

import argparse
import datetime
import glob
import json
import os
from collections import Counter
from pathlib import Path

import wandb


def find_stem_files(data_dir: Path):
    """Locate <stem>_ner_spans.jsonl (and its siblings _ner_bio.jsonl,
    _ner_labels.json, _summary.md) in data_dir. Returns (stem, spans_file,
    bio_file, labels_file, summary_file_or_None)."""
    candidates = glob.glob(os.path.join(data_dir, "*_ner_spans.jsonl"))
    if not candidates:
        raise SystemExit(f"No *_ner_spans.jsonl file found in {data_dir!r}")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple *_ner_spans.jsonl files found in "
                          f"{data_dir!r}: {candidates}")
    spans_file = Path(candidates[0])
    stem = spans_file.name[: -len("_ner_spans.jsonl")]
    bio_file = data_dir / f"{stem}_ner_bio.jsonl"
    labels_file = data_dir / f"{stem}_ner_labels.json"
    summary_file = data_dir / f"{stem}_summary.md"
    if not summary_file.exists():
        summary_file = None
    return stem, spans_file, bio_file, labels_file, summary_file


def analyze_dataset(spans_file: Path, bio_file: Path, labels_file: Path):
    """Read the dataset files and compute the stats that go into the artifact
    metadata and the dataset card (record/entity counts, label distribution)."""
    num_span_records = 0
    label_counts = Counter()
    with open(spans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            num_span_records += 1
            for ent in record.get("entities", []):
                label_counts[ent["label"]] += 1
    total_entities = sum(label_counts.values())

    num_bio_records = 0
    with open(bio_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num_bio_records += 1

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data.get("labels", sorted(label_counts.keys()))

    return {
        "num_span_records": num_span_records,
        "num_bio_records": num_bio_records,
        "total_entities": total_entities,
        "label_counts": label_counts,
        "labels": labels,
    }


def write_dataset_card(
    card_path: Path,
    stats: dict,
    spans_file: Path,
    bio_file: Path,
    labels_file: Path,
    artifact_name: str,
    source: str,
    summary_file: Path = None,
):
    """Regenerate DATASET_CARD.md from the current files (kept in sync with
    whatever version is being registered, not hand-maintained)."""
    total_entities = stats["total_entities"]
    num_span_records = stats["num_span_records"]
    avg_entities = total_entities / num_span_records if num_span_records else 0

    spans_mb = spans_file.stat().st_size / (1024 ** 2)
    bio_mb = bio_file.stat().st_size / (1024 ** 2)

    label_rows = "\n".join(
        f"| {label} | {count} | {count / total_entities * 100:.1f}% |"
        for label, count in stats["label_counts"].most_common()
    )

    card = f"""# NER Dataset Card

## Overview
- **Records (spans format):** {num_span_records}
- **Records (BIO format):** {stats['num_bio_records']}
- **Total Entities:** {total_entities}
- **Average Entities/Record:** {avg_entities:.2f}

## Files
- `{spans_file.name}` – char-offset format ({{text, entities: [{{start, end, label, text}}]}})
- `{bio_file.name}` – word-level BIO format ({{tokens, ner_tags, ner_tags_str}})
- `{labels_file.name}` – label mapping (id2label, label2id)

## Label Distribution
| Label | Count | % |
|-------|-------|---|
{label_rows}

## Annotation Method
- Source: {source}
- Format: char-offset spans with labels
- Covers: ~{num_span_records:,} research paper abstracts from CS (arXiv)

## Size
- **{spans_file.name}:** {spans_mb:.2f} MB
- **{bio_file.name}:** {bio_mb:.2f} MB
- **Total:** {spans_mb + bio_mb:.2f} MB

## How to Use
1. **Training NER models:** Use `{bio_file.name}` for token-level classification (tokens + BIO tags)
2. **Evaluation:** Use `{spans_file.name}` for entity-level metrics (clean char-offset format)
3. **Reference:** Use `{labels_file.name}` for label mapping (id ↔ label strings)

## Linking to Training Runs
When training a model on this dataset, reference the artifact:
```python
artifact = wandb.use_artifact('project/{artifact_name}:vX')
```
This creates an audit trail: which model was trained on which dataset version.
"""
    if summary_file is not None:
        card += "\n---\n\n## Dataset Generation Summary\n\n"
        card += summary_file.read_text()

    card_path.write_text(card)


def register_dataset(
    data_dir: Path,
    project: str,
    artifact_name: str = "dataset-ner-3k",
    source: str = "DeepSeek LLM",
):
    """Analyze the dataset files in data_dir and log them as one W&B dataset
    artifact. Returns the artifact's full name/version string."""
    stem, spans_file, bio_file, labels_file, summary_file = find_stem_files(data_dir)
    card_file = data_dir / f"{stem}_DATASET_CARD.md"

    print(f"Initializing W&B (project: {project})")
    run = wandb.init(project=project, job_type="register-dataset")

    print("Analyzing dataset...")
    stats = analyze_dataset(spans_file, bio_file, labels_file)

    print("Writing dataset card...")
    write_dataset_card(card_file, stats, spans_file, bio_file, labels_file, artifact_name, source,
                        summary_file=summary_file)

    total_size_mb = sum(
        f.stat().st_size for f in data_dir.iterdir() if f.is_file()
    ) / (1024 ** 2)

    print(f"Creating artifact: {artifact_name}")
    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"NER dataset ({stats['num_span_records']} records, "
                     f"{len(stats['labels'])} labels) for research abstract entity extraction",
        metadata={
            "records": stats["num_span_records"],
            "total_entities": stats["total_entities"],
            "num_labels": len(stats["labels"]),
            "labels": stats["labels"],
            "source": source,
            "created": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
    )

    print(f"Adding all files in {data_dir} to artifact...")
    artifact.add_dir(str(data_dir))

    print("Logging artifact to W&B...")
    run.log_artifact(artifact)
    artifact.wait()  # block until upload finishes so artifact.version is populated below

    print("Logging statistics...")
    label_table = wandb.Table(
        columns=["label", "count", "percent"],
        data=[
            [label, count, round(count / stats["total_entities"] * 100, 2)]
            for label, count in stats["label_counts"].most_common()
        ],
    )
    run.log({
        "records": stats["num_span_records"],
        "total_entities": stats["total_entities"],
        "total_size_mb": round(total_size_mb, 2),
        "avg_entities_per_record": stats["total_entities"] / stats["num_span_records"],
        "label_distribution_table": label_table,
    })

    artifact_ref = f"{run.entity}/{project}/{artifact_name}:{artifact.version}"
    run.finish()

    print("\n" + "=" * 70)
    print("✓ Dataset registered successfully!")
    print(f"  Artifact URI: {artifact_ref}")
    print(f"  Records: {stats['num_span_records']}")
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Size: {total_size_mb:.2f} MB")
    print("=" * 70)
    print("\nNext step: Link this artifact to training runs:")
    print(f"  artifact = wandb.use_artifact('{artifact_ref}')")
    print("=" * 70)

    return artifact_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the NER dataset as a versioned W&B dataset artifact"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/Work/Experiments/citation-processor-ranker/data/SEKU/processed/ner_data_3k"),
        help="Directory containing ner_spans.jsonl, ner_bio.jsonl, ner_labels.json",
    )
    parser.add_argument("--project", default="PSM-IISER-NER", help="W&B project name")
    parser.add_argument("--artifact-name", default="dataset-ner-3k", help="W&B artifact name")
    parser.add_argument("--source", default="DeepSeek LLM", help="Annotation source, for the dataset card/metadata")

    args = parser.parse_args()

    register_dataset(
        data_dir=args.data_dir,
        project=args.project,
        artifact_name=args.artifact_name,
        source=args.source,
    )
