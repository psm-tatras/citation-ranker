"""
Convert NER span data (ner_spans.jsonl from build_ner_dataset.py) into
instruction/completion pairs for QLoRA SFT of Qwen2.5-Instruct.

Each training example becomes a chat with:
  - system : the annotator role
  - user   : the NER instruction + definitions + few-shot examples + abstract
  - assistant (TARGET) : {"entities":[{"text","label"}, ...]}

CRITICAL DESIGN CHOICES
-----------------------
1. The assistant target is EXACTLY the JSON your inference parser
   (slm_generate.extract_json / resolve_offsets) already consumes -- text +
   label, NO offsets (offsets are re-resolved at inference by string match).
   Train == inference format, so nothing downstream changes.
2. Entities are emitted in order of appearance (sorted by start), matching the
   prompt's "list in order of appearance" instruction.
3. Labels are read FROM THE DATA (and/or --labels), never hardcoded, so this
   is correct whether your set is the 8-label (v1) or 9-label (v2, +METRIC)
   scheme.

Outputs one JSON object per line (JSONL), each: {"messages":[...]} in the
format trl SFTTrainer expects with a chat template.

Usage
-----
    python -m training.convert_ner_format_to_qwen_sft \
        --input_dir data/training_data/synthetic_dgft_abstracts_6000 \
        --val_frac 0.05

--input_dir must contain <stem>_ner_spans.jsonl (and, if present,
<stem>_ner_labels.json), e.g. synthetic_dgft_abstracts_6000_ner_spans.jsonl.
The stem is inferred from that filename, and outputs are written back into
--input_dir as <stem>_train.jsonl and (if --val_frac > 0) <stem>_val.jsonl.
"""

import argparse
import glob
import json
import os
import random

from utils.prompts import NER_SYSTEM, NER_TRAIN_USER_TMPL


def find_stem_files(input_dir):
    """Locate <stem>_ner_spans.jsonl (and its sibling _ner_labels.json) in
    input_dir. Returns (stem, spans_path, labels_path_or_None)."""
    candidates = glob.glob(os.path.join(input_dir, "*_ner_spans.jsonl"))
    if not candidates:
        raise SystemExit(f"No *_ner_spans.jsonl file found in {input_dir!r}")
    if len(candidates) > 1:
        raise SystemExit(f"Multiple *_ner_spans.jsonl files found in "
                          f"{input_dir!r}: {candidates}")
    spans_path = candidates[0]
    stem = os.path.basename(spans_path)[: -len("_ner_spans.jsonl")]
    labels_path = os.path.join(input_dir, f"{stem}_ner_labels.json")
    if not os.path.exists(labels_path):
        labels_path = None
    return stem, spans_path, labels_path


def load_label_list(spans_path, labels_path):
    """Prefer the labels file; fall back to labels observed in the data.
    Returns the plain label list (no BIO prefixes)."""
    labels = []
    if labels_path:
        with open(labels_path, encoding="utf-8") as f:
            lab = json.load(f)
        labels = lab.get("labels") or []
    observed = set()
    with open(spans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for e in json.loads(line).get("entities", []):
                if e.get("label"):
                    observed.add(e["label"])
    # union, but keep the file's order first, then any extra observed labels
    for l in sorted(observed):
        if l not in labels:
            labels.append(l)
    return labels


def build_example(rec, labels_csv):
    text = rec["text"]
    ents = sorted(rec.get("entities", []), key=lambda e: e.get("start", 0))
    # target: text + label only, in order, exactly as the parser expects
    target = {"entities": [{"text": e["text"], "label": e["label"]} for e in ents]}
    user = NER_TRAIN_USER_TMPL.format(labels_csv=labels_csv, abstract=text)
    return {
        "messages": [
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant",
             "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing <stem>_ner_spans.jsonl "
                    "(and <stem>_ner_labels.json). Outputs are written "
                    "back into this directory.")
    ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_entities", type=int, default=0,
                    help="Skip abstracts with fewer than N entities (0 = keep all).")
    args = ap.parse_args()

    stem, spans_path, labels_path = find_stem_files(args.input_dir)
    out_path = os.path.join(args.input_dir, f"{stem}_train.jsonl")
    val_path = os.path.join(args.input_dir, f"{stem}_val.jsonl")
    print(f"Spans   : {spans_path}")
    print(f"Labels  : {labels_path}")

    labels = load_label_list(spans_path, labels_path)
    labels_csv = ", ".join(labels)
    print(f"Labels ({len(labels)}): {labels_csv}")

    records = []
    skipped_empty = 0
    with open(spans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("text"):
                continue
            n = len(rec.get("entities", []))
            if n < args.min_entities:
                skipped_empty += 1
                continue
            records.append(rec)

    print(f"Loaded {len(records)} usable records "
          f"(skipped {skipped_empty} below --min_entities).")
    if len(records) < 500:
        print("WARNING: fewer than 500 records. If this is meant to be the full "
              "10K training set, you have the wrong/sample file.")

    random.Random(args.seed).shuffle(records)

    val = []
    if args.val_frac > 0:
        k = int(len(records) * args.val_frac)
        val, records = records[:k], records[k:]

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(build_example(rec, labels_csv),
                               ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} train examples -> {out_path}")

    if val:
        with open(val_path, "w", encoding="utf-8") as f:
            for rec in val:
                f.write(json.dumps(build_example(rec, labels_csv),
                                   ensure_ascii=False) + "\n")
        print(f"Wrote {len(val)} val examples -> {val_path}")


if __name__ == "__main__":
    main()
