"""
Narrative extraction dataset builder for research abstract decomposition.

Label scheme is selected with --version, matching a NARRATIVE_LABELS_v*/
NARRATIVE_SYSTEM_v*/NARRATIVE_USER_TEMPLATE_v* set in utils/prompts.py:
    v1 (default): TASK, LIMITATION, CONTRIBUTION
    v2:           TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT

Pipeline
--------
    abstracts CSV
        -> (DeepSeek LLM) extract the selected version's labels per abstract
        -> parse + validate JSON
        -> Qwen SFT chat format (messages)                    [train.jsonl / val.jsonl]

The LLM call uses langchain-openai (DeepSeek endpoint). Raw responses are
cached incrementally (one cache per --version, so switching versions never
resumes from another version's incompatible cache) so re-runs resume where
they left off and never re-pay for an abstract already processed.

OUTPUT FORMAT (direct to Qwen SFT, no conversion step needed)
-------------------------------------------------------------
Each line in {stem}_narrative_train.jsonl is:
{
  "messages": [
    {"role": "system", "content": "You are an expert at decomposing..."},
    {"role": "user", "content": "Extract <labels for this version> from: [abstract]"},
    {"role": "assistant", "content": "{\"TASK\": \"...\", ...}"}
  ]
}

This skips the separate conversion step entirely — output is ready for train_qlora.py.

Usage
-----
Extract + build (first run):
    python training/train_data_generator/synthetic_narrative_labelling_dataset_building.py \\
        --input_csv /Work/.../abstracts.csv \\
        --text_column abstract \\
        --out_dir narrative_data \\
        --version v2 \\
        -n 100

Re-parse cached outputs only (no API calls):
    python ... --input_csv ... --out_dir narrative_data --version v2 --skip_extract
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from utils.prompts import NARRATIVE_VERSIONS
from dotenv import load_dotenv

load_dotenv("/Work/Experiments/seku-crd-citation-processor/.env")

DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# --------------------------------------------------------------------------
# Label scheme -- selected by --version, resolved against the single shared
# NARRATIVE_VERSIONS registry in utils/prompts.py (prediction/predict_narrative.py
# uses the same registry, so generation and inference can never disagree on
# which system/template/labels go together). Defaults to v1 below so
# importing this module doesn't require a version to already be chosen;
# main() applies the actual --version choice to these globals before
# extraction/parsing runs.
# --------------------------------------------------------------------------
NARRATIVE_VERSION = "v1"
SYSTEM_PROMPT = NARRATIVE_VERSIONS["v1"]["system"]
USER_TEMPLATE = NARRATIVE_VERSIONS["v1"]["template"]
LABELS = list(NARRATIVE_VERSIONS["v1"]["labels"])
LABEL_SET = set(LABELS)
OPTIONAL_LABELS = set()


# --------------------------------------------------------------------------
# LLM extraction (DeepSeek only)
# --------------------------------------------------------------------------
def call_deepseek(messages, model: str = DEEPSEEK_MODEL,
                  temperature: float = 0.0, max_retries: int = 3) -> str:
    """Call DeepSeek API (OpenAI-compatible endpoint)."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages).content
        except Exception as ex:
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


async def acall_deepseek(messages, model: str = DEEPSEEK_MODEL,
                         temperature: float = 0.0, max_retries: int = 3) -> str:
    """Async call to DeepSeek API."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            return (await llm.ainvoke(messages)).content
        except Exception as ex:
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def extract_narrative(abstract: str) -> str:
    """Sync wrapper for DeepSeek extraction."""
    messages = [("system", SYSTEM_PROMPT),
                ("human", USER_TEMPLATE.format(abstract=abstract))]
    return call_deepseek(messages)


async def aextract_narrative(abstract: str) -> str:
    """Async wrapper for DeepSeek extraction."""
    messages = [("system", SYSTEM_PROMPT),
                ("human", USER_TEMPLATE.format(abstract=abstract))]
    return await acall_deepseek(messages)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def extract_json(raw):
    """Pull the first balanced {...} JSON object out of raw completion."""
    if isinstance(raw, dict):
        return raw
    raw = raw.strip()
    # Try to find code fence first
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    # Find first balanced JSON object
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError("unbalanced JSON braces")


def validate_narrative(obj):
    """Validate that the parsed object has EXACTLY the keys in LABELS, all
    string-valued. Labels in OPTIONAL_LABELS (v2: LIMITATION/METHOD/RESULT)
    are allowed to be "" -- the prompt itself instructs the model to leave
    those empty rather than pad them for papers that don't state them.
    Returns (valid, errors_list)."""
    errors = []
    if not isinstance(obj, dict):
        return False, ["output is not a dict"]

    extra = set(obj.keys()) - LABEL_SET
    if extra:
        errors.append(f"unexpected extra field(s): {sorted(extra)}")

    for label in LABELS:
        if label not in obj:
            errors.append(f"missing required field: {label}")
        elif not isinstance(obj[label], str):
            errors.append(f"{label} is not a string")
        elif not obj[label] and label not in OPTIONAL_LABELS:
            errors.append(f"empty value for {label}")

    return len(errors) == 0, errors


# --------------------------------------------------------------------------
# CSV / Cache
# --------------------------------------------------------------------------
def read_abstracts(csv_path, text_column="abstract", limit=None):
    """Read abstracts from CSV. Returns list of {"id": i, "abstract": text}."""
    import csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            abs_text = row.get(text_column, "").strip()
            if abs_text:
                rows.append({"id": i, "abstract": abs_text})
            if limit and len(rows) >= limit:
                break
    return rows


def load_raw_cache(path):
    """Load cached raw outputs. Returns {id: record}."""
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["id"]] = rec
                except Exception:
                    pass
    return cache


# --------------------------------------------------------------------------
# Qwen SFT Chat Format Generation
# --------------------------------------------------------------------------
def build_sft_example(abstract, narrative_obj):
    """Build a chat message for Qwen SFT training.
    
    Args:
        abstract: the input abstract text
        narrative_obj: parsed dict with TASK, LIMITATION, CONTRIBUTION
    
    Returns:
        dict with "messages" key (trl SFTTrainer format)
    """
    # Assistant message: JSON with extracted narratives (text only, no offsets)
    assistant_content = json.dumps(narrative_obj, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(abstract=abstract)},
            {"role": "assistant", "content": assistant_content}
        ]
    }


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main():
    global NARRATIVE_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, LABELS, LABEL_SET, OPTIONAL_LABELS

    ap = argparse.ArgumentParser(
        description="Generate narrative extraction training data for Qwen, "
                    "output directly in SFT chat format (no conversion step needed)."
    )
    ap.add_argument("--input_csv", required=True,
                    help="CSV file containing abstracts")
    ap.add_argument("--text_column", default="abstract",
                    help="Column name containing abstract text")
    ap.add_argument("--out_dir", default="narrative_data",
                    help="Output directory for train/val JSONL files")
    ap.add_argument("--version", choices=sorted(NARRATIVE_VERSIONS), default="v1",
                    help="Narrative label scheme to extract, from utils/prompts.py "
                         "(v1: TASK/LIMITATION/CONTRIBUTION; "
                         "v2: + METHOD/RESULT). Default: v1")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="Process at most N abstracts (for cheap dry runs)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max number of concurrent LLM calls")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip LLM calls; only (re)parse cached raw outputs")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of data to use as validation set")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split")
    args = ap.parse_args()

    NARRATIVE_VERSION = args.version
    version_cfg = NARRATIVE_VERSIONS[NARRATIVE_VERSION]
    SYSTEM_PROMPT = version_cfg["system"]
    USER_TEMPLATE = version_cfg["template"]
    LABELS = list(version_cfg["labels"])
    LABEL_SET = set(LABELS)
    OPTIONAL_LABELS = version_cfg["optional"]

    csv_stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    # version-scoped so v1/v2 runs against the same CSV/out_dir never share a
    # raw-output cache -- their prompts (and expected JSON shape) differ, so
    # resuming a v2 run from a v1 cache would silently produce incomplete records.
    out_dir = os.path.join(args.out_dir, f"{csv_stem}_{NARRATIVE_VERSION}")
    resuming = os.path.isdir(out_dir)

    print(f"Provider           : deepseek")
    print(f"Model              : {DEEPSEEK_MODEL}")
    print(f"Input CSV          : {args.input_csv}")
    print(f"Narrative version  : {NARRATIVE_VERSION} ({version_cfg['description']})")
    print(f"Output dir         : {out_dir}")
    print(f"Labels             : {', '.join(LABELS)}")

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{csv_stem}_raw_outputs.jsonl")
    train_path = os.path.join(out_dir, f"{csv_stem}_narrative_train.jsonl")
    val_path = os.path.join(out_dir, f"{csv_stem}_narrative_val.jsonl")
    labels_path = os.path.join(out_dir, f"{csv_stem}_narrative_labels.json")
    summary_path = os.path.join(out_dir, f"{csv_stem}_narrative_summary.md")

    print("\nReading abstracts from CSV...")
    abstracts = read_abstracts(args.input_csv, args.text_column, args.limit)
    print(f"  {len(abstracts)} abstracts loaded.")

    cache = load_raw_cache(raw_path)
    if resuming and cache:
        print(f"\nOutput dir already exists -- resuming: {len(cache)} abstract(s) "
              f"already extracted (cached in {os.path.basename(raw_path)}) will be skipped.")

    # --- Extraction phase (append-only, resumable) ---
    if not args.skip_extract:
        from tqdm import tqdm

        todo = [r for r in abstracts if r["id"] not in cache]
        print(f"\nExtraction: {len(todo)} to do, {len(cache)} cached "
              f"(concurrency={args.concurrency}).")

        async def extract_one(row, semaphore):
            async with semaphore:
                try:
                    raw = await aextract_narrative(row["abstract"])
                    return row["id"], row["abstract"], raw, None
                except Exception as ex:
                    return row["id"], row["abstract"], None, ex

        async def run_all():
            semaphore = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.ensure_future(extract_one(row, semaphore)) for row in todo]
            with open(raw_path, "a", encoding="utf-8") as fout:
                for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                                desc="Extracting narratives", unit="abstract"):
                    rid, abstract, raw, err = await fut
                    if err is not None:
                        print(f"  [id {rid}] extraction failed: {err}",
                              file=sys.stderr)
                        continue
                    rec = {"id": rid, "abstract": abstract, "raw_output": raw}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    cache[rid] = rec

        asyncio.run(run_all())
        print("Extraction complete.")
    else:
        print("Skipping extraction (--skip_extract); parsing cached outputs only.")

    # --- Parse phase ---
    print("\nParsing raw outputs into narrative JSON...")
    kept, dropped = [], 0
    label_counter = Counter()

    for rid, rec in sorted(cache.items()):
        try:
            obj = extract_json(rec["raw_output"])
        except Exception as ex:
            print(f"[id {rid}] JSON parse error: {ex}", file=sys.stderr)
            dropped += 1
            continue

        # Validate all three fields are present and non-empty
        valid, errors = validate_narrative(obj)
        if not valid:
            for e in errors:
                print(f"[id {rid}] validation error: {e}", file=sys.stderr)
            dropped += 1
            continue

        # Count labels (all three must be present to reach here)
        for label in LABELS:
            label_counter[label] += 1

        kept.append({
            "id": rid,
            "abstract": rec["abstract"],
            "narrative": obj
        })

    print(f"Parsed {len(kept)} kept, {dropped} dropped.")

    # --- Train/val split ---
    import random
    random.Random(args.seed).shuffle(kept)

    val_count = int(len(kept) * args.val_frac)
    val_set = kept[:val_count]
    train_set = kept[val_count:]

    # --- Build Qwen SFT format ---
    print("\nBuilding Qwen SFT chat format...")
    train_examples = []
    val_examples = []

    for rec in train_set:
        example = build_sft_example(rec["abstract"], rec["narrative"])
        train_examples.append(example)

    for rec in val_set:
        example = build_sft_example(rec["abstract"], rec["narrative"])
        val_examples.append(example)

    # --- Write output files ---
    print("Writing output files...")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_examples)} train examples -> {train_path}")

    if val_examples:
        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_examples)} val examples -> {val_path}")

    # --- Write labels metadata ---
    labels_metadata = {
        "labels": LABELS,
        "version": NARRATIVE_VERSION,
        "description": version_cfg["description"],
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_metadata, f, indent=2)

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"abstracts considered : {len(abstracts)}")
    print(f"raw cached           : {len(cache)}")
    print(f"kept (valid)         : {len(kept)}")
    print(f"dropped              : {dropped}")
    print(f"train set            : {len(train_set)}")
    print(f"val set              : {len(val_set)}")
    print(f"entities (labels)    : {sum(label_counter.values())}")
    print("per-label            :")
    for lab in LABELS:
        print(f"    {lab:<15} {label_counter.get(lab, 0)}")
    print(f"\nWrote: {train_path}, {val_path}, {labels_path}")

    summary_md = "\n".join([
        f"# {csv_stem} (Narrative Extraction {NARRATIVE_VERSION})",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Source",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Text column: `{args.text_column}`",
        f"- Provider / model: `deepseek` / `{DEEPSEEK_MODEL}`",
        "",
        f"## Labels ({version_cfg['description']})",
        "",
        *[f"- **{lab}**" for lab in LABELS],
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---|",
        f"| abstracts considered | {len(abstracts)} |",
        f"| raw cached | {len(cache)} |",
        f"| kept (valid) | {len(kept)} |",
        f"| dropped | {dropped} |",
        f"| train set | {len(train_set)} |",
        f"| val set | {len(val_set)} |",
        f"| entities (per-label) | {sum(label_counter.values())} |",
        "",
        "## Per-label counts",
        "",
        "| label | count |",
        "|---|---|",
        *[f"| {lab} | {label_counter.get(lab, 0)} |" for lab in LABELS],
        "",
        "## Output Format",
        "",
        "Direct Qwen SFT chat format (no conversion step needed):",
        "```json",
        '{',
        '  "messages": [',
        '    {"role": "system", "content": "..."},',
        f'    {{"role": "user", "content": "Extract {", ".join(LABELS)} from: [abstract]"}},',
        '    {"role": "assistant", "content": "' +
            json.dumps(json.dumps({lab: "..." for lab in LABELS})).strip('"') + '"}',
        "  ]",
        "}",
        "```",
        "",
        "## Files",
        "",
        f"- `{os.path.basename(train_path)}` — training set (ready for train_qlora.py)",
        f"- `{os.path.basename(val_path)}` — validation set",
        f"- `{os.path.basename(labels_path)}` — label metadata",
        "",
    ])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()