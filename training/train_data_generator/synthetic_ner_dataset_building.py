"""
End-to-end NER dataset builder for the abstract -> research-entity task.

Pipeline
--------
    abstracts CSV
        -> (LLM) extract entities per abstract        [raw_outputs.jsonl]
        -> parse + resolve offsets + validate
        -> HF token-classification dataset            [spans / bio / labels]

The LLM call uses langchain-openai (your original code). Raw responses are
cached incrementally so re-runs resume where they left off and never re-pay
for an abstract already processed.

Usage
-----
Extract + build (first run):
    python utils.py \
        --input_csv /Work/.../MixSub-SciHigh_train_FIRE.csv \
        --text_column abstract \
        --out_dir ner_data \
        --provider openai \
        -n 20

--provider is "openai" or "deepseek". Drop -n for the full set.
Model names come from OPENAI_MODEL / DEEPSEEK_MODEL in .env, not the CLI.

Re-parse cached outputs only (no API calls):
    python utils.py --input_csv ... --out_dir ner_data --skip_extract
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from utils.prompts import NER_PROMPT_v3 as NER_PROMPT
from dotenv import load_dotenv

load_dotenv("/Work/Experiments/seku-crd-citation-processor/.env")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# --------------------------------------------------------------------------
# Label scheme (locked 9-label research-role set)
# --------------------------------------------------------------------------
LABELS = ["PROBLEM", "OBJECTIVE", "METHOD", "CONTRIBUTION",
          "FINDING", "IMPROVEMENT", "DATASET", "METRIC", "KEY_CONCEPT"]
LABEL_SET = set(LABELS)


# --------------------------------------------------------------------------
# LLM extraction  (your original functions, hardened + lazy imports)
# --------------------------------------------------------------------------
def call_openai(prompt: str, model: str = OPENAI_MODEL,
                temperature: float = 0.0, max_retries: int = 3) -> str:
    from langchain_openai import ChatOpenAI  # lazy: parse-only runs don't need it
    llm = ChatOpenAI(model=model, temperature=temperature)
    last_err = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as ex:            # transient API/network errors
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def call_deepseek(prompt: str, model: str = DEEPSEEK_MODEL,
                   temperature: float = 0.0, max_retries: int = 3) -> str:
    """DeepSeek's API is OpenAI-compatible, so we reuse ChatOpenAI pointed
    at DeepSeek's endpoint instead of a separate client."""
    from langchain_openai import ChatOpenAI  # lazy: parse-only runs don't need it
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as ex:            # transient API/network errors
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def extract_entities(abstract: str, provider: str = "openai") -> str:
         # lazy: parse-only runs don't need it
    prompt = NER_PROMPT.replace("{{ABSTRACT}}", abstract)
    if provider == "openai":
        return call_openai(prompt)
    elif provider == "deepseek":
        return call_deepseek(prompt)
    raise ValueError(f"unknown provider: {provider!r}")


# --------------------------------------------------------------------------
# Async variants (for concurrent extraction over many abstracts)
# --------------------------------------------------------------------------
async def acall_openai(prompt: str, model: str = OPENAI_MODEL,
                        temperature: float = 0.0, max_retries: int = 3) -> str:
    from langchain_openai import ChatOpenAI  # lazy: parse-only runs don't need it
    llm = ChatOpenAI(model=model, temperature=temperature)
    last_err = None
    for attempt in range(max_retries):
        try:
            return (await llm.ainvoke(prompt)).content
        except Exception as ex:            # transient API/network errors
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


async def acall_deepseek(prompt: str, model: str = DEEPSEEK_MODEL,
                          temperature: float = 0.0, max_retries: int = 3) -> str:
    from langchain_openai import ChatOpenAI  # lazy: parse-only runs don't need it
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            return (await llm.ainvoke(prompt)).content
        except Exception as ex:            # transient API/network errors
            last_err = ex
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt+1}/{max_retries}): {ex} "
                  f"-> retrying in {wait}s", file=sys.stderr)
            await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


async def aextract_entities(abstract: str, provider: str = "openai") -> str:
    prompt = NER_PROMPT.replace("{{ABSTRACT}}", abstract)
    if provider == "openai":
        return await acall_openai(prompt)
    elif provider == "deepseek":
        return await acall_deepseek(prompt)
    raise ValueError(f"unknown provider: {provider!r}")


# --------------------------------------------------------------------------
# Parsing / offset resolution / BIO conversion
# --------------------------------------------------------------------------
def build_tag_maps():
    tags = ["O"]
    for lab in LABELS:
        tags += ["B-" + lab, "I-" + lab]
    id2label = {i: t for i, t in enumerate(tags)}
    label2id = {t: i for i, t in id2label.items()}
    return id2label, label2id


def extract_json(raw):
    """Pull the first balanced {...} JSON object out of a raw completion."""
    if isinstance(raw, dict):
        return raw
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
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


def _remove_overlaps(spans):
    """Resolve overlapping spans by keeping the longest and dropping any span
    that overlaps an already-kept (longer) one. This handles the common case
    where the LLM emits both an outer span and a nested inner span
    (e.g. 'different scenarios on a driving simulator' + 'driving simulator'):
    we keep the outer/longer span, which carries the paper-specific signal.
    Returns (kept_spans_sorted_by_start, n_dropped)."""
    order = sorted(spans, key=lambda x: (-(x["end"] - x["start"]), x["start"]))
    kept = []
    for sp in order:
        if any(sp["start"] < k["end"] and sp["end"] > k["start"] for k in kept):
            continue  # overlaps an already-kept longer span -> drop this one
        kept.append(sp)
    kept.sort(key=lambda x: x["start"])
    return kept, len(spans) - len(kept)


def resolve_offsets(text, entities):
    """Greedy left-to-right offset resolution, then longest-span-wins overlap
    resolution. Returns (spans, errors). Overlaps are RESOLVED (not fatal);
    a record is only unusable if nothing resolves at all."""
    resolved, errors = [], []
    cursor = 0
    for ent in entities:
        surface = ent.get("text", "")
        label = ent.get("label", "")
        if label not in LABEL_SET:
            errors.append(f"invalid label {label!r} ({surface!r})")
            continue
        if not surface:
            errors.append("empty surface string")
            continue
        idx = text.find(surface, cursor)
        if idx != -1:
            s, e = idx, idx + len(surface)
        else:
            pat = re.compile(r"\s+".join(re.escape(t) for t in surface.split()))
            m = pat.search(text, cursor) or pat.search(text)
            if m is None:
                errors.append(f"surface not found: {surface!r}")
                continue
            s, e = m.span()
        resolved.append({"start": s, "end": e, "label": label, "text": text[s:e]})
        cursor = max(cursor, e)  # keep cursor monotonic

    resolved, n_over = _remove_overlaps(resolved)
    if n_over:
        errors.append(f"resolved {n_over} overlapping span(s) (kept longest)")
    return resolved, errors


def word_spans(text):
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def to_bio(text, spans, label2id):
    words = word_spans(text)
    tags = ["O"] * len(words)
    for ent in spans:
        s, e, label = ent["start"], ent["end"], ent["label"]
        first = True
        for i, (ws, we, _) in enumerate(words):
            if ws < e and we > s:
                tags[i] = ("B-" if first else "I-") + label
                first = False
    tokens = [w for _, _, w in words]
    return tokens, tags, [label2id[t] for t in tags]


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def read_abstracts(path, text_column, limit=None):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_column not in reader.fieldnames:
            raise SystemExit(f"Column {text_column!r} not in CSV. "
                             f"Found: {reader.fieldnames}")
        for i, row in enumerate(reader):
            abs_text = (row.get(text_column) or "").strip()
            if abs_text:
                rows.append({"id": i, "abstract": abs_text})
            if limit and len(rows) >= limit:
                break
    return rows


def load_raw_cache(path):
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
# Pipeline
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--text_column", default="abstract")
    ap.add_argument("--out_dir", default="ner_data")
    ap.add_argument("--provider", choices=["openai", "deepseek"], default="openai",
                    help="Which LLM provider to use. Model names come from "
                    "OPENAI_MODEL / DEEPSEEK_MODEL in .env.")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="Process at most N abstracts (for cheap dry runs).")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max number of LLM extraction calls in flight at once.")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip LLM calls; only (re)parse cached raw outputs.")
    args = ap.parse_args()

    model_name = OPENAI_MODEL if args.provider == "openai" else DEEPSEEK_MODEL
    csv_stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_dir = os.path.join(args.out_dir, csv_stem)

    print(f"Provider           : {args.provider}")
    print(f"Model              : {model_name}")
    print(f"Input CSV          : {args.input_csv}")
    print(f"Output dir         : {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{csv_stem}_raw_outputs.jsonl")
    spans_path = os.path.join(out_dir, f"{csv_stem}_ner_spans.jsonl")
    bio_path = os.path.join(out_dir, f"{csv_stem}_ner_bio.jsonl")
    labels_path = os.path.join(out_dir, f"{csv_stem}_ner_labels.json")
    summary_path = os.path.join(out_dir, f"{csv_stem}_summary.md")

    id2label, label2id = build_tag_maps()

    print("Reading abstracts from CSV...")
    abstracts = read_abstracts(args.input_csv, args.text_column, args.limit)
    print(f"  {len(abstracts)} abstracts loaded.")

    cache = load_raw_cache(raw_path)

    # --- Extraction phase (append-only, resumable) ---
    if not args.skip_extract:
        from tqdm import tqdm

        todo = [r for r in abstracts if r["id"] not in cache]
        print(f"Extraction: {len(todo)} to do, {len(cache)} cached "
              f"(concurrency={args.concurrency}).")

        async def extract_one(row, semaphore):
            async with semaphore:
                try:
                    raw = await aextract_entities(row["abstract"], provider=args.provider)
                    return row["id"], row["abstract"], raw, None
                except Exception as ex:
                    return row["id"], row["abstract"], None, ex

        async def run_all():
            semaphore = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.ensure_future(extract_one(row, semaphore)) for row in todo]
            with open(raw_path, "a", encoding="utf-8") as fout:
                for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                                 desc="Extracting entities", unit="abstract"):
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
    print("\nParsing raw outputs into spans + BIO tags...")
    kept, dropped = [], 0
    label_counter = Counter()
    for rid, rec in sorted(cache.items()):
        try:
            obj = extract_json(rec["raw_output"])
            text = obj["text"]
            entities = obj.get("entities", [])
        except Exception as ex:
            print(f"[id {rid}] parse error: {ex}", file=sys.stderr)
            dropped += 1
            continue

        spans, errors = resolve_offsets(text, entities)
        for e in errors:
            print(f"[id {rid}] {e}", file=sys.stderr)
        if not spans:
            dropped += 1
            continue

        tokens, tags, tag_ids = to_bio(text, spans, label2id)
        for sp in spans:
            label_counter[sp["label"]] += 1
        kept.append({
            "id": rid,
            "spans": {"text": text, "entities": spans},
            "bio": {"tokens": tokens, "ner_tags": tag_ids, "ner_tags_str": tags},
        })

    print(f"Parsed {len(kept)} kept, {dropped} dropped.")

    print("Writing output files...")
    with open(spans_path, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k["spans"], ensure_ascii=False) + "\n")
    with open(bio_path, "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k["bio"], ensure_ascii=False) + "\n")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id,
                   "labels": LABELS}, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"abstracts considered : {len(abstracts)}")
    print(f"raw cached           : {len(cache)}")
    print(f"kept                 : {len(kept)}")
    print(f"dropped              : {dropped}")
    print(f"entities             : {sum(label_counter.values())}")
    print("per-label            :")
    for lab in LABELS:
        print(f"    {lab:<14} {label_counter.get(lab, 0)}")
    print(f"\nWrote: {spans_path}, {bio_path}, {labels_path}")

    n_entities = sum(label_counter.values())
    summary_md = "\n".join([
        f"# {csv_stem}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Source",
        "",
        f"- Input CSV: `{args.input_csv}`",
        f"- Text column: `{args.text_column}`",
        f"- Provider / model: `{args.provider}` / `{model_name}`",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---|",
        f"| abstracts considered | {len(abstracts)} |",
        f"| raw cached | {len(cache)} |",
        f"| kept | {len(kept)} |",
        f"| dropped | {dropped} |",
        f"| entities | {n_entities} |",
        "",
        "## Per-label counts",
        "",
        "| label | count |",
        "|---|---|",
        *[f"| {lab} | {label_counter.get(lab, 0)} |" for lab in LABELS],
        "",
        "## Files",
        "",
        f"- `{os.path.basename(spans_path)}`",
        f"- `{os.path.basename(bio_path)}`",
        f"- `{os.path.basename(labels_path)}`",
        "",
    ])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()