"""
Layer-2 citation-relation dataset builder: for every citation CONTEXT of every
reference R of a main paper M, classify what role R plays for M --
FUNCTION / M_ASPECT / POLARITY / EVIDENCE / R_MATCH / CONFIDENCE.

Label scheme is selected with --version, matching a LAYER_2_VERSIONS entry
(utils/prompts.py) -- currently just "v1". prediction-side inference isn't
written yet; this script only builds the DeepSeek-labeled training data.

Unit of work is ONE CITATION CONTEXT, not one reference: a reference cited 3
times in M produces 3 records, one per citing sentence.

Pipeline
--------
    <input_dir>/*.json  (one file per M: {M_id: {title, abstract, narrative,
                          references: {R_id: {title, abstract, narrative,
                          citation_context: [...]}}}}, i.e. the shape
                          utils/citation_context.py's find_citation_context()
                          writes)
        -> build one record per citation context (skip/count refs with no
           context, and M/R whose narrative lacks TASK or CONTRIBUTION)
        -> preprocess (title whitespace, narrative rendering, section
           cleanup, before/after word-capping, co_cited detection)
        -> (DeepSeek LLM) classify FUNCTION/M_ASPECT/POLARITY/EVIDENCE/
           R_MATCH/CONFIDENCE per record
        -> parse + validate (exact keys, allowed value sets, the
           POLARITY<->COMPARISON rule; EVIDENCE substring-checked but never
           dropped on failure -- stored as evidence_verified)
        -> canonical labeled JSONL + Qwen SFT chat format (train/val, split
           BY MAIN PAPER so no M straddles both sets)

This module reuses call_deepseek/acall_deepseek/extract_json from
synthetic_narrative_labelling_dataset_building.py rather than duplicating
them -- that script is not modified.

The LLM call uses langchain-openai (DeepSeek endpoint). Raw responses are
cached incrementally, keyed by (record_id, input_hash) -- a record only
counts as cached if BOTH match, so a preprocessing change (e.g. a different
--max_side_words) naturally invalidates stale cache entries and re-queries,
rather than silently reusing output rendered from different input text.

Usage
-----
Extract + build (first run). -n / --nb_input_json selects how many
main-paper JSON files to read (-1 = all, default); every reference and
every citation context of each selected file is processed -- there is no
record-level limit:
    python -m training.train_data_generator.synthetic_context_classification_dataset_building \\
        --input_dir data/citation_contexts \\
        --out_dir layer2_data \\
        --version v1 \\
        -n 5

Re-parse cached outputs only and render the SFT train/val files with the lean
Qwen prompt (no API calls). --version must be the LAYER_2 version the cached
labels were extracted with (input hashes depend on it); --sft_version selects a
CITATION_CLASSIFIER_SLM_VERSIONS entry (utils/prompts.py):
    python -m training.train_data_generator.synthetic_context_classification_dataset_building \\
        --input_dir data/citation_contexts --out_dir layer2_data --version v2 \\
        --skip_extract --sft_version v1 --require_evidence --require_evidence_len

SFT targets contain only the SLM version's output_keys (CONFIDENCE is dropped;
it is used only for --min_confidence filtering). Train/val files are named
*_layer2_{train,val}_slm_<sft_version>.jsonl so a rebuild never overwrites
files rendered with another prompt.
"""

import argparse
import glob as glob_module
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

from utils.prompts import LAYER_2_VERSIONS, CITATION_CLASSIFIER_SLM_VERSIONS
from dotenv import load_dotenv

from training.train_data_generator.synthetic_narrative_labelling_dataset_building import (
    call_deepseek, acall_deepseek, extract_json, DEEPSEEK_MODEL,
)

load_dotenv("/Work/Experiments/seku-crd-citation-processor/.env")

LABEL_KEYS = ("FUNCTION", "M_ASPECT", "POLARITY", "EVIDENCE", "R_MATCH", "CONFIDENCE")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
MARKER_RE = re.compile(r"\{\{(?:cite|figure|table):[^}]*\}\}")
WS_RE = re.compile(r"\s+")

# --------------------------------------------------------------------------
# Label scheme -- selected by --version, resolved against the shared
# LAYER_2_VERSIONS registry in utils/prompts.py. Defaults to v1 below so
# importing this module doesn't require a version to already be chosen;
# main() applies the actual --version choice to these globals before
# extraction/parsing runs.
# --------------------------------------------------------------------------
LAYER_2_VERSION = "v1"
VERSION_CFG = LAYER_2_VERSIONS["v1"]


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------
def collapse_ws(text):
    """Collapse any whitespace run (incl. the "\\n  " arXiv titles/sentences
    contain) into a single space."""
    return WS_RE.sub(" ", text or "").strip()


def clean_section(section):
    """Strip leftover {{cite:...}}/{{figure:...}}/{{table:...}} markers and
    whitespace from a section heading; "" becomes "(none)"."""
    s = MARKER_RE.sub("", section or "")
    s = collapse_ws(s)
    return s if s else "(none)"


def clean_narrative(narrative):
    """Replace each empty/missing narrative field with "(not stated)"."""
    narrative = narrative or {}
    return {k: (narrative.get(k) or "(not stated)") for k in ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")}


def has_required_narrative(narrative):
    """TASK and CONTRIBUTION must both be present and non-empty."""
    narrative = narrative or {}
    return bool(narrative.get("TASK")) and bool(narrative.get("CONTRIBUTION"))


def cap_side(words, side, max_words):
    """Keep the LAST max_words of `words` for side="before", or the FIRST
    max_words for side="after"; mark truncation with a leading/trailing "…"."""
    if len(words) <= max_words:
        return " ".join(words)
    if side == "before":
        return "… " + " ".join(words[-max_words:])
    return " ".join(words[:max_words]) + " …"


def build_context_strings(before_sentences, after_sentences, max_side_words):
    """Join before/after sentence lists with " ", then cap each side at
    max_side_words. Returns (before_text, after_text, was_truncated)."""
    before_words = " ".join(before_sentences or []).split()
    after_words = " ".join(after_sentences or []).split()
    truncated = len(before_words) > max_side_words or len(after_words) > max_side_words
    return (cap_side(before_words, "before", max_side_words),
            cap_side(after_words, "after", max_side_words),
            truncated)


def render_narrative_block(narrative, narrative_block_template):
    return narrative_block_template.format(**narrative)


def render_user_message(record, version_cfg):
    """Re-render the full user message from a record's RAW/preprocessed
    pieces against `version_cfg` -- used both at extraction time (--version)
    and at SFT-build time (--sft_version), so a different --sft_version can
    reproduce the exact same preprocessed content under a different
    template/narrative-block rendering without re-querying DeepSeek."""
    m_block = render_narrative_block(record["m_narrative"], version_cfg["narrative_block"])
    r_block = render_narrative_block(record["r_narrative"], version_cfg["narrative_block"])
    return version_cfg["template"].format(
        m_title=record["m_title"], m_narrative=m_block,
        r_title=record["r_title"], r_narrative=r_block,
        section=record["section"], co_cited=record["co_cited"],
        before=record["before"], center=record["center"], after=record["after"],
    )


def input_hash(system, user):
    return hashlib.sha1((system + "\x00" + user).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Record building (one record per citation context)
# --------------------------------------------------------------------------
def build_records(input_dir, glob_pattern, nb_input_json, max_side_words, version_cfg):
    """Walk --glob files in --input_dir (each one main paper M) and build one
    record per citation context of every reference R with a usable narrative
    and non-empty citation_context. Returns (records, file_stats, files_used)
    where file_stats is {filename: Counter} for the summary tables.

    nb_input_json: -1 processes every matching file; else only the first N
    (sorted, so this is deterministic) are read."""
    files = sorted(glob_module.glob(os.path.join(input_dir, glob_pattern)))
    if nb_input_json is not None and nb_input_json != -1:
        files = files[:nb_input_json]

    records = []
    seen_ids = set()
    file_stats = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        stats = Counter()
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[{fname}] could not read/parse file: {e}", file=sys.stderr)
            stats["file_unreadable"] = 1
            file_stats[fname] = stats
            continue

        if not data:
            stats["file_empty"] = 1
            file_stats[fname] = stats
            continue

        m_id = next(iter(data))
        m_data = data[m_id] or {}
        m_narrative_raw = m_data.get("narrative") or {}

        if not has_required_narrative(m_narrative_raw):
            stats["m_missing_narrative"] = 1
            file_stats[fname] = stats
            continue

        m_title = collapse_ws(m_data.get("title", ""))
        m_narrative = clean_narrative(m_narrative_raw)

        refs = m_data.get("references") or {}
        stats["n_refs"] = len(refs)

        # index every (section, sentence_index) pair across ALL references of
        # this M, so co_cited can be computed for each context below
        key_to_refs = defaultdict(set)
        for rid, rdata in refs.items():
            for ctx in (rdata.get("citation_context") or []):
                key_to_refs[(ctx.get("section"), ctx.get("sentence_index"))].add(rid)

        for r_id, rdata in refs.items():
            ctx_list = rdata.get("citation_context") or []
            if not ctx_list:
                stats["refs_no_context"] += 1
                continue

            r_narrative_raw = rdata.get("narrative") or {}
            if not has_required_narrative(r_narrative_raw):
                stats["refs_missing_narrative"] += 1
                continue

            r_title = collapse_ws(rdata.get("title", ""))
            r_narrative = clean_narrative(r_narrative_raw)

            for ctx in ctx_list:
                stats["contexts_seen"] += 1
                section_raw = ctx.get("section", "")
                sentence_index = ctx.get("sentence_index")
                ref_id = ctx.get("ref_id")
                record_id = f"{m_id}|{r_id}|{ref_id}|{sentence_index}|{section_raw}"
                if record_id in seen_ids:
                    stats["duplicates"] += 1
                    continue
                seen_ids.add(record_id)

                section = clean_section(section_raw)
                center = collapse_ws(ctx.get("center", ""))
                before, after, truncated = build_context_strings(
                    ctx.get("before"), ctx.get("after"), max_side_words)
                if truncated:
                    stats["truncated"] += 1

                co_cited = "yes" if len(key_to_refs.get((section_raw, sentence_index), ())) > 1 else "no"
                stats["co_cited_yes" if co_cited == "yes" else "co_cited_no"] += 1

                record = {
                    "id": record_id,
                    "m_id": m_id, "r_id": r_id, "ref_id": ref_id,
                    "section": section, "sentence_index": sentence_index,
                    "co_cited": co_cited, "co_cited_method": "corpus_overlap",
                    "source_file": fname,
                    "m_title": m_title, "m_narrative": m_narrative,
                    "r_title": r_title, "r_narrative": r_narrative,
                    "before": before, "center": center, "after": after,
                }
                user_msg = render_user_message(record, version_cfg)
                record["input_hash"] = input_hash(version_cfg["system"], user_msg)
                stats["records_built"] += 1
                records.append(record)

        file_stats[fname] = stats

    return records, file_stats, files


# --------------------------------------------------------------------------
# Parsing / validation
# --------------------------------------------------------------------------
def validate_label(obj, version_cfg):
    """Validate the parsed object has exactly LABEL_KEYS, all strings, each
    value in its registry-defined allowed set, and the POLARITY<->COMPARISON
    rule. Returns (valid, errors_list)."""
    errors = []
    if not isinstance(obj, dict):
        return False, ["output is not a dict"]

    extra = set(obj.keys()) - set(LABEL_KEYS)
    if extra:
        errors.append(f"unexpected extra field(s): {sorted(extra)}")

    for key in LABEL_KEYS:
        if key not in obj:
            errors.append(f"missing required field: {key}")
        elif not isinstance(obj[key], str):
            errors.append(f"{key} is not a string")

    function = obj.get("FUNCTION")
    if isinstance(function, str) and function not in version_cfg["functions"]:
        errors.append(f"FUNCTION {function!r} not in allowed set {version_cfg['functions']}")

    m_aspect = obj.get("M_ASPECT")
    if isinstance(m_aspect, str) and m_aspect not in version_cfg["m_aspects"]:
        errors.append(f"M_ASPECT {m_aspect!r} not in allowed set {version_cfg['m_aspects']}")

    polarity = obj.get("POLARITY")
    if isinstance(polarity, str):
        if polarity not in version_cfg["polarities"]:
            errors.append(f"POLARITY {polarity!r} not in allowed set {version_cfg['polarities']}")
        elif isinstance(function, str):
            if function == "COMPARISON" and polarity == "":
                errors.append("POLARITY must be non-empty when FUNCTION is COMPARISON")
            elif function != "COMPARISON" and polarity != "":
                errors.append(f"POLARITY must be empty when FUNCTION is {function!r}")

    r_match = obj.get("R_MATCH")
    if isinstance(r_match, str) and r_match not in version_cfg["r_match"]:
        errors.append(f"R_MATCH {r_match!r} not in allowed set {version_cfg['r_match']}")

    confidence = obj.get("CONFIDENCE")
    if isinstance(confidence, str) and confidence not in version_cfg["confidence"]:
        errors.append(f"CONFIDENCE {confidence!r} not in allowed set {version_cfg['confidence']}")

    evidence = obj.get("EVIDENCE")
    if isinstance(evidence, str) and not evidence.strip():
        errors.append("EVIDENCE is empty")

    return len(errors) == 0, errors


def check_evidence(evidence, before, center, after):
    """Whitespace-normalized, case-insensitive substring check of EVIDENCE
    against the rendered context. EVIDENCE containing the [Citation] marker
    is unverified (the marker is not text of M). Never causes a drop --
    callers store the bool result as evidence_verified."""
    haystack = collapse_ws(f"{before} {center} {after}").lower()
    needle = collapse_ws(evidence).lower()
    if "[citation]" in needle:
        return False
    return bool(needle) and needle in haystack


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def load_raw_cache(path):
    """Load cached raw outputs. Returns {record_id: {"input_hash", "raw_output"}}."""
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["record_id"]] = {"input_hash": rec["input_hash"], "raw_output": rec["raw_output"]}
                except Exception:
                    pass
    return cache


# --------------------------------------------------------------------------
# Qwen SFT chat format
# --------------------------------------------------------------------------
def build_sft_example(record, label_obj, sft_version_cfg):
    """Build a chat message for Qwen SFT training from a labeled record's
    raw/preprocessed inputs (`record["inputs"]`), re-rendered fresh against
    `sft_version_cfg` (a CITATION_CLASSIFIER_SLM_VERSIONS entry) -- so the
    lean Qwen prompt can be rendered without re-querying DeepSeek for the
    label itself. The assistant target holds only sft_version_cfg's
    output_keys (CONFIDENCE excluded)."""
    inputs = record["inputs"]
    m_block = render_narrative_block(inputs["m_narrative"], sft_version_cfg["narrative_block"])
    r_block = render_narrative_block(inputs["r_narrative"], sft_version_cfg["narrative_block"])
    user_msg = sft_version_cfg["template"].format(
        m_title=inputs["m_title"], m_narrative=m_block,
        r_title=inputs["r_title"], r_narrative=r_block,
        section=inputs["section"], co_cited=record["co_cited"],
        before=inputs["before"], center=inputs["center"], after=inputs["after"],
    )
    target = {k: label_obj[k] for k in sft_version_cfg["output_keys"]}
    assistant_content = json.dumps(target, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": sft_version_cfg["system"]},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def passes_sft_filters(labeled_record, min_confidence, require_evidence, require_evidence_len):
    if CONFIDENCE_ORDER[labeled_record["label"]["CONFIDENCE"]] < CONFIDENCE_ORDER[min_confidence]:
        return False
    if require_evidence and not labeled_record["evidence_verified"]:
        return False
    if require_evidence_len and not labeled_record["evidence_len_ok"]:
        return False
    return True  # R_MATCH="no" rows are always kept -- the model must learn to flag them


def group_split_by_m_id(labeled_records, val_frac, seed):
    """Split BY MAIN PAPER (m_id) so no M appears in both sets."""
    m_ids = sorted({r["m_id"] for r in labeled_records})
    random.Random(seed).shuffle(m_ids)
    val_m_ids = set(m_ids[:int(len(m_ids) * val_frac)])
    train, val = [], []
    for r in labeled_records:
        (val if r["m_id"] in val_m_ids else train).append(r)
    return train, val


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main():
    global LAYER_2_VERSION, VERSION_CFG

    ap = argparse.ArgumentParser(
        description="Generate Layer-2 citation-relation-function training data for Qwen "
                    "(one record per citation context), output directly in SFT chat format."
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory of *.json files (find_citation_context() output shape)")
    ap.add_argument("--glob", default="*.json", help="Glob pattern within --input_dir")
    ap.add_argument("-n", "--nb_input_json", type=int, default=-1,
                    help="Number of main-paper JSON files to process; -1 (default) processes "
                         "every file matching --glob, else only the first N (sorted). ALL "
                         "references and ALL citation contexts of each selected file are "
                         "processed -- there is no record-level limit")
    ap.add_argument("--out_dir", default="layer2_data",
                    help="Output directory root for the version-scoped output subdirectory")
    ap.add_argument("--version", choices=sorted(LAYER_2_VERSIONS), default="v1",
                    help="Layer-2 label scheme to extract, from utils/prompts.py. Default: v1")
    ap.add_argument("--sft_version", choices=sorted(CITATION_CLASSIFIER_SLM_VERSIONS), default="v1",
                    help="CITATION_CLASSIFIER_SLM_VERSIONS entry (utils/prompts.py) to render SFT "
                         "train/val messages with -- must be the prompt the classifier is trained "
                         "and run with. Combine with --skip_extract to rebuild without DeepSeek calls.")
    ap.add_argument("--max_side_words", type=int, default=40,
                    help="Max words kept on each side of the citing sentence "
                         "(last N of before, first N of after)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max number of concurrent LLM calls")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip LLM calls; only (re)parse cached raw outputs")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="Fraction of MAIN PAPERS (not records) to hold out for validation")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/val group split")
    ap.add_argument("--min_confidence", choices=["low", "medium", "high"], default="low",
                    help="Drop SFT rows below this CONFIDENCE (default low = keep all)")
    ap.add_argument("--require_evidence", action="store_true",
                    help="Drop SFT rows where evidence_verified is false (default: keep them)")
    ap.add_argument("--require_evidence_len", action="store_true",
                    help="Drop SFT rows where evidence_len_ok is false, i.e. EVIDENCE is longer "
                         "than 12 words (default: keep them)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every dropped record (with reason) and every "
                         "evidence_verified=false record -- noisy on a full run, meant for dry runs")
    args = ap.parse_args()

    LAYER_2_VERSION = args.version
    VERSION_CFG = LAYER_2_VERSIONS[LAYER_2_VERSION]
    sft_version = args.sft_version
    sft_version_cfg = CITATION_CLASSIFIER_SLM_VERSIONS[sft_version]
    missing_keys = set(sft_version_cfg["output_keys"]) - set(LABEL_KEYS)
    if missing_keys:
        raise ValueError(f"SLM {sft_version} output_keys {sorted(missing_keys)} not in LABEL_KEYS {LABEL_KEYS}")

    input_dir_stem = os.path.basename(os.path.normpath(args.input_dir))
    out_dir = os.path.join(args.out_dir, f"{input_dir_stem}_layer2_{LAYER_2_VERSION}")
    resuming = os.path.isdir(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, f"{input_dir_stem}_raw_outputs.jsonl")
    labeled_path = os.path.join(out_dir, f"{input_dir_stem}_layer2_labeled.jsonl")
    train_path = os.path.join(out_dir, f"{input_dir_stem}_layer2_train_slm_{sft_version}.jsonl")
    val_path = os.path.join(out_dir, f"{input_dir_stem}_layer2_val_slm_{sft_version}.jsonl")
    labels_path = os.path.join(out_dir, f"{input_dir_stem}_layer2_labels.json")
    summary_path = os.path.join(out_dir, f"{input_dir_stem}_layer2_summary.md")

    print(f"Provider           : deepseek")
    print(f"Model              : {DEEPSEEK_MODEL}")
    print(f"Input dir          : {args.input_dir} (glob={args.glob}, nb_input_json={args.nb_input_json})")
    print(f"Layer-2 version    : {LAYER_2_VERSION} ({VERSION_CFG['description']})")
    print(f"SFT version        : {sft_version} (CITATION_CLASSIFIER_SLM_VERSIONS, "
          f"targets={list(sft_version_cfg['output_keys'])})")
    print(f"Output dir         : {out_dir}")

    print("\nBuilding records from citation contexts...")
    records, file_stats, files_used = build_records(
        args.input_dir, args.glob, args.nb_input_json, args.max_side_words, VERSION_CFG)
    total_records_built = len(records)
    print(f"  {len(files_used)} file(s) read, {total_records_built} record(s) built "
          f"(one per citation context).")

    cache = load_raw_cache(raw_path)
    if resuming and cache:
        print(f"\nOutput dir already exists -- resuming: {len(cache)} record(s) cached "
              f"(in {os.path.basename(raw_path)}); only new/changed records will be re-queried.")

    # --- Extraction phase (append-only, resumable by record_id + input_hash) ---
    if not args.skip_extract:
        import asyncio
        from tqdm import tqdm

        todo = [r for r in records if cache.get(r["id"], {}).get("input_hash") != r["input_hash"]]
        print(f"\nExtraction: {len(todo)} to do, {len(records) - len(todo)} cached "
              f"(concurrency={args.concurrency}).")

        async def extract_one(record, semaphore):
            async with semaphore:
                try:
                    user_msg = render_user_message(record, VERSION_CFG)
                    messages = [("system", VERSION_CFG["system"]), ("human", user_msg)]
                    raw = await acall_deepseek(messages)
                    return record["id"], record["input_hash"], raw, None
                except Exception as ex:
                    return record["id"], record["input_hash"], None, ex

        async def run_all():
            semaphore = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.ensure_future(extract_one(r, semaphore)) for r in todo]
            with open(raw_path, "a", encoding="utf-8") as fout:
                for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                                desc="Classifying citation relations", unit="context"):
                    rid, ihash, raw, err = await fut
                    if err is not None:
                        print(f"  [id {rid}] extraction failed: {err}", file=sys.stderr)
                        continue
                    rec = {"record_id": rid, "input_hash": ihash, "raw_output": raw}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    cache[rid] = {"input_hash": ihash, "raw_output": raw}

        asyncio.run(run_all())
        print("Extraction complete.")
    else:
        print("Skipping extraction (--skip_extract); parsing cached outputs only.")

    # --- Parse + validate phase ---
    print("\nParsing raw outputs into Layer-2 labels...")
    labeled, drop_reason_counts = [], Counter()
    evidence_unverified = []
    printed_samples = []

    for record in records:
        cached = cache.get(record["id"])
        if cached is None or cached["input_hash"] != record["input_hash"]:
            continue  # not (yet) extracted -- not an error, just not ready

        raw = cached["raw_output"]
        try:
            obj = extract_json(raw)
        except Exception as ex:
            print(f"[id {record['id']}] JSON parse error: {ex}", file=sys.stderr)
            drop_reason_counts["json_parse_error"] += 1
            continue

        valid, errors = validate_label(obj, VERSION_CFG)
        if not valid:
            for e in errors:
                print(f"[id {record['id']}] validation error: {e}", file=sys.stderr)
            drop_reason_counts[errors[0] if errors else "unknown_validation_error"] += 1
            continue

        label = {k: obj[k] for k in LABEL_KEYS}
        verified = check_evidence(label["EVIDENCE"], record["before"], record["center"], record["after"])
        # Flag-only (never a drop reason): the prompt asks for EVIDENCE of "at
        # most 12 words"; the model doesn't always comply, so this is tracked
        # separately from evidence_verified (which is about EVIDENCE being a
        # real substring of the context, not about its length).
        evidence_len_ok = len(label["EVIDENCE"].split()) <= 12

        labeled_record = {
            "id": record["id"], "m_id": record["m_id"], "r_id": record["r_id"], "ref_id": record["ref_id"],
            "section": record["section"], "sentence_index": record["sentence_index"],
            "co_cited": record["co_cited"], "co_cited_method": record["co_cited_method"],
            "source_file": record["source_file"],
            "inputs": {
                "m_title": record["m_title"], "m_narrative": record["m_narrative"],
                "r_title": record["r_title"], "r_narrative": record["r_narrative"],
                "section": record["section"], "before": record["before"],
                "center": record["center"], "after": record["after"],
            },
            "label": label,
            "evidence_verified": verified,
            "evidence_len_ok": evidence_len_ok,
        }
        labeled.append(labeled_record)
        if not verified:
            evidence_unverified.append(labeled_record)
        if len(printed_samples) < 3:
            printed_samples.append((record, render_user_message(record, VERSION_CFG), raw, label))

    n_dropped = sum(drop_reason_counts.values())
    print(f"Parsed {len(labeled)} kept, {n_dropped} dropped.")

    if args.verbose:
        print("\n=== DROPPED RECORDS ===")
        for reason, count in drop_reason_counts.most_common():
            print(f"  {reason}: {count}")
        print("\n=== evidence_verified = false ===")
        for r in evidence_unverified:
            print(f"  id={r['id']}  EVIDENCE={r['label']['EVIDENCE']!r}")

    # --- Sample rendering (always, up to 3) ---
    print("\n=== SAMPLE RENDERED RECORDS (up to 3) ===")
    for record, user_msg, raw, label in printed_samples:
        print(f"\n--- id={record['id']} ---")
        print("USER MESSAGE (tail):")
        print(user_msg[-800:])
        print("RAW OUTPUT:")
        print(raw)
        print("PARSED LABEL:")
        print(json.dumps(label, indent=2))

    # --- Train/val split (grouped by m_id) ---
    train_set, val_set = group_split_by_m_id(labeled, args.val_frac, args.seed)

    # --- SFT filtering + build ---
    print("\nBuilding Qwen SFT chat format...")
    train_examples = [build_sft_example(r, r["label"], sft_version_cfg) for r in train_set
                       if passes_sft_filters(r, args.min_confidence, args.require_evidence, args.require_evidence_len)]
    val_examples = [build_sft_example(r, r["label"], sft_version_cfg) for r in val_set
                     if passes_sft_filters(r, args.min_confidence, args.require_evidence, args.require_evidence_len)]

    # --- Write output files ---
    print("Writing output files...")
    with open(labeled_path, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labeled)} labeled records -> {labeled_path}")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_examples)} train examples -> {train_path}")

    if val_examples:
        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_examples)} val examples -> {val_path}")

    labels_metadata = {
        "version": LAYER_2_VERSION,
        "sft_version": sft_version,
        "description": VERSION_CFG["description"],
        "keys": list(LABEL_KEYS),
        "functions": list(VERSION_CFG["functions"]),
        "m_aspects": list(VERSION_CFG["m_aspects"]),
        "polarities": list(VERSION_CFG["polarities"]),
        "r_match": list(VERSION_CFG["r_match"]),
        "confidence": list(VERSION_CFG["confidence"]),
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_metadata, f, indent=2)

    # --- Distributions for the summary ---
    function_counts = Counter(r["label"]["FUNCTION"] for r in labeled)
    m_aspect_counts = Counter(r["label"]["M_ASPECT"] for r in labeled)
    polarity_counts = Counter(r["label"]["POLARITY"] for r in labeled)
    r_match_counts = Counter(r["label"]["R_MATCH"] for r in labeled)
    confidence_counts = Counter(r["label"]["CONFIDENCE"] for r in labeled)
    co_cited_counts = Counter(r["co_cited"] for r in labeled)
    cross_table = Counter((r["label"]["FUNCTION"], r["label"]["M_ASPECT"]) for r in labeled)
    n_verified = sum(1 for r in labeled if r["evidence_verified"])
    n_evidence_len_ok = sum(1 for r in labeled if r["evidence_len_ok"])
    n_truncated_total = sum(s.get("truncated", 0) for s in file_stats.values())

    # --- Console summary ---
    print("\n=== SUMMARY ===")
    print(f"files read              : {len(files_used)}")
    print(f"records built (all)     : {total_records_built}")
    print(f"records processed       : {len(records)}")
    print(f"raw cached              : {len(cache)}")
    print(f"kept (valid)            : {len(labeled)}")
    print(f"dropped                 : {n_dropped}")
    print(f"train set (SFT)         : {len(train_examples)}")
    print(f"val set (SFT)           : {len(val_examples)}")
    print(f"evidence_verified       : {n_verified}/{len(labeled)}")
    print(f"evidence_len_ok (<=12w) : {n_evidence_len_ok}/{len(labeled)}")
    print(f"truncated (before/after): {n_truncated_total}")
    print("FUNCTION distribution   :", dict(function_counts))
    print("M_ASPECT distribution   :", dict(m_aspect_counts))
    print("POLARITY distribution   :", dict(polarity_counts))
    print("R_MATCH distribution    :", dict(r_match_counts))
    print("CONFIDENCE distribution :", dict(confidence_counts))
    print("co_cited                :", dict(co_cited_counts))

    print(f"\nWrote: {labeled_path}, {train_path}, {val_path}, {labels_path}")

    # --- Per-file table + summary.md ---
    per_file_rows = []
    for fname in sorted(file_stats):
        s = file_stats[fname]
        per_file_rows.append(
            f"| {fname} | {s.get('n_refs', 0)} | {s.get('refs_no_context', 0)} | "
            f"{s.get('refs_missing_narrative', 0)} | {s.get('contexts_seen', 0)} | "
            f"{s.get('records_built', 0)} | {s.get('duplicates', 0)} | {s.get('truncated', 0)} |"
        )
        if s.get("m_missing_narrative"):
            per_file_rows[-1] += " (M missing TASK/CONTRIBUTION -- file skipped entirely)"

    summary_md = "\n".join([
        f"# {input_dir_stem} (Layer-2 Citation Relation {LAYER_2_VERSION})",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Source",
        "",
        f"- Input dir: `{args.input_dir}` (glob=`{args.glob}`, nb_input_json={args.nb_input_json})",
        f"- Provider / model: `deepseek` / `{DEEPSEEK_MODEL}`",
        f"- Layer-2 version: `{LAYER_2_VERSION}` -- {VERSION_CFG['description']}",
        f"- SFT version: `{sft_version}`",
        f"- max_side_words: {args.max_side_words}",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---|",
        f"| files read | {len(files_used)} |",
        f"| records built (all) | {total_records_built} |",
        f"| records processed | {len(records)} |",
        f"| kept (valid) | {len(labeled)} |",
        f"| dropped | {n_dropped} |",
        f"| train set (SFT) | {len(train_examples)} |",
        f"| val set (SFT) | {len(val_examples)} |",
        f"| evidence_verified | {n_verified}/{len(labeled)} |",
        f"| evidence_len_ok (<=12w) | {n_evidence_len_ok}/{len(labeled)} |",
        f"| truncated before/after | {n_truncated_total} |",
        "",
        "## Dropped, by reason",
        "",
        "| reason | count |",
        "|---|---|",
        *[f"| {reason} | {count} |" for reason, count in drop_reason_counts.most_common()],
        "",
        "## FUNCTION distribution",
        "",
        "| FUNCTION | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in function_counts.most_common()],
        "",
        "## M_ASPECT distribution",
        "",
        "| M_ASPECT | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in m_aspect_counts.most_common()],
        "",
        "## POLARITY distribution",
        "",
        "| POLARITY | count |",
        "|---|---|",
        *[f"| {k or '(empty)'} | {v} |" for k, v in polarity_counts.most_common()],
        "",
        "## R_MATCH distribution",
        "",
        "| R_MATCH | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in r_match_counts.most_common()],
        "",
        "## CONFIDENCE distribution",
        "",
        "| CONFIDENCE | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in confidence_counts.most_common()],
        "",
        "## FUNCTION x M_ASPECT",
        "",
        "| FUNCTION | M_ASPECT | count |",
        "|---|---|---|",
        *[f"| {fn} | {asp} | {c} |" for (fn, asp), c in cross_table.most_common()],
        "",
        "## co_cited",
        "",
        "| co_cited | count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in co_cited_counts.most_common()],
        "",
        "## Per-file",
        "",
        "| file | refs | refs_no_context | refs_missing_narrative | contexts_seen | records_built | duplicates | truncated |",
        "|---|---|---|---|---|---|---|---|",
        *per_file_rows,
        "",
        "## Output Format",
        "",
        "Labeled record (`*_layer2_labeled.jsonl`, one per line):",
        "```json",
        json.dumps({
            "id": "M|R|ref_id|sentence_index|section", "m_id": "...", "r_id": "...", "ref_id": "...",
            "section": "...", "sentence_index": 0, "co_cited": "yes|no", "co_cited_method": "corpus_overlap",
            "source_file": "...",
            "inputs": {"m_title": "...", "m_narrative": {"TASK": "..."}, "r_title": "...",
                       "r_narrative": {"TASK": "..."}, "section": "...", "before": "...",
                       "center": "...", "after": "..."},
            "label": {k: "..." for k in LABEL_KEYS},
            "evidence_verified": True,
        }, indent=2),
        "```",
        "",
        f"SFT chat format (`*_layer2_train_slm_{sft_version}.jsonl` / `*_layer2_val_slm_{sft_version}.jsonl`):",
        "```json",
        '{"messages": [{"role": "system", "content": "..."}, '
        '{"role": "user", "content": "..."}, '
        '{"role": "assistant", "content": "' +
            json.dumps(json.dumps({k: "..." for k in sft_version_cfg["output_keys"]})).strip('"') + '"}]}',
        "```",
        "",
        "## Files",
        "",
        f"- `{os.path.basename(labeled_path)}` — canonical labeled dataset",
        f"- `{os.path.basename(train_path)}` — SFT training set (ready for train_qlora.py)",
        f"- `{os.path.basename(val_path)}` — SFT validation set",
        f"- `{os.path.basename(labels_path)}` — label metadata",
        "",
    ])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()