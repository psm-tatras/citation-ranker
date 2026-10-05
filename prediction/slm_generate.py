"""
Generate NER spans and research highlights from abstracts with a local SLM
(default: Qwen2.5-7B-Instruct, 4-bit) for prediction only.

Two independent tasks, each written in a format your EXISTING scoring code
already consumes -- so you score separately:

  ner        -> spans jsonl: {"text","entities":[{text,label,start,end}]}
                (same contract as predict_ner.py / ner_spans.jsonl; feed to
                 eval_against_llm.py against your LLM/SciBERT references)

  highlights -> jsonl: {"abstract","generated","reference"?}
                (feed to your ROUGE / oracle-ceiling scoring)

Nothing here computes metrics -- generation only, by request.

Setup
-----
    pip install "transformers>=4.44" accelerate bitsandbytes

Usage
-----
    # NER over a CSV of abstracts
    python -m prediction.slm_generate --task ner \
        --input_file test.csv --text_column abstract \
        --output_file slm_ner_spans.jsonl

    # highlights (include --summary_column to carry the reference through)
    python -m prediction.slm_generate --task highlights \
        --input_file test.csv --text_column abstract \
        --summary_column contribution \
        --output_file slm_highlights.jsonl
"""

import argparse
import json
import re
import sys

from utils.prompts import (
    HL_SYSTEM,
    HL_USER_TMPL,
    LABELS,
    NER_SYSTEM,
    NER_USER_TMPL,
)


def load_model(model_id, load_in_4bit=True, dtype="bfloat16"):
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def chat(tok, model, system, user, max_new_tokens=512, temperature=0.0):
    import torch
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]        # strip the prompt
    return tok.decode(gen, skip_special_tokens=True).strip()


# ---- parsing / offset resolution (mirrors your build_ner_dataset pipeline) ----
def _sanitize_json_escapes(s):
    """Escape stray backslashes (e.g. LaTeX) and control chars so json.loads succeeds.

    Valid escapes are consumed as whole units, so already-escaped sequences
    are left unchanged.
    """
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "u" and i + 5 < n and all(ch in "0123456789abcdefABCDEF" for ch in s[i + 2:i + 6]):
                out.append(s[i:i + 6]); i += 6; continue
            if nxt in '"\\/bfnrtu':
                out.append(s[i:i + 2]); i += 2; continue
            out.append("\\\\")  # lone backslash not starting a valid escape
            i += 1
            continue
        out.append(c)
        i += 1
    s = "".join(out)
    s = s.replace("\t", " ").replace("\r", " ")
    return s


def extract_json(raw):
    """Parse the first JSON object in a model completion (code fences allowed)."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    s = raw.find("{")
    if s == -1:
        raise ValueError("no JSON object")
    frag = raw[s:]
    # attempt 1: first balanced object (string-aware)
    for candidate in (frag, _sanitize_json_escapes(frag)):
        try:
            depth = 0
            in_str = esc = False
            for i, c in enumerate(candidate):
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                if in_str:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(candidate[:i + 1])
        except Exception:
            continue
    # attempt 2: harvest {text,label} objects from a malformed entities array
    objs = re.findall(
        r'\{\s*"text"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"label"\s*:\s*"[A-Z_]+"\s*\}',
        _sanitize_json_escapes(frag))
    if objs:
        out = []
        for o in objs:
            try:
                out.append(json.loads(o))
            except Exception:
                continue
        if out:
            return {"entities": out}
    raise ValueError("unbalanced/unrepairable JSON")


def resolve_offsets(text, entities):
    """Greedy left-to-right, whitespace-flexible fallback. Same contract as
    the training pipeline so downstream scoring is apples-to-apples."""
    out, cursor = [], 0
    for e in entities:
        surface = (e.get("text") or "").strip()
        label = e.get("label")
        if not surface or label not in LABELS:
            continue
        idx = text.find(surface, cursor)
        if idx != -1:
            st, en = idx, idx + len(surface)
        else:
            pat = re.compile(r"\s+".join(re.escape(t) for t in surface.split()))
            mm = pat.search(text, cursor) or pat.search(text)
            if mm is None:
                continue
            st, en = mm.span()
        out.append({"text": text[st:en], "label": label, "start": st, "end": en})
        cursor = max(cursor, en)
    return out


def read_rows(path, text_col, sum_col=None):
    rows = []
    if path.endswith(".csv"):
        import csv
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = (r.get(text_col) or "").strip()
                if t:
                    rows.append((t, (r.get(sum_col) or "").strip() if sum_col else None))
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append((r[text_col],
                                 r.get(sum_col) if sum_col else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ner", "highlights"], required=True)
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--text_column", default="abstract")
    ap.add_argument("--summary_column", default=None,
                    help="(highlights) carry the reference into the output.")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report_every", type=int, default=25)
    ap.add_argument("--debug_raw", default=None,
                    help="(ner) also dump {text, raw_output} here to diagnose "
                         "empty output with inspect_slm_ner.py.")
    args = ap.parse_args()

    rows = read_rows(args.input_file, args.text_column, args.summary_column)
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} abstracts. Loading {args.model_id} ...")

    tok, model = load_model(args.model_id, load_in_4bit=not args.no_4bit)

    n_parse_fail = 0
    dbg = open(args.debug_raw, "w", encoding="utf-8") if (args.debug_raw and args.task == "ner") else None
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i, (abstract, reference) in enumerate(rows, 1):
            if args.task == "ner":
                raw = chat(tok, model, NER_SYSTEM,
                           NER_USER_TMPL.format(abstract=abstract),
                           max_new_tokens=args.max_new_tokens)
                if dbg is not None:
                    dbg.write(json.dumps({"text": abstract, "raw_output": raw},
                                         ensure_ascii=False) + "\n")
                    dbg.flush()
                try:
                    ents = extract_json(raw).get("entities", [])
                    spans = resolve_offsets(abstract, ents)
                except Exception:
                    n_parse_fail += 1
                    spans = []
                rec = {"text": abstract, "entities": spans}
            else:
                gen = chat(tok, model, HL_SYSTEM,
                           HL_USER_TMPL.format(abstract=abstract),
                           max_new_tokens=args.max_new_tokens)
                rec = {"abstract": abstract, "generated": gen}
                if reference is not None:
                    rec["reference"] = reference

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % args.report_every == 0:
                print(f"  {i}/{len(rows)}"
                      + (f"  (parse fails: {n_parse_fail})"
                         if args.task == "ner" else ""))

    if dbg is not None:
        dbg.close()
    print(f"Wrote {len(rows)} records to {args.output_file}")
    if args.task == "ner":
        print(f"JSON parse failures: {n_parse_fail}/{len(rows)}")


if __name__ == "__main__":
    main()