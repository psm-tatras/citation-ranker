"""
Inference helpers for the two fine-tuned models in this project:
  - the NER token classifier (train_ner.py)
  - the seq2seq contribution generator (train_contribution_baseline.py)

Functions
---------
  load_ner_model(model_dir)          -> (tokenizer, model, device)
  load_summarizer_model(model_dir)   -> (tokenizer, model, device)
  predict_ner(abstract, ...)         -> list of {text, label, start, end}
  predict_contribution(abstract, ...) -> str
  predict_contribution_with_ner(abstract, ...) -> str
      (runs NER first, then feeds the abstract + extracted entities to the
      summarizer, so the generated contribution is guided by the tagged
      research-role spans)
"""

import os
import re
from pathlib import Path
from utils.prompts import *
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
)
from .slm_generate import extract_json, resolve_offsets
from utils.prompts import NER_SYSTEM


def load_ner_model(model_dir=None, device=None):
    model_dir = model_dir or os.environ.get("MODEL_DIR")
    if not model_dir:
        raise ValueError("model_dir not provided and MODEL_DIR env var is not set")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device

def combine_entities_by_label(entities):
    grouped = {}
    for e in entities:
        grouped.setdefault(e['label'], []).append(e['text'])
    return {label: "|\n".join(texts) for label, texts in grouped.items()}

def load_summarizer_model(model_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def _word_spans(text):
    """Whitespace tokens with their char offsets (matches training format,
    where abstracts are space-separated incl. punctuation)."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def predict_ner(abstract, tokenizer, model, device, max_length=256):
    words = _word_spans(abstract)
    if not words:
        return []
    word_toks = [w for _, _, w in words]

    enc = tokenizer(
        word_toks,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    word_ids = enc.word_ids(batch_index=0)
    id2label = model.config.id2label

    # one tag per WORD = tag of its first subword
    word_tags = {}
    prev = None
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None or wid == prev:
            prev = wid
            continue
        word_tags[wid] = id2label[pid]
        prev = wid

    # merge consecutive words into entity spans (BIO-aware)
    entities = []
    cur = None
    for i, (ws, we, _) in enumerate(words):
        tag = word_tags.get(i, "O")
        if tag == "O":
            if cur:
                entities.append(cur); cur = None
            continue
        prefix, label = (tag.split("-", 1) + ["", ""])[:2] if "-" in tag else ("B", tag)
        if prefix == "B" or cur is None or cur["label"] != label:
            if cur:
                entities.append(cur)
            cur = {"label": label, "start": ws, "end": we}
        else:  # I- continuation of same label
            cur["end"] = we
    if cur:
        entities.append(cur)

    for e in entities:
        e["text"] = abstract[e["start"]:e["end"]]
    return [{"text": e["text"], "label": e["label"],
             "start": e["start"], "end": e["end"]} for e in entities]


def _strip_latex(text):
    r"""Remove/normalize LaTeX so it doesn't dominate embeddings of math-heavy
    (e.g. physics) abstracts. Inline math $...$ and \command{...} tokenize into
    meaningless subwords ('\\equiv', '{', '}') that dilute the vector. This
    replaces each inline-math span with a single placeholder token <MATH> and
    strips stray backslash-commands, keeping the surrounding prose intact.

    Use for the EMBEDDING input; keep the raw text for display."""
    # $...$ and $$...$$ inline/display math -> placeholder
    text = re.sub(r'\${1,2}[^$]*\${1,2}', ' <MATH> ', text)
    # leftover \command or \command{..}
    text = re.sub(r'\\[a-zA-Z]+\*?(\{[^}]*\})?', ' ', text)
    # stray braces / carets / underscores from math that escaped the above
    text = re.sub(r'[{}^_]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_ws(text):
    """Collapse hard line-wrapping and runs of whitespace to single spaces.
    Many abstracts (arXiv dumps especially) wrap lines every ~70 chars with
    '\\n' that are NOT sentence boundaries; splitting on those fragments
    entities mid-phrase. Returns (normalized_text, offset_map) where
    offset_map[i] is the char index in the ORIGINAL text for normalized char i,
    so windows can still be reported against the original if needed."""
    out_chars, offset_map = [], []
    prev_space = False
    for i, c in enumerate(text):
        if c in " \t\r\n":
            if not prev_space:
                out_chars.append(" ")
                offset_map.append(i)
                prev_space = True
        else:
            out_chars.append(c)
            offset_map.append(i)
            prev_space = False
    offset_map.append(len(text))          # sentinel for end offsets
    return "".join(out_chars), offset_map


def _sentence_spans(text):
    """Split text into sentences and return (start, end) char offsets for each.
    Offsets index into the ORIGINAL text (whitespace preserved), so a window
    can be sliced back out verbatim.

    Boundary heuristic tuned for scientific abstracts, where the corpus
    convention is space-padded punctuation ('... link . The next ...'):
      - break on . ! ? that are followed by whitespace + a capital/opening,
      - but NOT after common abbreviations (e.g., i.e., et al., Fig.) or after
        a lone capital initial ('J . Smith'), which would over-split.
    It is deliberately simple -- no NLP dependency -- and errs toward slightly
    longer sentences rather than fragmenting mid-clause."""
    if not text.strip():
        return []
    _ABBREV = {"e.g", "i.e", "et al", "cf", "vs", "fig", "eq", "no", "ref",
               "al", "approx", "resp"}
    spans = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c in ".!?":
            # look at what follows the terminator
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            nxt = text[j] if j < n else ""
            # candidate boundary only if followed by end-of-text or a capital/(
            is_boundary = (j >= n) or nxt.isupper() or nxt in "([\"'"
            # suppress if the token before '.' is a known abbreviation or a
            # single capital letter (initials)
            if is_boundary and c == ".":
                k = i - 1
                while k >= start and text[k] not in " \t":
                    k -= 1
                prev_tok = text[k + 1:i].lower().rstrip(".")
                if prev_tok in _ABBREV or (len(prev_tok) == 1 and prev_tok.isalpha()):
                    is_boundary = False
            if is_boundary:
                end = i + 1
                if text[start:end].strip():
                    spans.append((start, end))
                start = j
                i = j
                continue
        i += 1
    if start < n and text[start:].strip():
        spans.append((start, n))
    return spans


def augment_context(entity, abstract, window=1, max_chars=0,
                    return_meta=False):
    """Post-process a single NER hit into a text WINDOW that encloses it: the
    sentence the entity falls in, plus `window` sentences on each side.

    Why: a bare entity string ('driving simulator') is ambiguous out of
    context; the surrounding sentence(s) carry the relation the entity
    participates in ('... measured using a driving simulator across 45
    participants'). Useful for downstream RAG-style guidance, evidence
    snippets, or human review.

    Parameters
    ----------
    entity   : dict with at least 'start' and 'end' (char offsets into
               `abstract`); the shape predict_ner / predict_ner_ft return.
    abstract : the full source text the offsets index into.
    window   : sentences to include on EACH side of the entity's sentence.
               window=0 -> just the entity's own sentence;
               window=1 -> previous + own + next (default);
               window=2 -> two on each side, etc.
    max_chars: optional hard cap on the returned window length. If the
               sentence window exceeds it, the window is trimmed symmetrically
               around the entity while ALWAYS keeping the entity's own
               sentence whole (0 = no cap).
    return_meta: if True, return a dict with the window text plus the entity's
               offsets re-based into the window and which sentences were used.

    Returns
    -------
    str (the window text) or, with return_meta=True, a dict:
        {text, entity_start_in_window, entity_end_in_window,
         window_start, window_end, sentence_index, n_sentences}

    Strategy notes
    --------------
    The window is chosen by SENTENCE membership, not a fixed char radius, so
    it never cuts a sentence mid-clause. If the entity straddles a (mis-
    detected) sentence boundary, every sentence it overlaps is treated as the
    'home' block and the +/- window is measured from that block's edges -- so
    a boundary-spanning entity still yields coherent context rather than half
    a sentence."""
    if entity is None or "start" not in entity or "end" not in entity:
        raise ValueError("entity must have 'start' and 'end' char offsets")
    es, ee = entity["start"], entity["end"]
    sents = _sentence_spans(abstract)
    if not sents:
        return _pack_window(abstract, 0, len(abstract), es, ee,
                            0, 1, return_meta)

    # home block = index range of sentences the entity overlaps
    home = [idx for idx, (s, e) in enumerate(sents) if s < ee and e > es]
    if not home:
        # entity offsets fell outside detected sentences (e.g. trailing
        # whitespace); attach to the nearest sentence by start position
        nearest = min(range(len(sents)),
                      key=lambda idx: abs(sents[idx][0] - es))
        home = [nearest]

    lo = max(0, home[0] - window)
    hi = min(len(sents) - 1, home[-1] + window)
    win_start = sents[lo][0]
    win_end = sents[hi][1]

    # optional length cap: shrink outward context symmetrically but keep the
    # entity's home sentence(s) intact
    if max_chars and (win_end - win_start) > max_chars:
        home_start = sents[home[0]][0]
        home_end = sents[home[-1]][1]
        budget = max_chars - (home_end - home_start)
        if budget > 0:
            left = budget // 2
            right = budget - left
            win_start = max(win_start, home_start - left)
            win_end = min(win_end, home_end + right)
        else:
            win_start, win_end = home_start, home_end

    return _pack_window(abstract, win_start, win_end, es, ee,
                        home[0], len(sents), return_meta)


def _pack_window(abstract, win_start, win_end, es, ee, sent_idx, n_sents,
                 return_meta):
    text = abstract[win_start:win_end]
    if not return_meta:
        return text
    return {
        "text": text,
        "entity_start_in_window": es - win_start,
        "entity_end_in_window": ee - win_start,
        "window_start": win_start,
        "window_end": win_end,
        "sentence_index": sent_idx,
        "n_sentences": n_sents,
    }


def augment_context_batch(entities, abstract, window=1, merge_overlapping=True,
                          normalize_ws=True, strip_latex=False):
    """Build context windows for MANY entities at once, merging windows that
    overlap so shared context is emitted ONCE instead of duplicated per entity.

    On dense abstracts, window=1 around several nearby entities yields
    near-identical windows (see the Muon g-2 example, where the first three
    entities all pulled the same opening block). Embedding those separately
    wastes compute and skews aggregation toward entity-dense regions. This
    groups entities by shared window.

    Returns a list of blocks, each:
        {"text": <window text>,
         "entities": [ {..entity.., "start_in_window", "end_in_window"}, ...]}
    Entities within a block follow position; blocks follow document order.
    merge_overlapping=False -> one block per entity, deduped when byte-identical.
    """
    src = abstract
    if normalize_ws:
        src, _ = _normalize_ws(abstract)
        ents = []
        cursor = 0
        for e in sorted(entities, key=lambda x: x.get("start", 0)):
            surf = _normalize_ws(e.get("text", ""))[0].strip()
            if not surf:
                continue
            idx = src.find(surf, cursor)
            if idx == -1:
                idx = src.find(surf)
            if idx == -1:
                continue
            ne = dict(e)
            ne["start"], ne["end"], ne["text"] = idx, idx + len(surf), surf
            ents.append(ne)
            cursor = idx
    else:
        ents = sorted(entities, key=lambda x: x.get("start", 0))

    sents = _sentence_spans(src)
    if not sents:
        return [{"text": src, "entities":
                 [dict(e, start_in_window=e["start"], end_in_window=e["end"])
                  for e in ents]}]

    spans = []
    for e in ents:
        home = [idx for idx, (s, en) in enumerate(sents)
                if s < e["end"] and en > e["start"]]
        if not home:
            home = [min(range(len(sents)),
                        key=lambda idx: abs(sents[idx][0] - e["start"]))]
        lo = max(0, home[0] - window)
        hi = min(len(sents) - 1, home[-1] + window)
        spans.append([sents[lo][0], sents[hi][1], e])

    blocks = []
    for lo, hi, e in spans:
        if merge_overlapping and blocks and lo < blocks[-1]["hi"]:
            blocks[-1]["hi"] = max(blocks[-1]["hi"], hi)
            blocks[-1]["ents"].append(e)
        elif (not merge_overlapping) and blocks and \
                blocks[-1]["lo"] == lo and blocks[-1]["hi"] == hi:
            blocks[-1]["ents"].append(e)
        else:
            blocks.append({"lo": lo, "hi": hi, "ents": [e]})

    out = []
    for b in blocks:
        wtext = src[b["lo"]:b["hi"]]
        block = {"text": wtext}
        if strip_latex:
            block["embed_text"] = _strip_latex(wtext)   # LaTeX-free for encoding
        block_ents = []
        for e in b["ents"]:
            ne = dict(e)
            ne["start_in_window"] = e["start"] - b["lo"]
            ne["end_in_window"] = e["end"] - b["lo"]
            block_ents.append(ne)
        block["entities"] = block_ents
        out.append(block)
    return out


def augment_context_by_label(entities, abstract, window=1,
                             merge_overlapping=True, normalize_ws=True,
                             strip_latex=False):
    """Same windowing as augment_context_batch, but the result is PIVOTED BY
    LABEL: a dict mapping each label -> list of the window blocks that contain
    an entity of that label, with the matching entities attached.

    Shape:
        {
          "KEY_CONCEPT": [
             {"text": <window>, ["embed_text": <latex-free>,]
              "entities": [ {..only the KEY_CONCEPT entities in this window..} ]},
             ... one entry per window that has a KEY_CONCEPT ...
          ],
          "FINDING": [ ... ],
          ...
        }

    A window that holds three labels appears under all three keys (with only
    that label's entities under each), so per-label you see every paragraph the
    label participates in. Blocks keep document order within each label."""
    blocks = augment_context_batch(
        entities, abstract, window=window,
        merge_overlapping=merge_overlapping,
        normalize_ws=normalize_ws, strip_latex=strip_latex)

    by_label = {}
    for b in blocks:
        # group this block's entities by label
        per = {}
        for e in b["entities"]:
            per.setdefault(e["label"], []).append(e)
        for label, ents in per.items():
            entry = {"text": _highlight_entities(b["text"], ents), "entities": ents}
            if "embed_text" in b:
                entry["embed_text"] = b["embed_text"]
            by_label.setdefault(label, []).append(entry)
    return by_label


def _highlight_entities(text, entities):
    """Wrap each entity's span in <...>, using its window-relative offsets
    (start_in_window/end_in_window) rather than a text search -- a text
    search would mis-highlight when an entity's surface text repeats
    elsewhere in the window or is a substring of another entity's text."""
    ordered = sorted(entities, key=lambda e: e["start_in_window"])
    out = []
    cursor = 0
    for e in ordered:
        s, en = e["start_in_window"], e["end_in_window"]
        if s < cursor:
            continue  # overlaps an already-highlighted span; skip it
        out.append(text[cursor:s])
        out.append(f"<{text[s:en]}>")
        cursor = en
    out.append(text[cursor:])
    return "".join(out)


# --- Fine-tuned (QLoRA adapter) NER -----------------------------------------
# Shared prompt definitions are used here and in the SFT converter so inference
# always uses the EXACT prompt the model was trained on. The prompt contains
# definitions and rules, with no few-shot examples.


def load_ft_ner_model(local_dir, wandb_project=None, artifact_name=None,
                      base_model_id="Qwen/Qwen2.5-7B-Instruct",
                      load_in_4bit=True, dtype="bfloat16", device_map="auto"):
    """Load the QLoRA fine-tuned NER model: resolves the latest trained
    adapter via its W&B model artifact and downloads it (utils.load_model.
    load_latest_adapter -- the same helper train_qlora.py uses to resume
    training) into local_dir, then loads the frozen base from HF with the
    adapter on top. Returns (tokenizer, model) ready for generation.

    The base loads in 4-bit by default so it fits a 12 GB GPU (e.g. RTX 3060),
    matching how it was trained. On a larger GPU you can pass load_in_4bit=False
    for fp16/bf16. The tokenizer is loaded from the adapter dir, so the exact
    chat template you trained with is used.

    local_dir   : where the adapter is downloaded to / then loaded from.
    wandb_project, artifact_name : W&B project / model artifact name to
                  resolve "latest" from; default to train_qlora.py's
                  WANDB_PROJECT / WANDB_ADAPTER_ARTIFACT_NAME. The S3
                  bucket/prefix always come from that artifact's own
                  metadata, not a parameter here.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel
    from utils.load_model import load_latest_adapter, WANDB_PROJECT, WANDB_ADAPTER_ARTIFACT_NAME


    adapter_dir = load_latest_adapter(
        Path(local_dir),
        wandb_project=wandb_project or WANDB_PROJECT,
        artifact_name=artifact_name or WANDB_ADAPTER_ARTIFACT_NAME,
    )

    kwargs = {"device_map": device_map}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)

    base = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    # Adapter uploads contain weights/configuration only. Fall back to the
    # base model tokenizer when tokenizer files were not uploaded with them.
    tokenizer_dir = adapter_dir if any(adapter_dir.glob("tokenizer*")) else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=True,
    )
    return tokenizer, model


def predict_ner_ft(abstract, tokenizer, model, max_new_tokens=512):
    """Runs the QLoRA fine-tuned NER model on an abstract; same return contract
    as predict_ner (list of {text, label, start, end}). Uses NER_USER_TMPL
    (the same prompt used to build the training data), so inference matches
    what the QLoRA model was trained on. JSON parsing + offset resolution
    reuse slm_generate's robust helpers (array-close repair, LaTeX
    sanitizing)."""

    messages = [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": build_ner_user_prompt(abstract, LABELS)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)  # True: generate answer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]      # strip the prompt
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    try:
        entities = extract_json(raw).get("entities", [])
    except Exception:
        return []
    return resolve_offsets(abstract, entities)
