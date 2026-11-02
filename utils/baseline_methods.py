"""
baseline_methods.py

Baseline M->R scores for judging FACET ranking quality. Driven by baseline_main.py,
which loads each main paper's FACET output JSON (`record`) and its <m_id>_bl.json
dict (`bl`) once; every baseline works in memory and adds its own key to every
reference in `bl`. baseline_main writes `bl` back once per paper.

Metrics
-------
BM25 (Okapi, Lucene IDF), per aspect a in {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}:

    query      q = tokens(M_a)
    documents  D = {tokens(R_a) : R in references of M, R_a non-empty}   (corpus per M per aspect)
    N          = |D|
    n(t)       = number of documents in D containing term t
    idf(t)     = log(1 + (N - n(t) + 0.5) / (n(t) + 0.5))                (Lucene, always > 0)
    f(t, d)    = frequency of t in document d
    |d|        = length of d in tokens, avgdl = mean |d| over D

    BM25(q, d) = sum_{t in q} idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

    k1 = 1.5, b = 0.75; query terms summed per occurrence; raw score (no normalisation).
    tokens(): lowercase, alphanumeric tokens, English stopwords removed, no stemming.
    BM25 is null when M_a or R_a is empty.

SPECTER2 cosine similarity, per aspect a:

    e(x)              = L2-normalised CLS embedding of text x from SPECTER2
                        (allenai/specter2_base + proximity adapter allenai/specter2, max 512 tokens)
    SPECTER2(M_a,R_a) = cos(e(M_a), e(R_a)) = e(M_a) . e(R_a)               (raw cosine, [-1, 1])
    SPECTER2 is null when M_a or R_a is empty.

SciNCL cosine similarity, per aspect a:

    e(x)            = L2-normalised CLS embedding of text x from SciNCL (malteos/scincl, max 512 tokens)
    SciNCL(M_a,R_a) = cos(e(M_a), e(R_a)) = e(M_a) . e(R_a)                 (raw cosine, [-1, 1])
    SciNCL is null when M_a or R_a is empty.

SciBERT cosine similarity, per aspect a:

    e(x)             = L2-normalised mean-pooled (attention-masked) last hidden state of text x
                       from SciBERT (allenai/scibert_scivocab_uncased, max 512 tokens)
    SciBERT(M_a,R_a) = cos(e(M_a), e(R_a)) = e(M_a) . e(R_a)                (raw cosine, [-1, 1])
    SciBERT is null when M_a or R_a is empty.

SPECTER2 whole-abstract cosine similarity (paper-level, not aspect-specific):

    x_P                 = title_P + " [SEP] " + abstract_P     (SPECTER2 input format; abstract only if no title)
    SPECTER2_abs(M, R)  = cos(e(x_M), e(x_R))                 (e as for SPECTER2, raw cosine, [-1, 1])
    The same value is stored for all 5 aspects, so evaluation ranks one paper-level score
    against each aspect's ground truth. Null when M or R has no abstract.

Output per reference in <m_id>_bl.json:
    "bm25":     {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                = (BM25(T_M,T_R), BM25(L_M,L_R), BM25(C_M,C_R), BM25(M_M,M_R), BM25(R_M,R_R))
    "specter2": {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                = (SPECTER2(T_M,T_R), ..., SPECTER2(R_M,R_R))
    "scincl":   {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                = (SciNCL(T_M,T_R), ..., SciNCL(R_M,R_R))
    "scibert":  {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                = (SciBERT(T_M,T_R), ..., SciBERT(R_M,R_R))
    "specter2_abstract": {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                = SPECTER2_abs(M, R) in every aspect

Functions:
    iter_r_entries   (r_id, entry) for every reference of a FACET record
    load_json        read a JSON file
    save_json        write a JSON file atomically
    tokenize         lowercase alphanumeric tokens without stopwords
    bm25_scores      raw BM25 of one query against a list of documents (Lucene IDF)
    bm25             adds "bm25" to every reference of bl (CPU)
    aspect_texts     unique non-empty aspect texts of M and its references (input to encode())
    embedding_cosine adds key: per-aspect cosine to every reference of bl, from precomputed vectors
    specter2         embedding_cosine with key "specter2" (vectors: utils.embedding_utils.Specter2Model)
    scincl           embedding_cosine with key "scincl"   (vectors: utils.embedding_utils.SciNCLModel)
    scibert          embedding_cosine with key "scibert"  (vectors: utils.embedding_utils.SciBERTModel)
    paper_text       title [SEP] abstract of a paper, or None without abstract
    abstract_texts   unique paper_text of M and its references (input to encode())
    specter2_abstract adds "specter2_abstract": whole-abstract cosine in every aspect (Specter2Model vectors)
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASPECTS = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")

BM25_K1 = 1.5
BM25_B = 0.75
BM25_KEY = "bm25"
SPECTER2_KEY = "specter2"
SCINCL_KEY = "scincl"
SCIBERT_KEY = "scibert"
SPECTER2_ABSTRACT_KEY = "specter2_abstract"
SEP = " [SEP] "
ROUND = 4

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my
myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves via within without upon among thus however whereas e g i e et al
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def iter_r_entries(record):
    """Yield (r_id, entry) for references -> {R: [ {R: entry}, ... ]}."""
    for r_id, items in (record.get("references") or {}).items():
        for item in items if isinstance(items, list) else [items]:
            entry = item.get(r_id) if isinstance(item, dict) else None
            if isinstance(entry, dict):
                yield r_id, entry


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=3)
    os.replace(tmp, path)


def tokenize(text):
    """Lowercase alphanumeric tokens, stopwords removed; [] for empty/non-string."""
    if not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Baseline: BM25 per aspect
# ---------------------------------------------------------------------------
def bm25_scores(query, docs, k1=BM25_K1, b=BM25_B):
    """Raw BM25 of query (token list) against each doc (token list), Lucene IDF."""
    N = len(docs)
    if N == 0 or not query:
        return [0.0] * N
    avgdl = sum(len(d) for d in docs) / N
    df = Counter(t for d in docs for t in set(d))
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query)}
    scores = []
    for d in docs:
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avgdl) if avgdl > 0 else k1
        s = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                s += idf[t] * f * (k1 + 1) / (f + norm)
        scores.append(s)
    return scores


def bm25(record, bl):
    """Add "bm25" {5 aspects} to every reference of bl (in memory), using texts from record."""
    entries = list(iter_r_entries(record))
    m_aspects = record.get("m_aspects") or {}
    sims = {r_id: dict.fromkeys(ASPECTS) for r_id, _ in entries}
    for a in ASPECTS:
        query = tokenize(m_aspects.get(a))
        if not query:
            continue
        ids, docs = [], []
        for r_id, e in entries:
            toks = tokenize((e.get("aspect") or {}).get(a))
            if toks:
                ids.append(r_id)
                docs.append(toks)
        for r_id, sc in zip(ids, bm25_scores(query, docs)):
            sims[r_id][a] = round(sc, ROUND)

    for r_id, sim in sims.items():
        bl.setdefault(r_id, {})[BM25_KEY] = sim
    return bl


# ---------------------------------------------------------------------------
# Baselines: embedding cosine similarity per aspect (SPECTER2, SciNCL, SciBERT)
# ---------------------------------------------------------------------------
def _clean(text):
    """Whitespace-normalised text, or None if empty/non-string."""
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    return text or None


def aspect_texts(record):
    """Set of unique non-empty (cleaned) aspect texts of M and all its references."""
    texts = {_clean(t) for t in (record.get("m_aspects") or {}).values()}
    for _, e in iter_r_entries(record):
        texts.update(_clean(t) for t in (e.get("aspect") or {}).values())
    texts.discard(None)
    return texts


def embedding_cosine(record, bl, vecs, key):
    """Add key: {5 aspects} cos(e(M_a), e(R_a)) to every reference of bl (in memory).
    vecs: {text: L2-normalised vector} covering aspect_texts(record)."""
    entries = list(iter_r_entries(record))
    m_aspects = record.get("m_aspects") or {}
    sims = {r_id: dict.fromkeys(ASPECTS) for r_id, _ in entries}
    for a in ASPECTS:
        m_text = _clean(m_aspects.get(a))
        if m_text is None:
            continue
        m_vec = vecs[m_text]
        for r_id, e in entries:
            t = _clean((e.get("aspect") or {}).get(a))
            if t is not None:
                sims[r_id][a] = round(float(m_vec @ vecs[t]), ROUND)

    for r_id, sim in sims.items():
        bl.setdefault(r_id, {})[key] = sim
    return bl


def specter2(record, bl, vecs):
    """Add "specter2" {5 aspects} to every reference of bl; vecs from Specter2Model.encode."""
    return embedding_cosine(record, bl, vecs, SPECTER2_KEY)


def scincl(record, bl, vecs):
    """Add "scincl" {5 aspects} to every reference of bl; vecs from SciNCLModel.encode."""
    return embedding_cosine(record, bl, vecs, SCINCL_KEY)


def scibert(record, bl, vecs):
    """Add "scibert" {5 aspects} to every reference of bl; vecs from SciBERTModel.encode."""
    return embedding_cosine(record, bl, vecs, SCIBERT_KEY)


# ---------------------------------------------------------------------------
# Baseline: SPECTER2 whole-abstract cosine similarity (paper-level)
# ---------------------------------------------------------------------------
def paper_text(title, abstract):
    """title [SEP] abstract (SPECTER2 input format); abstract alone without title; None without abstract."""
    abstract, title = _clean(abstract), _clean(title)
    if abstract is None:
        return None
    return title + SEP + abstract if title else abstract


def abstract_texts(record):
    """Set of unique paper_text of M and all its references."""
    texts = {paper_text(record.get("m_title"), record.get("m_abstract"))}
    texts.update(paper_text(e.get("title"), e.get("abstract")) for _, e in iter_r_entries(record))
    texts.discard(None)
    return texts


def specter2_abstract(record, bl, vecs):
    """Add "specter2_abstract" {5 aspects} = cos(e(x_M), e(x_R)) to every reference of bl;
    vecs from Specter2Model.encode over abstract_texts(record)."""
    m_text = paper_text(record.get("m_title"), record.get("m_abstract"))
    for r_id, e in iter_r_entries(record):
        r_text = paper_text(e.get("title"), e.get("abstract"))
        sim = None
        if m_text is not None and r_text is not None:
            sim = round(float(vecs[m_text] @ vecs[r_text]), ROUND)
        bl.setdefault(r_id, {})[SPECTER2_ABSTRACT_KEY] = dict.fromkeys(ASPECTS, sim)
    return bl
