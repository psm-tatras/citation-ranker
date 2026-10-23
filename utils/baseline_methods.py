"""
baseline_methods.py

Baseline M->R scores for judging FACET ranking quality. Each baseline reads a
FACET output JSON (facet_main / facet_scoring output) and writes its own key
into every R entry of the same JSON.

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

Output per R entry:
    bm25_similarity  {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
                     = (BM25(T_M,T_R), BM25(L_M,L_R), BM25(C_M,C_R), BM25(M_M,M_R), BM25(R_M,R_R))
                     inserted after `aspect_similarity` (or after `aspect` if absent)

Functions:
    tokenize            lowercase alphanumeric tokens without stopwords
    BaselineScorer      iter_r_entries / insert_after: shared JSON helpers
      .bm25_scores      raw BM25 of one query against a list of documents (Lucene IDF)
      .bm25_record      adds bm25_similarity to every R entry of one M record
      .process_file     one JSON -> output_dir/<name> (or in place)
      .run              file or directory of JSONs

Usage:
    python -m utils.baseline_methods --input data/facet_scores/0704.0262.json --output_dir data/baselines
    python -m utils.baseline_methods --input data/facet_scores --output_dir data/baselines --overwrite
    (without --output_dir the input JSON is updated in place)
"""

import argparse
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASPECTS = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")

BM25_K1 = 1.5
BM25_B = 0.75
BM25_KEY = "bm25_similarity"
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


def tokenize(text):
    """Lowercase alphanumeric tokens, stopwords removed; [] for empty/non-string."""
    if not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BaselineScorer:
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def iter_r_entries(record):
        """Yield (r_id, entry) for references -> {R: [ {R: entry}, ... ]}."""
        for r_id, items in (record.get("references") or {}).items():
            for item in items if isinstance(items, list) else [items]:
                entry = item.get(r_id) if isinstance(item, dict) else None
                if isinstance(entry, dict):
                    yield r_id, entry

    @staticmethod
    def insert_after(entry, anchors, key, value):
        """Place key right after the first anchor present in entry (else at the end)."""
        anchor = next((a for a in anchors if a in entry and a != key), None)
        items = [(k, v) for k, v in entry.items() if k != key]
        entry.clear()
        placed = False
        for k, v in items:
            entry[k] = v
            if k == anchor:
                entry[key] = value
                placed = True
        if not placed:
            entry[key] = value

    # --------------------------------------------------------------------- BM25
    @staticmethod
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

    def bm25_record(self, record):
        """Add bm25_similarity (5 aspects) to every R entry of one M record."""
        entries = [e for _, e in self.iter_r_entries(record)]
        if not entries:
            return 0
        m_aspects = record.get("m_aspects") or {}
        sims = [dict.fromkeys(ASPECTS) for _ in entries]
        for a in ASPECTS:
            query = tokenize(m_aspects.get(a))
            if not query:
                continue
            idx, docs = [], []
            for i, e in enumerate(entries):
                toks = tokenize((e.get("aspect") or {}).get(a))
                if toks:
                    idx.append(i)
                    docs.append(toks)
            for i, s in zip(idx, self.bm25_scores(query, docs)):
                sims[i][a] = round(s, ROUND)
        for e, sim in zip(entries, sims):
            self.insert_after(e, ("aspect_similarity", "aspect"), BM25_KEY, sim)
        return len(entries)

    # ---------------------------------------------------------------------- I/O
    @staticmethod
    def already_done(path):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError):
            return False
        entries = list(BaselineScorer.iter_r_entries(record))
        return bool(entries) and all(BM25_KEY in e for _, e in entries)

    def process_file(self, path, output_dir=None, overwrite=False):
        """Score one FACET JSON; write to output_dir/<name> if given, else in place."""
        out_path = Path(output_dir) / path.name if output_dir else path
        if not overwrite and self.already_done(out_path):
            return 0
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        n = self.bm25_record(record)
        if n:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=3)
            os.replace(tmp, out_path)
        return n

    def run(self, input_path, output_dir=None, pattern="*.json", overwrite=False, limit=None):
        p = Path(input_path)
        files = [p] if p.is_file() else sorted(p.glob(pattern))
        if limit:
            files = files[:limit]
        done = failed = 0
        for path in files:
            try:
                n = self.process_file(path, output_dir=output_dir, overwrite=overwrite)
            except Exception as exc:
                failed += 1
                logger.error("%s: %s", path.name, exc)
                continue
            if n:
                done += 1
                logger.info("%s: BM25 for %d references", path.name, n)
        logger.info("files=%d scored=%d failed=%d", len(files), done, failed)


def main():
    parser = argparse.ArgumentParser(description="Baseline M->R scores (BM25 per aspect).")
    parser.add_argument("--input", required=True, help="A FACET JSON file or a directory of them")
    parser.add_argument("--output_dir", default=None,
                        help="Write JSONs here (same file names); without it, input files are updated in place")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--overwrite", action="store_true", help="Recompute files already scored at the destination")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Process only the first N files when --input is a directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    BaselineScorer().run(args.input, args.output_dir, args.pattern, args.overwrite, args.limit)


if __name__ == "__main__":
    main()
