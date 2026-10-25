"""
facet_scoring.py

Final M->R importance score from (a) aspect similarity (utils/embedding.py)
and (b) Layer-3 citation classification (facet_relevance).

Per main paper M and reference R (contexts k of R with R_MATCH == "yes"):

    P_R,a   = percentile rank of sim(M_a, R_a) among M's references
              (missing similarity -> MISSING_ASPECT_VALUE)
    p_R     = normalised mask  sum_k w(F_k) * onehot(M_ASPECT_k)
    alpha_R = ALPHA_BY_FUNCTION[F of the strongest context]  (weak labels -> flatter mask)
    q_R     = alpha_R * p_R + (1 - alpha_R) / |A|             (soft mask)
    idf_a   = log((N + 1) / (n_a + 1)) + 1                    (per M, over contexts)
    W_R,a   = q_R,a * (idf_a / mean(idf))                     (context weight, W_mr)
    F_R,a   = P_R,a + BETA * W_R,a                            (Final Relevance Vector F_mr, additive:
                                                               context boosts the cited aspects and
                                                               never lowers the others; BETA = 0 -> P)
    s_R     = max_k w(F_k)                                    (relation strength)
    c_R     = log(1 + n_R) / log(1 + max_R' n_R')             (citation count, n_R = valid contexts of R, per M)
    score_R = LAMBDA * s_R + GAMMA * c_R + (1 - LAMBDA - GAMMA) * mean_a F_R,a

BETA = 0.1 chosen on DeepSeek per-aspect grades over 9,886 papers
(evaluation_main.py): pooled pairwise accuracy (excl. LIMITATION) 0.772, equal to
P alone (0.772), vs 0.693 for the earlier multiplicative F = W * P and
0.694-0.702 for BM25 / SPECTER2 / SciNCL.

Writes `facet_score` into every R entry (after `facet_relevance`) and
`facet_ranking` + `facet_scoring` (constants, idf) at the top level of the same JSON.

Citation FUNCTION (Layer 3) and its weight w(F):
    EXTENDS               1.0  M improves, generalizes or updates R's contribution; R is a special case or limit of M's result
    USES                  0.9  M adopts a method, model, dataset, tool, value or setting from R
    MOTIVATION            0.6  a gap, limitation or problem in R motivates M
    COMPARISON/DIFFERENT  0.6  M sets its approach or results against R and differs from / outperforms it (incl. R as a baseline)
    COMPARISON/SIMILAR    0.5  M sets its approach or results against R and agrees / is consistent with it
    SUPPORTS              0.5  R's result is evidence for a claim or choice of M
    FUTURE                0.2  R appears in work M leaves for the future
    BACKGROUND            0.1  general context; M does not depend on R
    (R_MATCH != yes or unknown FUNCTION -> context ignored)

Output per R entry:
    aspect_similarity  (from embedding.py)  M~R similarity per aspect, content only, no citation context
    facet_score        (this script)        facet_similarity = F_mr, percentile = V_mr (P),
                                            context = W_mr, mask = soft mask q; score, strength, count

Usage:
    python facet_scoring.py --input data/facet_predictions/2302.08982.json --output_dir data/facet_scores
    python facet_scoring.py --input data/facet_predictions --output_dir data/facet_scores --overwrite
    (without --output_dir the input JSON is updated in place)
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

from utils.prompts import ASPECT_KEYS as ASPECTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights and constants (hard-coded for the draft; to become user-configurable)
# ---------------------------------------------------------------------------
# aspect -> key in R["aspect_similarity"] (identity mapping; kept as an
# explicit dict in case a future aspect's key in aspect_similarity ever
# diverges from its name in ASPECTS)
SIM_KEYS = {
    "TASK": "TASK",
    "LIMITATION": "LIMITATION",
    "CONTRIBUTION": "CONTRIBUTION",
    "METHOD": "METHOD",
    "RESULT": "RESULT",
}

FUNCTION_WEIGHTS = {
    "EXTENDS": 1.0,
    "USES": 0.9,
    "MOTIVATION": 0.6,
    "SUPPORTS": 0.5,
    "FUTURE": 0.2,
    "BACKGROUND": 0.1,
}
COMPARISON_WEIGHTS = {          # COMPARISON is weighted by POLARITY
    "DIFFERENT": 0.6,
    "SIMILAR": 0.5,
    "": 0.5,
}
UNKNOWN_FUNCTION_WEIGHT = 0.0   # out-of-schema FUNCTION -> context ignored
R_MATCH_REQUIRED = "yes"        # contexts with any other R_MATCH are dropped

ALPHA_BY_FUNCTION = {           # soft-mask mixing per FUNCTION of R's strongest context
    "EXTENDS": 0.8,             # 1.0 = hard mask, 0.0 = uniform (plain content similarity)
    "USES": 0.8,
    "MOTIVATION": 0.8,
    "COMPARISON": 0.8,
    "SUPPORTS": 0.8,
    "FUTURE": 0.4,
    "BACKGROUND": 0.4,
}
ALPHA_DEFAULT = 0.4             # R with no valid context
BETA = 0.1                      # F_mr = percentile + BETA * context (W_mr); 0 -> content similarity only
LAMBDA = 0.4                    # score = LAMBDA * strength + GAMMA * count + (1 - LAMBDA - GAMMA) * mean(facet_similarity)
GAMMA = 0.2                     # weight of the log-normalised context count of R
MISSING_ASPECT_VALUE = 0.5      # P_R,a when the similarity is missing (neutral = median)
SINGLE_REF_PERCENTILE = 0.5     # P_R,a when only one reference has that similarity
ROUND = 4


class FacetScorer:
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
    def relation(item):
        """facet_relevance item is either the relation dict or {..., 'relation': {...}}."""
        if isinstance(item, dict) and isinstance(item.get("relation"), dict):
            return item["relation"]
        return item if isinstance(item, dict) else {}

    @staticmethod
    def function_weight(rel):
        func = rel.get("FUNCTION", "")
        if func == "COMPARISON":
            return COMPARISON_WEIGHTS.get(rel.get("POLARITY", ""), COMPARISON_WEIGHTS[""])
        return FUNCTION_WEIGHTS.get(func, UNKNOWN_FUNCTION_WEIGHT)

    def valid_contexts(self, entry):
        """[(weight, m_aspect, function)] for contexts with R_MATCH == yes and known labels."""
        out = []
        for item in entry.get("facet_relevance") or []:
            rel = self.relation(item)
            if rel.get("R_MATCH") != R_MATCH_REQUIRED:
                continue
            w = self.function_weight(rel)
            aspect = rel.get("M_ASPECT")
            if w <= 0 or aspect not in ASPECTS:
                continue
            out.append((w, aspect, rel.get("FUNCTION")))
        return out

    @staticmethod
    def percentile_ranks(values):
        """{key: percentile in [0, 1]} with average rank for ties."""
        items = sorted(values.items(), key=lambda kv: kv[1])
        n = len(items)
        if n == 1:
            return {items[0][0]: SINGLE_REF_PERCENTILE}
        ranks, i = {}, 0
        while i < n:
            j = i
            while j + 1 < n and items[j + 1][1] == items[i][1]:
                j += 1
            avg = (i + j) / 2
            for k in range(i, j + 1):
                ranks[items[k][0]] = avg / (n - 1)
            i = j + 1
        return ranks

    @staticmethod
    def insert_after(entry, anchor, key, value):
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

    # --------------------------------------------------------------------- core
    def score_record(self, record):
        entries = list(self.iter_r_entries(record))
        if not entries:
            return 0

        # percentile of each aspect similarity among M's references
        P = {id(e): {} for _, e in entries}
        for a in ASPECTS:
            vals = {}
            for _, e in entries:
                v = (e.get("aspect_similarity") or {}).get(SIM_KEYS[a])
                if isinstance(v, (int, float)):
                    vals[id(e)] = float(v)
            ranks = self.percentile_ranks(vals) if vals else {}
            for _, e in entries:
                P[id(e)][a] = ranks.get(id(e), MISSING_ASPECT_VALUE)

        # IDF over all valid contexts of M
        contexts = {id(e): self.valid_contexts(e) for _, e in entries}
        all_ctx = [a for ctx in contexts.values() for _, a, _ in ctx]
        N = len(all_ctx)
        idf = {a: math.log((N + 1) / (all_ctx.count(a) + 1)) + 1 for a in ASPECTS}
        mean_idf = sum(idf.values()) / len(idf)
        idf_w = {a: idf[a] / mean_idf for a in ASPECTS}
        max_n = max(len(c) for c in contexts.values())

        ranking = []
        for r_id, e in entries:
            ctx = contexts[id(e)]
            m = {a: 0.0 for a in ASPECTS}
            for w, a, _ in ctx:
                m[a] += w
            total = sum(m.values())
            p = {a: (m[a] / total if total > 0 else 1 / len(ASPECTS)) for a in ASPECTS}
            strongest = max(ctx, key=lambda c: c[0]) if ctx else None
            strength = strongest[0] if strongest else 0.0
            alpha = ALPHA_BY_FUNCTION.get(strongest[2], ALPHA_DEFAULT) if strongest else ALPHA_DEFAULT
            q = {a: alpha * p[a] + (1 - alpha) / len(ASPECTS) for a in ASPECTS}
            W = {a: q[a] * idf_w[a] for a in ASPECTS}                      # W_mr
            F = {a: P[id(e)][a] + BETA * W[a] for a in ASPECTS}             # F_mr (additive)
            count = math.log1p(len(ctx)) / math.log1p(max_n) if max_n > 0 else 0.0
            score = (LAMBDA * strength + GAMMA * count
                     + (1 - LAMBDA - GAMMA) * sum(F.values()) / len(ASPECTS))

            self.insert_after(e, "facet_relevance", "facet_score", {
                "score": round(score, ROUND),
                "strength": round(strength, ROUND),
                "count": round(count, ROUND),
                "alpha": alpha,
                "n_contexts_used": len(ctx),
                "facet_similarity": {a: round(F[a], ROUND) for a in ASPECTS},
                "context": {a: round(W[a], ROUND) for a in ASPECTS},
                "mask": {a: round(q[a], ROUND) for a in ASPECTS},
                "percentile": {a: round(P[id(e)][a], ROUND) for a in ASPECTS},
            })
            ranking.append((r_id, score))

        ranking.sort(key=lambda x: -x[1])
        record["facet_ranking"] = [{"r_id": r, "score": round(s, ROUND)} for r, s in ranking]
        record["facet_scoring"] = {
            "aspects": list(ASPECTS),
            "function_weights": FUNCTION_WEIGHTS,
            "comparison_weights": COMPARISON_WEIGHTS,
            "alpha_by_function": ALPHA_BY_FUNCTION,
            "alpha_default": ALPHA_DEFAULT,
            "beta": BETA,
            "lambda": LAMBDA,
            "gamma": GAMMA,
            "max_contexts": max_n,
            "missing_aspect_value": MISSING_ASPECT_VALUE,
            "n_contexts": N,
            "idf": {a: round(idf[a], ROUND) for a in ASPECTS},
            "idf_weight": {a: round(idf_w[a], ROUND) for a in ASPECTS},
        }
        return len(entries)

    @staticmethod
    def already_scored(path):
        try:
            with open(path, encoding="utf-8") as f:
                return "facet_ranking" in json.load(f)
        except (OSError, ValueError):
            return False

    @staticmethod
    def update_report_if_exists(input_path, record):
        """If a Markdown report already exists for this input, regenerate it
        from the now-scored record so it doesn't go stale. Looked up next to
        `input_path` (not wherever --output_dir redirected the scored JSON
        to), since that's where the report/JSON pair actually lives --
        tries both a 'reports/' dir sibling to input_path itself (old layout:
        data/facet_predictions/{id}.json + data/facet_predictions/reports/)
        and one sibling to its PARENT dir (current layout:
        data/facet_predictions/layer_1_output/{id}.json +
        data/facet_predictions/reports/)."""
        from utils.generate_report import generate_report

        candidates = (
            input_path.parent / "reports" / f"{input_path.stem}_report.md",
            input_path.parent.parent / "reports" / f"{input_path.stem}_report.md",
        )
        report_path = next((c for c in candidates if c.is_file()), None)
        if report_path is None:
            return False
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(generate_report(record))
        logger.info("Updated report -> %s", report_path)
        return True

    def process_file(self, path, output_dir=None, overwrite=False):
        """Score one FACET JSON; write to output_dir/<name> if given, else in place.
        Also regenerates path's Markdown report (see update_report_if_exists), if any --
        even when already-scored output is reused below, so the report stays in sync
        with whatever's currently scored instead of only updating on a fresh score."""
        out_path = Path(output_dir) / path.name if output_dir else path
        if not overwrite and self.already_scored(out_path):
            with open(out_path, encoding="utf-8") as f:
                self.update_report_if_exists(path, json.load(f))
            return 0
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        n = self.score_record(record)
        if n:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=3)
            os.replace(tmp, out_path)
            self.update_report_if_exists(path, record)
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
                logger.info("%s: scored %d references", path.name, n)
        logger.info("files=%d scored=%d failed=%d", len(files), done, failed)


def main():
    parser = argparse.ArgumentParser(description="Final FACET importance scoring for M->R pairs.")
    parser.add_argument("--input", required=True, help="A FACET JSON file or a directory of them")
    parser.add_argument("--output_dir", default=None,
                        help="Write scored JSONs here (same file names); input files are left untouched. "
                             "Without it, input files are updated in place")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--overwrite", action="store_true", help="Re-score files already scored at the destination")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Process only the first N files when --input is a directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FacetScorer().run(args.input, args.output_dir, args.pattern, args.overwrite, args.limit)


if __name__ == "__main__":
    main()