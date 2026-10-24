"""
Evaluate per-aspect reference rankings against ground-truth grades.

Ground truth (<m_id>_gt.json): DeepSeek grades H/M/L per aspect for each M->R pair
    {"m_id": ..., "pairs": [{"r_id": ..., "labels": {"TASK": "H", ...}}, ...]}
Prediction / baseline file (<m_id>_bl.json): per-reference scores of several methods
    {"<r_id>": {"<method>": {"TASK": 0.91, ..., "LIMITATION": null}, ...}, ...}

For every method and aspect, the references of M are ranked by the method's
score and compared with the ground-truth grades:
    pairwise_acc   share of reference pairs with DIFFERENT grades that the method
                   orders the same way (tie in the method = 0.5; 0.5 = random)
    kendall_tau_b  rank correlation, ties handled (tau-b)
    ndcg           NDCG over all references, gains H=2, M=1, L=0
Missing / null scores are ranked last for every method.

Pooled scores combine the aspects: "pooled" over all aspects and "pooled_excl"
without POOLED_EXCLUDE (LIMITATION is almost always graded L, so it carries few
informative pairs). Pooled pairwise_acc pools the pairs; pooled tau / NDCG are
the mean over the aspects.

Across many papers (evaluate_baselines):
    macro  mean of the per-paper values (papers where a value is undefined are
           skipped for that cell; n_papers says how many contributed)
    micro  all reference pairs of all papers pooled (pairwise_acc only)

Usage:
    python evaluation_main.py --gt data/gt/0704.0262_gt.json --bl data/bl/0704.0262_bl.json
    python evaluation_main.py --gt data/gt --bl data/bl --out data/eval_summary.json
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (hard-coded; edit here)
# ---------------------------------------------------------------------------
METHODS = ["percentile", "facet_similarity", "context", "bm25", "specter2", "scincl"]
ASPECTS = ["TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT"]
GRADES = {"H": 2, "M": 1, "L": 0}
POOLED_EXCLUDE = ["LIMITATION"]
GT_SUFFIX = "_gt.json"
BL_SUFFIX = "_bl.json"
MISSING_SCORE = float("-inf")   # missing / null scores are ranked last


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pairwise_counts(scores, grades):
    """(concordant, n_pairs) over pairs with different grades; a tie in scores counts 0.5."""
    concordant, n_pairs = 0.0, 0
    for i, j in combinations(range(len(grades)), 2):
        if grades[i] == grades[j]:
            continue
        n_pairs += 1
        if scores[i] == scores[j]:
            concordant += 0.5
        elif (scores[i] > scores[j]) == (grades[i] > grades[j]):
            concordant += 1.0
    return concordant, n_pairs


def kendall_tau_b(x, y):
    """Kendall tau-b; None if either ranking is constant."""
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            ties_x += 1
            ties_y += 1
        elif dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1
    n0 = len(x) * (len(x) - 1) // 2
    denom = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    return (concordant - discordant) / denom if denom > 0 else None


def ndcg(scores, grades):
    """NDCG over the whole list, gains 2^g - 1; None if no reference has a positive grade."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ideal = sorted(grades, reverse=True)
    dcg = sum((2 ** grades[i] - 1) / math.log2(rank + 2) for rank, i in enumerate(order))
    idcg = sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _score(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else MISSING_SCORE


# ---------------------------------------------------------------------------
# One main paper
# ---------------------------------------------------------------------------
def evaluate_baseline(gt_path, bl_path, methods=METHODS, aspects=ASPECTS):
    """Evaluate every method's per-aspect ranking for one main paper.

    Returns:
        {"m_id", "n_refs" (evaluated), "n_refs_gt", "n_refs_bl",
         "methods": {method: {
             "aspects": {aspect: {"pairwise_acc", "n_pairs", "concordant",
                                  "kendall_tau_b", "ndcg", "n_refs"}},
             "pooled":      {"pairwise_acc", "n_pairs", "concordant", "kendall_tau_b", "ndcg"},
             "pooled_excl": {...same...}}}}
    """
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    with open(bl_path, encoding="utf-8") as f:
        bl = json.load(f)

    gt_labels = {p["r_id"]: p.get("labels") or {} for p in gt.get("pairs", [])}
    refs = [r for r in gt_labels if r in bl]
    result = {
        "m_id": gt.get("m_id") or Path(gt_path).name[:-len(GT_SUFFIX)],
        "n_refs": len(refs), "n_refs_gt": len(gt_labels), "n_refs_bl": len(bl),
        "methods": {},
    }

    for method in methods:
        per_aspect = {}
        for aspect in aspects:
            # references with a valid ground-truth grade for this aspect
            rows = [(r, GRADES[gt_labels[r][aspect]]) for r in refs
                    if gt_labels[r].get(aspect) in GRADES]
            grades = [g for _, g in rows]
            scores = [_score((bl[r].get(method) or {}).get(aspect)) for r, _ in rows]
            concordant, n_pairs = pairwise_counts(scores, grades)
            # tau-b / NDCG need finite numbers: map "missing" below every real score
            finite = [s for s in scores if s != MISSING_SCORE]
            floor = (min(finite) - 1.0) if finite else 0.0
            ranked = [s if s != MISSING_SCORE else floor for s in scores]
            per_aspect[aspect] = {
                "pairwise_acc": concordant / n_pairs if n_pairs else None,
                "n_pairs": n_pairs,
                "concordant": concordant,
                "kendall_tau_b": kendall_tau_b(ranked, grades) if len(rows) > 1 else None,
                "ndcg": ndcg(ranked, grades) if rows else None,
                "n_refs": len(rows),
            }
        result["methods"][method] = {
            "aspects": per_aspect,
            "pooled": _pool(per_aspect, aspects),
            "pooled_excl": _pool(per_aspect, [a for a in aspects if a not in POOLED_EXCLUDE]),
        }
    return result


def _pool(per_aspect, aspects):
    concordant = sum(per_aspect[a]["concordant"] for a in aspects)
    n_pairs = sum(per_aspect[a]["n_pairs"] for a in aspects)
    return {
        "pairwise_acc": concordant / n_pairs if n_pairs else None,
        "n_pairs": n_pairs,
        "concordant": concordant,
        "kendall_tau_b": _mean(per_aspect[a]["kendall_tau_b"] for a in aspects),
        "ndcg": _mean(per_aspect[a]["ndcg"] for a in aspects),
    }


# ---------------------------------------------------------------------------
# Many main papers
# ---------------------------------------------------------------------------
def evaluate_baselines(gt_dir, bl_dir, methods=METHODS, aspects=ASPECTS, progress_every=500):
    """Evaluate all <m_id>_gt.json in gt_dir against <m_id>_bl.json in bl_dir.

    Returns:
        {"n_gt_files", "n_evaluated", "missing_bl": [m_id...], "failed": {m_id: error},
         "methods": {method: {
             "aspects":     {aspect: {"macro": {...}, "micro": {...}}},
             "pooled":      {"macro": {...}, "micro": {...}},
             "pooled_excl": {"macro": {...}, "micro": {...}}}}}
        macro = {"pairwise_acc", "kendall_tau_b", "ndcg", "n_papers"} (means of per-paper values)
        micro = {"pairwise_acc", "n_pairs"} (all pairs pooled)
    """
    gt_files = sorted(Path(gt_dir).glob(f"*{GT_SUFFIX}"))
    blocks = ["pooled", "pooled_excl"]
    acc = {m: {key: {"pa": [], "tau": [], "ndcg": [], "conc": 0.0, "pairs": 0}
               for key in [("aspect", a) for a in aspects] + [("block", b) for b in blocks]}
           for m in methods}
    missing_bl, failed, n_evaluated = [], {}, 0

    for k, gt_path in enumerate(gt_files, 1):
        m_id = gt_path.name[:-len(GT_SUFFIX)]
        bl_path = Path(bl_dir) / f"{m_id}{BL_SUFFIX}"
        if not bl_path.exists():
            missing_bl.append(m_id)
            continue
        try:
            res = evaluate_baseline(gt_path, bl_path, methods, aspects)
        except Exception as e:  # malformed file: log and continue
            failed[m_id] = f"{type(e).__name__}: {e}"
            continue
        n_evaluated += 1
        for m in methods:
            mres = res["methods"][m]
            items = [(("aspect", a), mres["aspects"][a]) for a in aspects] + \
                    [(("block", b), mres[b]) for b in blocks]
            for key, r in items:
                a = acc[m][key]
                a["pa"].append(r["pairwise_acc"])
                a["tau"].append(r["kendall_tau_b"])
                a["ndcg"].append(r["ndcg"])
                a["conc"] += r["concordant"]
                a["pairs"] += r["n_pairs"]
        if progress_every and k % progress_every == 0:
            print(f"  evaluated {k}/{len(gt_files)} ground-truth files")

    def summarise(a):
        return {
            "macro": {"pairwise_acc": _mean(a["pa"]), "kendall_tau_b": _mean(a["tau"]),
                      "ndcg": _mean(a["ndcg"]),
                      "n_papers": sum(v is not None for v in a["pa"])},
            "micro": {"pairwise_acc": a["conc"] / a["pairs"] if a["pairs"] else None,
                      "n_pairs": a["pairs"]},
        }

    return {
        "n_gt_files": len(gt_files),
        "n_evaluated": n_evaluated,
        "missing_bl": missing_bl,
        "failed": failed,
        "methods": {m: {"aspects": {a: summarise(acc[m][("aspect", a)]) for a in aspects},
                        "pooled": summarise(acc[m][("block", "pooled")]),
                        "pooled_excl": summarise(acc[m][("block", "pooled_excl")])}
                    for m in methods},
    }


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def _f(v):
    return f"{v:.2f}" if v is not None else "  - "


def format_table(result, level="macro"):
    """Text table of pairwise accuracy per aspect + pooled scores.

    result: output of evaluate_baseline() (one paper) or evaluate_baselines()
    (level = "macro" or "micro"; micro has pairwise accuracy only).
    """
    many = "n_evaluated" in result
    aspects = list(next(iter(result["methods"].values()))["aspects"])
    head = f"{'method':18}" + "".join(f"{a[:6]:>8}" for a in aspects) + \
        f"{'pooled':>8}{'pool-ex':>9}{'tau-ex':>8}{'ndcg-ex':>9}"
    if many:
        title = (f"{level} over {result['n_evaluated']} paper(s) "
                 f"(missing prediction: {len(result['missing_bl'])}, failed: {len(result['failed'])})")
    else:
        title = f"{result['m_id']}: {result['n_refs']} reference(s)"
    lines = [title, "pairwise accuracy per aspect; -ex = without " + ", ".join(POOLED_EXCLUDE), head]
    for method, mres in result["methods"].items():
        def get(block, key):
            r = block[level] if many else block
            return r.get(key)
        row = f"{method:18}" + "".join(f"{_f(get(mres['aspects'][a], 'pairwise_acc')):>8}" for a in aspects)
        row += f"{_f(get(mres['pooled'], 'pairwise_acc')):>8}{_f(get(mres['pooled_excl'], 'pairwise_acc')):>9}"
        row += f"{_f(get(mres['pooled_excl'], 'kendall_tau_b')):>8}{_f(get(mres['pooled_excl'], 'ndcg')):>9}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-aspect ranking evaluation against ground-truth grades")
    ap.add_argument("--gt", required=True, help=f"A ground-truth file (*{GT_SUFFIX}) or a directory of them")
    ap.add_argument("--bl", required=True, help=f"A prediction/baseline file (*{BL_SUFFIX}) or a directory of them")
    ap.add_argument("--out", default=None, help="Write the result dict to this JSON file")
    args = ap.parse_args()

    if Path(args.gt).is_dir():
        result = evaluate_baselines(args.gt, args.bl)
        print(format_table(result, "macro"))
        print()
        print(format_table(result, "micro"))
    else:
        result = evaluate_baseline(args.gt, args.bl)
        print(format_table(result))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote -> {args.out}")
