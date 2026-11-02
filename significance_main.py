"""
significance_main.py

Paired significance testing of FACET (V_mr) against every baseline, with main papers as the
unit of analysis (the macro numbers of evaluation_main are means over main papers).

Per-paper values come from evaluation_main.evaluate_baseline (same metrics, same files):
    pairwise_acc, kendall_tau_b, ndcg  for every aspect, "pooled" and "pooled_excl".

For a reference method A (FACET) and a baseline B, per block (aspect or pooled) and metric:
    papers    P = main papers where both A and B have a defined value
    d_p       = A_p - B_p                                   (paired difference per paper)
    diff      = mean_p d_p                                  (= macro(A) - macro(B) on P)
    CI        = paired bootstrap 95% interval of diff       (resample papers, N_BOOT times)
    p         = two-sided paired randomization test         (random sign flip of each d_p,
                (count(|mean(s*d)| >= |diff|) + 1) / (N_PERM + 1))
    p_holm    = Holm-Bonferroni adjusted p over all tests of the same block
                (baselines x metrics)
    win/tie/loss = papers with d_p > 0 / = 0 / < 0
Significant (marked *) when p_holm < ALPHA.

Per-paper values can be saved (--per-paper) and reused (--from-per-paper) so the tests can be
re-run without re-evaluating.

Usage:
    python significance_main.py --gt data/gt --bl data/bl --out data/significance.json \
        --per-paper data/per_paper.jsonl --workers 4
    python significance_main.py --from-per-paper data/per_paper.jsonl --out data/significance.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluation_main import ASPECTS, BL_SUFFIX, GT_SUFFIX, evaluate_baseline

# ---------------------------------------------------------------------------
# Configuration (hard-coded; edit here)
# ---------------------------------------------------------------------------
REFERENCE = "percentile"          # FACET V_mr
BASELINES = ["bm25", "specter2", "scincl", "scibert", "qwen3_aspect",                      # family 1
             "bm25_abstract", "specter2_abstract", "scincl_abstract", "scibert_abstract",   # family 2
             "qwen3_abstract"]
METRICS = ["pairwise_acc", "kendall_tau_b", "ndcg"]
BLOCKS = ["pooled_excl", "pooled"] + ASPECTS
N_BOOT = 10000
N_PERM = 10000
ALPHA = 0.05
SEED = 13
BATCH = 500                       # resamples per numpy batch (memory: BATCH x n_papers)


# ---------------------------------------------------------------------------
# Per-paper values
# ---------------------------------------------------------------------------
def paper_values(gt_path, bl_path, methods):
    """{"m_id", "values": {method: {block: {metric: value or None}}}} for one main paper."""
    res = evaluate_baseline(gt_path, bl_path, methods)
    values = {}
    for m in methods:
        mres = res["methods"][m]
        values[m] = {b: {k: (mres[b] if b in ("pooled", "pooled_excl") else mres["aspects"][b])[k]
                         for k in METRICS} for b in BLOCKS}
    return {"m_id": res["m_id"], "values": values}


def _job(args):
    gt_path, bl_path, methods = args
    try:
        return paper_values(gt_path, bl_path, methods)
    except Exception as e:  # malformed file: reported, not fatal
        return {"m_id": Path(gt_path).name[:-len(GT_SUFFIX)], "error": f"{type(e).__name__}: {e}"}


def collect(gt_dir, bl_dir, methods, workers=1):
    """Per-paper values for every <m_id>_gt.json with a matching <m_id>_bl.json."""
    jobs = []
    for gt_path in sorted(Path(gt_dir).glob(f"*{GT_SUFFIX}")):
        bl_path = Path(bl_dir) / (gt_path.name[:-len(GT_SUFFIX)] + BL_SUFFIX)
        if bl_path.exists():
            jobs.append((str(gt_path), str(bl_path), methods))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_job, jobs, chunksize=64))
    else:
        rows = [_job(j) for j in jobs]
    errors = {r["m_id"]: r["error"] for r in rows if "error" in r}
    return [r for r in rows if "error" not in r], errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def paired_test(a, b, rng, n_boot=N_BOOT, n_perm=N_PERM):
    """Paired bootstrap CI + randomization test on mean(a - b). a, b: 1-d arrays (same papers)."""
    d = a - b
    n = len(d)
    diff = float(d.mean())

    boots = []
    for start in range(0, n_boot, BATCH):
        k = min(BATCH, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        boots.append(d[idx].mean(axis=1))
    boots = np.concatenate(boots)

    extreme = 0
    for start in range(0, n_perm, BATCH):
        k = min(BATCH, n_perm - start)
        signs = rng.integers(0, 2, size=(k, n), dtype=np.int8) * 2 - 1
        extreme += int((np.abs((signs * d).mean(axis=1)) >= abs(diff) - 1e-12).sum())

    return {
        "n_papers": n,
        "reference": float(a.mean()), "baseline": float(b.mean()),
        "diff": diff,
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
        "p": (extreme + 1) / (n_perm + 1),
        "win": int((d > 0).sum()), "tie": int((d == 0).sum()), "loss": int((d < 0).sum()),
    }


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def significance(rows, reference=REFERENCE, baselines=BASELINES, blocks=BLOCKS, metrics=METRICS,
                 seed=SEED):
    """{block: {baseline: {metric: test dict}}} with Holm adjustment per block."""
    rng = np.random.default_rng(seed)
    out = {}
    for block in blocks:
        out[block] = {}
        tests = []
        for base in baselines:
            out[block][base] = {}
            for metric in metrics:
                pairs = [(r["values"][reference][block][metric], r["values"][base][block][metric])
                         for r in rows]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    out[block][base][metric] = None
                    continue
                a, b = (np.array(v, dtype=float) for v in zip(*pairs))
                t = paired_test(a, b, rng)
                out[block][base][metric] = t
                tests.append(t)
        for t, p_adj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = p_adj
            t["significant"] = p_adj < ALPHA
    return out


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def format_table(result, block="pooled_excl", metrics=METRICS):
    """Text table: FACET - baseline, 95% CI, Holm p, win/tie/loss, for one block."""
    lines = [f"{REFERENCE} vs baselines, block = {block} (diff = {REFERENCE} - baseline; "
             f"* = Holm p < {ALPHA})"]
    for metric in metrics:
        lines.append(f"\n{metric}")
        lines.append(f"{'baseline':18}{'ref':>7}{'base':>7}{'diff':>8}{'95% CI':>18}"
                     f"{'p_holm':>9}{'win/tie/loss':>20}{'n':>7}")
        for base, per_metric in result[block].items():
            t = per_metric.get(metric)
            if t is None:
                lines.append(f"{base:18}  (no values)")
                continue
            star = "*" if t["significant"] else " "
            lines.append(f"{base:18}{t['reference']:7.3f}{t['baseline']:7.3f}{t['diff']:+8.3f}"
                         f"  [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]{t['p_holm']:9.4f}{star}"
                         f"{t['win']:>8}/{t['tie']}/{t['loss']:<6}{t['n_papers']:>7}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paired significance tests: FACET vs baselines")
    ap.add_argument("--gt", help=f"Directory of ground-truth files (*{GT_SUFFIX})")
    ap.add_argument("--bl", help=f"Directory of prediction/baseline files (*{BL_SUFFIX})")
    ap.add_argument("--per-paper", default=None, help="Write per-paper values to this JSONL file")
    ap.add_argument("--from-per-paper", default=None, help="Read per-paper values instead of --gt/--bl")
    ap.add_argument("--out", default=None, help="Write the test results to this JSON file")
    ap.add_argument("--workers", type=int, default=1, help="Processes for per-paper evaluation")
    args = ap.parse_args()

    methods = [REFERENCE] + BASELINES
    if args.from_per_paper:
        with open(args.from_per_paper, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        errors = {}
    else:
        if not (args.gt and args.bl):
            ap.error("--gt and --bl are required unless --from-per-paper is given")
        rows, errors = collect(args.gt, args.bl, methods, args.workers)
        if args.per_paper:
            Path(args.per_paper).parent.mkdir(parents=True, exist_ok=True)
            with open(args.per_paper, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    print(f"papers: {len(rows)}, failed: {len(errors)}")

    result = significance(rows)
    print(format_table(result, "pooled_excl"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"reference": REFERENCE, "n_papers": len(rows), "failed": errors,
                       "n_boot": N_BOOT, "n_perm": N_PERM, "alpha": ALPHA, "seed": SEED,
                       "results": result}, f, indent=2)
        print(f"Wrote -> {args.out}")
