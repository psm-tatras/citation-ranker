"""
baseline_main.py

Driver for baseline comparison against FACET. For every main paper M (one FACET
output JSON), builds <output_dir>/<m_id>_bl.json and runs each baseline on it.

Execution (per chunk of --chunk-size main papers):
    - the next chunk's JSONs are read in an I/O thread while the current chunk is computed
    - all unique aspect texts of the chunk are encoded in ONE encode() call per model;
      SPECTER2 and SciNCL run in two GPU threads at the same time (length-sorted
      batches of --batch-size, fp16 on CUDA)
    - BM25 (CPU) runs in a worker thread while the GPU encodes
    - each paper's record is read once and its _bl.json is written once

Functions:
    initialize      FACET record -> (bl dict, <output_dir>/<m_id>_bl.json path) holding, per
                    reference, the FACET facet_similarity, percentile (layer 1) and
                    context (layer 3) vectors
    load_record     (path, record or exception) -- safe JSON read for the prefetch thread
    run_chunk       initialize() + every baseline for a chunk of main papers; writes _bl.json files
    main            CLI: a FACET JSON file or a directory of them; loads models once

Output (<m_id>_bl.json):
    {
      "<r_id>": {
        "facet_similarity": {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   F_mr = W_mr (.) V_mr
        "percentile":       {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   layer 1 score V_mr
        "context":          {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   layer 3 score W_mr = mask * idf_weight
        "bm25":             {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "specter2":         {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "scincl":           {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
      },
      ...
    }

Usage:
    python -m baseline_main --input data/facet_scores/0704.0262.json --output_dir data/baselines
    python -m baseline_main --input data/facet_scores --output_dir data/baselines --chunk-size 32 --batch-size 128
"""

import argparse
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.baseline_methods import (ASPECTS, ROUND, aspect_texts, bm25, iter_r_entries, load_json,
                                    save_json, scincl, specter2)

logger = logging.getLogger(__name__)


def initialize(record, output_dir, paper_json=None):
    """(bl, out_path): bl = {r_id: {facet_similarity, percentile, context}} from a FACET record;
    out_path = <output_dir>/<m_id>_bl.json (written by run_chunk after all baselines).

    context[a] = mask[a] * idf_weight[a]  (W_mr; mask per R from facet_score,
    idf_weight per M from facet_scoring). Aspects missing either value are left out."""
    m_id = record.get("m_id") or Path(paper_json).stem
    idf_weight = (record.get("facet_scoring") or {}).get("idf_weight") or {}
    bl = {}
    for r_id, entry in iter_r_entries(record):
        fs = entry.get("facet_score") or {}
        mask = fs.get("mask") or {}
        bl[r_id] = {
            "facet_similarity": fs.get("facet_similarity", {}),
            "percentile": fs.get("percentile", {}),
            "context": {a: round(mask[a] * idf_weight[a], ROUND)
                               for a in ASPECTS if a in mask and a in idf_weight},
        }
    return bl, Path(output_dir) / f"{m_id.replace('/', '_')}_bl.json"


def load_record(path):
    """(path, record) or (path, exception) -- never raises."""
    try:
        return path, load_json(path)
    except Exception as exc:
        return path, exc


def run_chunk(loaded, output_dir, specter2_model, scincl_model, cpu_pool, gpu_pool, io_pool):
    """All baselines for one chunk of loaded (path, record) pairs.
    Returns (list of written _bl.json paths, number of failed papers)."""
    papers, failed = [], 0
    for path, rec in loaded:
        if isinstance(rec, Exception):
            failed += 1
            logger.error("%s: %s", path.name, rec)
            continue
        bl, out_path = initialize(rec, output_dir, path)
        papers.append((path, rec, bl, out_path))
    if not papers:
        return [], failed

    # GPU: one encode() per model over the whole chunk, both models at the same time
    texts = set().union(*(aspect_texts(rec) for _, rec, _, _ in papers))
    f_spec = gpu_pool.submit(specter2_model.encode, texts)
    f_scincl = gpu_pool.submit(scincl_model.encode, texts)

    # CPU: BM25 while the GPU encodes
    def bm25_all():
        errors = {}
        for path, rec, bl, _ in papers:
            try:
                bm25(rec, bl)
            except Exception:
                errors[path] = traceback.format_exc()
        return errors

    bm25_errors = cpu_pool.submit(bm25_all).result()
    v_spec, v_scincl = f_spec.result(), f_scincl.result()

    written, saves = [], []
    for path, rec, bl, out_path in papers:
        if path in bm25_errors:
            failed += 1
            logger.error("%s: %s", path.name, bm25_errors[path])
            continue
        try:
            specter2(rec, bl, v_spec)
            scincl(rec, bl, v_scincl)
        except Exception:
            failed += 1
            logger.error("%s: %s", path.name, traceback.format_exc())
            continue
        saves.append((path, out_path, io_pool.submit(save_json, bl, out_path)))
    for path, out_path, fut in saves:
        try:
            fut.result()
            written.append(out_path)
        except Exception as exc:
            failed += 1
            logger.error("%s: write failed: %s", path.name, exc)
    return written, failed


def main():
    parser = argparse.ArgumentParser(description="Baselines for FACET ranking quality.")
    parser.add_argument("--input", required=True, help="A FACET output JSON file or a directory of them")
    parser.add_argument("--output_dir", required=True, help="Where <m_id>_bl.json files are written")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--chunk-size", type=int, default=32, help="Main papers per encode() call")
    parser.add_argument("--batch-size", type=int, default=128, help="Texts per model forward pass")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = Path(args.input)
    files = [p] if p.is_file() else sorted(p.glob(args.pattern))
    chunks = [files[i:i + args.chunk_size] for i in range(0, len(files), args.chunk_size)]

    from utils.embedding_utils import SciNCLModel, Specter2Model
    specter2_model = Specter2Model(batch_size=args.batch_size)
    scincl_model = SciNCLModel(batch_size=args.batch_size)

    done = failed = 0
    with ThreadPoolExecutor(1) as cpu_pool, ThreadPoolExecutor(2) as gpu_pool, \
            ThreadPoolExecutor(4) as io_pool:
        pending = [io_pool.submit(load_record, f) for f in chunks[0]] if chunks else []
        for k in range(len(chunks)):
            loaded = [fut.result() for fut in pending]
            pending = [io_pool.submit(load_record, f) for f in chunks[k + 1]] if k + 1 < len(chunks) else []
            try:
                written, n_failed = run_chunk(loaded, args.output_dir, specter2_model, scincl_model,
                                              cpu_pool, gpu_pool, io_pool)
            except Exception:
                failed += len(loaded)
                logger.error("chunk %d failed: %s", k, traceback.format_exc())
                continue
            done += len(written)
            failed += n_failed
            logger.info("chunk %d/%d: %d written, %d failed", k + 1, len(chunks), len(written), n_failed)
    logger.info("files=%d done=%d failed=%d", len(files), done, failed)


if __name__ == "__main__":
    main()
