"""
baseline_main.py

Driver for baseline comparison against FACET. For every main paper M (one FACET
output JSON), builds or extends <output_dir>/<m_id>_bl.json with the baselines in BASELINES.

Incremental: if <bl_dir>/<m_id>_bl.json already exists (bl_dir defaults to output_dir), it is
loaded and only the methods of BASELINES whose key is missing (in any reference) are computed;
existing keys are kept unchanged. A paper with nothing missing is not rewritten. Embedding
models are loaded only when some paper needs them.

Execution (per chunk of --chunk-size main papers):
    - the next chunk's JSONs are read in an I/O thread while the current chunk is computed
    - per embedding baseline, the unique aspect texts of the chunk's papers missing it are
      encoded in ONE encode() call; the models run in parallel GPU threads
      (length-sorted batches of --batch-size, fp16 on CUDA)
    - BM25 (CPU) runs in a worker thread while the GPU encodes
    - each paper's record is read once and its _bl.json is written once

Functions:
    initialize      FACET record -> (bl dict, <output_dir>/<m_id>_bl.json path) holding, per
                    reference, the FACET facet_similarity, percentile (layer 1) and
                    context (layer 3) vectors
    load_existing   bl from <bl_dir>/<m_id>_bl.json if present, else initialize()
    missing_methods methods of BASELINES whose key is absent in any reference of bl
    load_record     (path, record or exception) -- safe JSON read for the prefetch thread
    run_chunk       missing baselines for a chunk of main papers; writes _bl.json files
    main            CLI: a FACET JSON file or a directory of them; loads models lazily

Output (<m_id>_bl.json):
    {
      "<r_id>": {
        "facet_similarity": {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   F_mr = V_mr + beta * W_mr
        "percentile":       {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   layer 1 score V_mr
        "context":          {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},   layer 3 score W_mr = mask * idf_weight
        "bm25":             {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "specter2":         {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "scincl":           {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "scibert":          {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
      },
      ...
    }

Usage:
    python -m baseline_main --input data/facet_scores/0704.0262.json --output_dir data/baselines
    python -m baseline_main --input data/facet_scores --output_dir data/baselines --chunk-size 32 --batch-size 128
    python -m baseline_main --input data/facet_scores --bl_dir data/baselines --output_dir data/baselines_v2
"""

import argparse
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.baseline_methods import (ASPECTS, ROUND, aspect_texts, bm25, embedding_cosine,
                                    iter_r_entries, load_json, save_json)

logger = logging.getLogger(__name__)

# Baselines to have in every _bl.json; missing ones are computed. Key = _bl.json key.
BASELINES = ["bm25", "specter2", "scincl", "scibert"]
# Embedding baselines: key -> encoder class name in utils.embedding_utils
EMBEDDING_MODELS = {"specter2": "Specter2Model", "scincl": "SciNCLModel", "scibert": "SciBERTModel"}


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


def load_existing(record, output_dir, bl_dir, paper_json=None):
    """(bl, out_path, is_new): bl from <bl_dir>/<m_id>_bl.json if it exists, else initialize()."""
    bl, out_path = initialize(record, output_dir, paper_json)
    existing = Path(bl_dir) / out_path.name
    if existing.is_file():
        return load_json(existing), out_path, False
    return bl, out_path, True


def missing_methods(bl, methods=BASELINES):
    """Methods whose key is absent in at least one reference of bl."""
    return [m for m in methods if any(m not in v for v in bl.values())]


def load_record(path):
    """(path, record) or (path, exception) -- never raises."""
    try:
        return path, load_json(path)
    except Exception as exc:
        return path, exc


class ModelCache:
    """Embedding models, created on first use (sequentially, in the calling thread)."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.models = {}

    def get(self, key):
        if key not in self.models:
            from utils import embedding_utils
            logger.info("loading %s", EMBEDDING_MODELS[key])
            self.models[key] = getattr(embedding_utils, EMBEDDING_MODELS[key])(batch_size=self.batch_size)
        return self.models[key]


def run_chunk(loaded, output_dir, bl_dir, models, cpu_pool, gpu_pool, io_pool):
    """Missing baselines for one chunk of loaded (path, record) pairs.
    Returns (list of written _bl.json paths, number of failed papers, number skipped)."""
    papers, failed, skipped = [], 0, 0
    for path, rec in loaded:
        if isinstance(rec, Exception):
            failed += 1
            logger.error("%s: %s", path.name, rec)
            continue
        try:
            bl, out_path, is_new = load_existing(rec, output_dir, bl_dir, path)
        except Exception:
            failed += 1
            logger.error("%s: %s", path.name, traceback.format_exc())
            continue
        todo = missing_methods(bl)
        if not todo and not is_new and Path(bl_dir) == Path(output_dir):
            skipped += 1
            continue
        papers.append((path, rec, bl, out_path, todo))
    if not papers:
        return [], failed, skipped

    # GPU: per embedding baseline, one encode() over the chunk's papers missing it
    futures = {}
    for key in EMBEDDING_MODELS:
        texts = set().union(*(aspect_texts(rec) for _, rec, _, _, todo in papers if key in todo))
        if texts:
            futures[key] = gpu_pool.submit(models.get(key).encode, texts)

    # CPU: BM25 while the GPU encodes
    def bm25_all():
        errors = {}
        for path, rec, bl, _, todo in papers:
            if "bm25" in todo:
                try:
                    bm25(rec, bl)
                except Exception:
                    errors[path] = traceback.format_exc()
        return errors

    bm25_errors = cpu_pool.submit(bm25_all).result()
    vecs = {key: fut.result() for key, fut in futures.items()}

    written, saves = [], []
    for path, rec, bl, out_path, todo in papers:
        if path in bm25_errors:
            failed += 1
            logger.error("%s: %s", path.name, bm25_errors[path])
            continue
        try:
            for key in todo:
                if key in EMBEDDING_MODELS:
                    embedding_cosine(rec, bl, vecs[key], key)
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
    return written, failed, skipped


def main():
    parser = argparse.ArgumentParser(description="Baselines for FACET ranking quality.")
    parser.add_argument("--input", required=True, help="A FACET output JSON file or a directory of them")
    parser.add_argument("--output_dir", required=True, help="Where <m_id>_bl.json files are written")
    parser.add_argument("--bl_dir", default=None,
                        help="Existing <m_id>_bl.json files to extend (default: output_dir)")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--chunk-size", type=int, default=32, help="Main papers per encode() call")
    parser.add_argument("--batch-size", type=int, default=128, help="Texts per model forward pass")
    args = parser.parse_args()
    bl_dir = args.bl_dir or args.output_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = Path(args.input)
    files = [p] if p.is_file() else sorted(p.glob(args.pattern))
    chunks = [files[i:i + args.chunk_size] for i in range(0, len(files), args.chunk_size)]
    models = ModelCache(args.batch_size)

    done = failed = skipped = 0
    with ThreadPoolExecutor(1) as cpu_pool, ThreadPoolExecutor(len(EMBEDDING_MODELS)) as gpu_pool, \
            ThreadPoolExecutor(4) as io_pool:
        pending = [io_pool.submit(load_record, f) for f in chunks[0]] if chunks else []
        for k in range(len(chunks)):
            loaded = [fut.result() for fut in pending]
            pending = [io_pool.submit(load_record, f) for f in chunks[k + 1]] if k + 1 < len(chunks) else []
            try:
                written, n_failed, n_skipped = run_chunk(loaded, args.output_dir, bl_dir, models,
                                                         cpu_pool, gpu_pool, io_pool)
            except Exception:
                failed += len(loaded)
                logger.error("chunk %d failed: %s", k, traceback.format_exc())
                continue
            done += len(written)
            failed += n_failed
            skipped += n_skipped
            logger.info("chunk %d/%d: %d written, %d skipped (complete), %d failed",
                        k + 1, len(chunks), len(written), n_skipped, n_failed)
    logger.info("files=%d done=%d skipped=%d failed=%d", len(files), done, skipped, failed)


if __name__ == "__main__":
    main()
