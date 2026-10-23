"""
baseline_main.py

Driver for baseline comparison against FACET. For every main paper M (one FACET
output JSON), creates <output_dir>/<m_id>_bl.json and runs each baseline on it in turn.

Functions:
    initialize      paper JSON -> <output_dir>/<m_id>_bl.json holding, per reference,
                    the FACET facet_similarity and percentile vectors; returns its path
    run_baselines   initialize() + every baseline for one M
    main            CLI: a FACET JSON file or a directory of them; loads models once

Output (<m_id>_bl.json):
    {
      "<r_id>": {
        "facet_similarity": {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "percentile":       {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "bm25":             {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "specter2":         {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT},
        "scincl":           {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
      },
      ...
    }

Usage:
    python -m baseline_main --input data/facet_scores/0704.0262.json --output_dir data/baselines
    python -m baseline_main --input data/facet_scores --output_dir data/baselines
"""

import argparse
import logging
from pathlib import Path

from utils.baseline_methods import bm25, iter_r_entries, load_json, save_json, scincl, specter2

logger = logging.getLogger(__name__)


def initialize(paper_json, output_dir):
    """Create <output_dir>/<m_id>_bl.json from a FACET output JSON; returns its path."""
    record = load_json(paper_json)
    m_id = record.get("m_id") or Path(paper_json).stem
    out = {}
    for r_id, entry in iter_r_entries(record):
        fs = entry.get("facet_score") or {}
        out[r_id] = {
            "facet_similarity": fs.get("facet_similarity", {}),
            "percentile": fs.get("percentile", {}),
        }
    out_path = Path(output_dir) / f"{m_id.replace('/', '_')}_bl.json"
    save_json(out, out_path)
    return out_path


def run_baselines(paper_json, output_dir, specter2_model, scincl_model):
    """initialize() + every baseline for one main paper; returns the _bl.json path."""
    bl_json = initialize(paper_json, output_dir)
    bm25(paper_json, bl_json)
    specter2(paper_json, bl_json, specter2_model)
    scincl(paper_json, bl_json, scincl_model)
    return bl_json


def main():
    parser = argparse.ArgumentParser(description="Baselines for FACET ranking quality.")
    parser.add_argument("--input", required=True, help="A FACET output JSON file or a directory of them")
    parser.add_argument("--output_dir", required=True, help="Where <m_id>_bl.json files are written")
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = Path(args.input)
    files = [p] if p.is_file() else sorted(p.glob(args.pattern))

    from utils.embedding_utils import SciNCLModel, Specter2Model
    specter2_model = Specter2Model()
    scincl_model = SciNCLModel()

    done = failed = 0
    for path in files:
        try:
            out = run_baselines(path, args.output_dir, specter2_model, scincl_model)
        except Exception as exc:
            failed += 1
            logger.error("%s: %s", path.name, exc)
            continue
        done += 1
        logger.info("%s -> %s", path.name, out)
    logger.info("files=%d done=%d failed=%d", len(files), done, failed)


if __name__ == "__main__":
    main()
