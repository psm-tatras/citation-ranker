"""
prediction/qwen_embedding.py

Aspect-level cosine similarity between a main paper M and each cited paper R.

Reads the per-M JSON files produced by Layer 1, embeds TASK / CONTRIBUTION /
METHOD / RESULT with Qwen3-Embedding (aspect-specific instruction), and writes
an `aspect_similarity` dict right after `aspect` in every R entry of the same
JSON. Embedding vectors are not stored.

Similarity is raw cosine in [-1, 1]; None when either side's aspect is empty.

Model defaults to the QWEN_EMBEDDING_MODEL_NAME env var (see .env -- e.g.
Qwen/Qwen3-Embedding-4B or the smaller Qwen/Qwen3-Embedding-0.6B); --model
overrides it explicitly.

Loaded once from the HF Hub, then saved to QWEN_EMBEDDING_MODEL_LOCAL_DIR
(see .env) and loaded from there on every later run -- this model is only
used for inference (never fine-tuned), so its weights are static and don't
need re-downloading each time.

Usage:
    python -m prediction.qwen_embedding --input_dir data/relations
    python -m prediction.qwen_embedding --input_dir data/relations --model Qwen/Qwen3-Embedding-0.6B -n 10 --overwrite
    python -m prediction.qwen_embedding --input_file data/relations/2308.01689.json --overwrite
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.prompts import ASPECT_KEYS

logger = logging.getLogger(__name__)


class AspectSimilarity:
    ASPECTS = ASPECT_KEYS

    INSTRUCTIONS = {
    "TASK": "Represent the research task or problem addressed by a scientific paper, "
            "for comparison with the tasks of other papers",
    "LIMITATION": "Represent the limitation, gap or open problem stated in a scientific paper, "
                  "for comparison with the limitations of other papers",
    "CONTRIBUTION": "Represent the main contribution of a scientific paper, "
                    "for comparison with the contributions of other papers",
    "METHOD": "Represent the method or approach used in a scientific paper, "
              "for comparison with the methods of other papers",
    "RESULT": "Represent the findings or results of a scientific paper, "
              "for comparison with the results of other papers",
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 256,
        local_dir: str | None = None,
    ):
        model_name = model_name or os.getenv("QWEN_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
        local_dir = local_dir or os.getenv(
            "QWEN_EMBEDDING_MODEL_LOCAL_DIR",
            str(Path(__file__).resolve().parent.parent / "model" / "embedding" / "qwen-4B"))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        local_path = Path(local_dir)
        cached = (local_path / "config.json").is_file()
        if cached:
            logger.info("Loading %s from local cache %s", model_name, local_path)
        else:
            logger.info("No local cache at %s; downloading %s", local_path, model_name)
        self.model = SentenceTransformer(
            str(local_path) if cached else model_name,
            device=device,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        logger.info("Loaded %s on %s (%s)", model_name, device, dtype)

        if not cached:
            local_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(local_path))
            logger.info("Saved %s to %s for future runs", model_name, local_path)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def clean(text) -> str | None:
        if not isinstance(text, str):
            return None
        text = " ".join(text.split())
        return text or None

    def prompt(self, aspect: str) -> str:
        return f"Instruct: {self.INSTRUCTIONS[aspect]}\nQuery:"

    def encode(self, aspect: str, texts: list[str]) -> dict[str, torch.Tensor]:
        """Encode unique texts for one aspect; returns {text: unit vector (float32, cpu)}."""
        unique = list(dict.fromkeys(texts))
        if not unique:
            return {}
        vecs = self.model.encode(
            unique,
            prompt=self.prompt(aspect),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        vecs = vecs.float().cpu()
        return dict(zip(unique, vecs))

    @staticmethod
    def iter_r_entries(record: dict):
        """Yield every R entry dict: references -> {R: [ {R: entry}, ... ]}."""
        for r_id, items in (record.get("references") or {}).items():
            for item in items if isinstance(items, list) else [items]:
                entry = item.get(r_id) if isinstance(item, dict) else None
                if isinstance(entry, dict):
                    yield entry

    @staticmethod
    def insert_after(entry: dict, anchor: str, key: str, value) -> None:
        """Set entry[key] = value, placed immediately after `anchor` (in place)."""
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

    # ------------------------------------------------------------------- core
    def process_record(self, record: dict, overwrite: bool = False) -> int:
        """Add aspect_similarity to every R entry of one M record. Returns #entries updated."""
        m_aspects = record.get("m_aspects") or {}
        entries = [
            e for e in self.iter_r_entries(record)
            if overwrite or "aspect_similarity" not in e
        ]
        if not entries:
            return 0

        sims = [dict.fromkeys(self.ASPECTS) for _ in entries]
        for aspect in self.ASPECTS:
            m_text = self.clean(m_aspects.get(aspect))
            if m_text is None:
                continue
            r_texts = [self.clean((e.get("aspect") or {}).get(aspect)) for e in entries]
            vecs = self.encode(aspect, [m_text] + [t for t in r_texts if t])
            m_vec = vecs[m_text]
            for sim, t in zip(sims, r_texts):
                if t is not None:
                    sim[aspect] = round(float(torch.dot(m_vec, vecs[t])), 4)

        for e, sim in zip(entries, sims):
            self.insert_after(e, "aspect", "aspect_similarity", sim)
        return len(entries)

    def process_file(self, path: Path, overwrite: bool = False) -> int:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        n = self.process_record(record, overwrite=overwrite)
        if n:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=3)
            os.replace(tmp, path)
        return n

    def run(self, input_dir: str | None = None, input_file: str | None = None,
            pattern: str = "*.json", limit: int | None = None, overwrite: bool = False) -> None:
        """Process either every file matching `pattern` under `input_dir`, or
        just the single `input_file` -- pass exactly one of the two."""
        if bool(input_dir) == bool(input_file):
            raise ValueError("pass exactly one of input_dir or input_file")

        if input_file:
            files = [Path(input_file)]
        else:
            files = sorted(Path(input_dir).glob(pattern))
            if limit:
                files = files[:limit]
        updated_files = updated_entries = failed = 0
        for path in tqdm(files, desc="aspect similarity"):
            try:
                n = self.process_file(path, overwrite=overwrite)
            except Exception as exc:  # keep going on bad files
                failed += 1
                logger.error("%s: %s", path.name, exc)
                continue
            if n:
                updated_files += 1
                updated_entries += n
        logger.info("files=%d updated_files=%d updated_R_entries=%d failed=%d",
                    len(files), updated_files, updated_entries, failed)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Add M-R aspect cosine similarity to relation JSONs.")
    parser.add_argument("--input_dir", default=None,
                         help="Directory with per-M JSON files (mutually exclusive with --input_file)")
    parser.add_argument("--input_file", default=None,
                         help="A single per-M JSON file to process (mutually exclusive with --input_dir)")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern within --input_dir (ignored with --input_file)")
    parser.add_argument("--model", default=None,
                         help="Defaults to QWEN_EMBEDDING_MODEL_NAME env var, else Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--local_dir", default=None,
                         help="Defaults to QWEN_EMBEDDING_MODEL_LOCAL_DIR env var, else "
                              "model/embedding/qwen-4B. Loaded from here if already cached, "
                              "else downloaded once and saved here for future runs.")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Process only first N files")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing aspect_similarity")
    args = parser.parse_args()
    if bool(args.input_dir) == bool(args.input_file):
        parser.error("pass exactly one of --input_dir or --input_file")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sim = AspectSimilarity(args.model, args.device, args.batch_size, args.max_seq_length, args.local_dir)
    try:
        sim.run(input_dir=args.input_dir, input_file=args.input_file,
                pattern=args.pattern, limit=args.limit, overwrite=args.overwrite)
    finally:
        sim.close()


if __name__ == "__main__":
    main()