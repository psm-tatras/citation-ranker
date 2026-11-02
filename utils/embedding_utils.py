"""
Text embedding utilities: load a HF encoder model and compute batched
embeddings, written out as {id: embedding} pickles.
"""

import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} env var is not set")
    return value


class EmbeddingModel:
    def __init__(self, device=None):
        self.model_name = _require_env("EMBEDDING_MODEL_NAME")
        self.local_dir = _require_env("EMBEDDING_MODEL_LOCAL_DIR")
        self.batch_size = int(_require_env("EMBEDDING_BATCH_SIZE"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.max_length = None

    def load_embedding_model(self):
        """Load the tokenizer/model from EMBEDDING_MODEL_LOCAL_DIR -- a plain
        local directory holding the model's own files (config.json,
        pytorch_model.bin, vocab.txt, ...), not an HF-hub cache dir. Loaded
        with local_files_only so this never tries to reach the network."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            self.local_dir, local_files_only=True).to(self.device)
        self.model.eval()
        # Some checkpoints (e.g. SciBERT) don't set tokenizer.model_max_length,
        # which then defaults to a huge sentinel -- truncation=True alone would
        # not actually cap input length, overrunning the model's real position
        # embedding limit. Fall back to the model config's own max positions.
        model_max_length = self.tokenizer.model_max_length
        if model_max_length is None or model_max_length > 100_000:
            model_max_length = getattr(
                self.model.config, "max_position_embeddings", 512)
        self.max_length = model_max_length
        return self.tokenizer, self.model

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def get_embedding_batch(self, id_text, output_path=None):
        """id_text: dict of {id: text}. Embeds texts in self.batch_size
        batches (mean-pooled last hidden state) and, if output_path is
        given, pickles {id: embedding} to it."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load_embedding_model() first")

        ids = list(id_text.keys())
        texts = [id_text[i] for i in ids]
        embeddings = {}

        with tqdm(total=len(texts), desc="Embedding", unit="text") as pbar:
            for start in range(0, len(texts), self.batch_size):
                batch_ids = ids[start:start + self.batch_size]
                batch_texts = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model(**enc)
                batch_emb = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                batch_emb = batch_emb.cpu().numpy()
                for id_, vec in zip(batch_ids, batch_emb):
                    embeddings[id_] = vec
                pbar.update(len(batch_ids))

        if output_path:
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"✓ Wrote {len(embeddings)} embeddings -> {output_path}")
        return embeddings


class ClsEncoder:
    """Shared encoder for the scientific-paper baselines (SPECTER2, SciNCL, SciBERT).

    encode() sorts texts by length (less padding), runs batches of batch_size,
    uses fp16 on CUDA, and returns L2-normalised vectors (float32, cpu), pooled
    by POOLING: "cls" (CLS token) or "mean" (mask-weighted mean of last hidden state).
    Subclasses set self.tokenizer / self.model and call self._finish_load().
    """

    MAX_LENGTH = 512
    POOLING = "cls"

    def _setup(self, local_dir, env_var, default_subdir, device, batch_size):
        self.local_dir = local_dir or os.environ.get(
            env_var,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "model", "embedding", default_subdir))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def _finish_load(self):
        if self.device.startswith("cuda"):
            self.model.half()
        self.model.to(self.device).eval()

    def encode(self, texts):
        """{text: L2-normalised CLS embedding (torch.Tensor, float32, cpu)} for unique texts."""
        texts = sorted(set(texts), key=len)
        out = {}
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.MAX_LENGTH,
                                 return_token_type_ids=False, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            if self.POOLING == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                vec = hidden[:, 0, :]
            vec = torch.nn.functional.normalize(vec.float(), dim=-1).cpu()
            out.update(zip(batch, vec))
        return out


class Specter2Model(ClsEncoder):
    """SPECTER2 encoder: allenai/specter2_base + proximity adapter allenai/specter2
    (adapters library), CLS pooling, L2-normalised vectors.

    First run downloads base + adapter from the HF Hub and saves them to
    local_dir (SPECTER2_LOCAL_DIR, default model/embedding/specter2); later
    runs load from local_dir only.
    """

    BASE_MODEL = "allenai/specter2_base"
    ADAPTER = "allenai/specter2"
    ADAPTER_NAME = "specter2"

    def __init__(self, local_dir=None, device=None, batch_size=128):
        from adapters import AutoAdapterModel  # only needed for SPECTER2

        self._setup(local_dir, "SPECTER2_LOCAL_DIR", "specter2", device, batch_size)
        adapter_dir = os.path.join(self.local_dir, "adapter")

        if os.path.isfile(os.path.join(self.local_dir, "config.json")) and os.path.isdir(adapter_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
            self.model = AutoAdapterModel.from_pretrained(self.local_dir, local_files_only=True)
            self.model.load_adapter(adapter_dir, load_as=self.ADAPTER_NAME, set_active=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
            self.model = AutoAdapterModel.from_pretrained(self.BASE_MODEL)
            self.model.load_adapter(self.ADAPTER, source="hf", load_as=self.ADAPTER_NAME, set_active=True)
            os.makedirs(self.local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(self.local_dir)
            self.model.save_pretrained(self.local_dir)
            self.model.save_adapter(adapter_dir, self.ADAPTER_NAME)
        # save_pretrained() also stores the adapter, so a local reload "overwrites" it
        # and set_active is dropped -- activate explicitly.
        self.model.set_active_adapters(self.ADAPTER_NAME)
        self._finish_load()


class SciNCLModel(ClsEncoder):
    """SciNCL encoder: malteos/scincl (plain BERT, no adapter), CLS pooling,
    L2-normalised vectors.

    First run downloads the model from the HF Hub and saves it to local_dir
    (SCINCL_LOCAL_DIR, default model/embedding/scincl); later runs load from
    local_dir only.
    """

    MODEL = "malteos/scincl"

    def __init__(self, local_dir=None, device=None, batch_size=128):
        self._setup(local_dir, "SCINCL_LOCAL_DIR", "scincl", device, batch_size)

        if os.path.isfile(os.path.join(self.local_dir, "config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.local_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModel.from_pretrained(self.MODEL)
            os.makedirs(self.local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(self.local_dir)
            self.model.save_pretrained(self.local_dir)
        self._finish_load()


class SciBERTModel(ClsEncoder):
    """SciBERT encoder: allenai/scibert_scivocab_uncased (plain BERT, not trained for
    sentence embeddings), mean pooling, L2-normalised vectors.

    First run downloads the model from the HF Hub and saves it to local_dir
    (SCIBERT_LOCAL_DIR, default model/embedding/scibert); later runs load from
    local_dir only.
    """

    MODEL = "allenai/scibert_scivocab_uncased"
    POOLING = "mean"

    def __init__(self, local_dir=None, device=None, batch_size=128):
        self._setup(local_dir, "SCIBERT_LOCAL_DIR", "scibert", device, batch_size)

        if os.path.isfile(os.path.join(self.local_dir, "config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.local_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModel.from_pretrained(self.MODEL)
            os.makedirs(self.local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(self.local_dir)
            self.model.save_pretrained(self.local_dir)
        self._finish_load()


class Qwen3Model:
    """Qwen3-Embedding (same model and local cache as prediction.qwen_embedding, the FACET
    aspect encoder) for the baselines: one instruction on both sides, max MAX_LENGTH tokens,
    batches of batch_size (length-sorted). encode(texts, instruction) -> {text: L2-normalised vector}."""

    PAPER_INSTRUCTION = ("Represent a scientific paper (title and abstract), "
                         "for comparison with other papers")
    ASPECT_INSTRUCTION = ("Represent a statement from a scientific paper, "
                          "for comparison with statements from other papers")
    MAX_LENGTH = 512

    def __init__(self, batch_size=8, **kwargs):
        from prediction.qwen_embedding import AspectSimilarity  # needs sentence_transformers
        self.batch_size = batch_size
        self.model = AspectSimilarity(batch_size=self.batch_size,
                                      max_seq_length=self.MAX_LENGTH, **kwargs).model

    def encode(self, texts, instruction=PAPER_INSTRUCTION):
        texts = sorted(set(texts), key=len)
        if not texts:
            return {}
        vecs = self.model.encode(texts, prompt=f"Instruct: {instruction}\nQuery:",
                                 batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_tensor=True, show_progress_bar=False)
        return dict(zip(texts, vecs.float().cpu()))


def reference_relevance_computation(base_text, ref_dict, embedding_model):
    """Cosine-similarity relevance of each reference text to base_text.

    base_text : the main paper's text.
    ref_dict  : {paper_id: text} for the reference papers to score.
    embedding_model : a loaded EmbeddingModel (load_embedding_model() already
                       called).

    Returns {paper_id: cosine similarity with base_text}.
    """
    ref_embeddings = embedding_model.get_embedding_batch(ref_dict)
    base_embedding = embedding_model.get_embedding_batch({"__base__": base_text})["__base__"]

    base_norm = np.linalg.norm(base_embedding)
    similarities = {}
    for paper_id, vec in ref_embeddings.items():
        denom = base_norm * np.linalg.norm(vec)
        similarities[paper_id] = float(np.dot(base_embedding, vec) / denom) if denom else 0.0
    return similarities


def main():
    sample_text = "We propose a new method for extracting citation contexts from scientific abstracts."

    embedding_model = EmbeddingModel()
    embedding_model.load_embedding_model()

    embeddings = embedding_model.get_embedding_batch({"sample": sample_text})
    vec = embeddings["sample"]
    print(f"Embedding shape: {vec.shape}")
    print(f"First 5 values: {vec[:5]}")


if __name__ == "__main__":
    main()
