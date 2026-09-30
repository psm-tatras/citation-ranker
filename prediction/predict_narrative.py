"""
Inference for the QLoRA-fine-tuned narrative-extraction model
(training/train_qlora_narrative.py): given an abstract, predict its
narrative decomposition using the EXACT same prompt the model was trained
on, so inference matches training.

Label scheme is selected with NarrativeModel(..., version=...), resolved
against the same NARRATIVE_VERSIONS registry (utils/prompts.py) that
training/train_data_generator/synthetic_narrative_labelling_dataset_building.py
uses to build the SFT data -- generation and inference can never disagree on
which system/template/labels go together. Defaults to "v1" because that is
the only version an adapter has actually been trained on so far; pass
version="v2" once a v2 adapter has been trained and is the one being loaded
-- loading a v2 adapter while predicting with the v1 prompt (or vice versa)
silently reintroduces the exact train/inference prompt mismatch this
registry exists to prevent.

Mirrors prediction/predict_ner_v2.py's FineTunedNERModel: resolves and
downloads the latest adapter via its W&B model artifact
(utils.load_model.load_latest_adapter), loads the frozen Qwen base with the
adapter on top, and offers both a single-abstract predict() and a real
batched-generation apredict() for many abstracts at once.

The narrative adapter is registered under its OWN W&B artifact name
(train_qlora_narrative.py's WANDB_ADAPTER_ARTIFACT_NAME, sourced from the
WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME env var, default "qwen-narrative-adapter")
-- separate from the NER adapter's "qwen-ner-adapter" -- so NarrativeModel
resolves "latest" against that name by default, NOT utils.load_model's
NER-flavored WANDB_ADAPTER_ARTIFACT_NAME constant (do not import/use that
one here). Required env vars to load successfully:
  WANDB_PROJECT                         -- must match the project the
                                            narrative adapter was registered in
  WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME -- optional; defaults to
                                            "qwen-narrative-adapter" (matches
                                            train_qlora_narrative.py's default)
  AWS creds (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or an IAM role) +
  AWS_REGION -- to download the adapter files from S3; the bucket/key prefix
                themselves come from the artifact's own metadata, not env vars
"""

import asyncio
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .slm_generate import extract_json
from utils.prompts import NARRATIVE_VERSIONS


def _normalize_narrative(obj, labels):
    """Ensure every label in `labels` is present as a string, so callers get
    a predictable shape regardless of what the model actually returned."""
    if not isinstance(obj, dict):
        obj = {}
    return {label: (obj[label] if isinstance(obj.get(label), str) else "")
            for label in labels}


def _regex_extract_labels(raw, labels):
    """Last-resort fallback for when extract_json can't parse the raw
    completion at all. This happens not just from bad backslash escapes
    (extract_json's own sanitizer handles that) but also from the model
    emitting a stray, unescaped quote INSIDE a value -- which closes the
    JSON string early, and no amount of escape-repair can recover from that
    ambiguity through strict brace/quote-balanced parsing.

    For each label, in order, finds '"<LABEL>": "' and takes everything up
    to the START of the next label's '"<NEXT_LABEL>":' as the value --
    anchored on the next KEY rather than a closing quote character, so a
    stray quote in the middle of a value doesn't truncate it (the value
    naturally spans both sides of the stray quote instead).

    Returns {} if none of the labels' keys could be found at all (raw is
    too garbled to be worth trusting)."""
    result = {}
    for i, label in enumerate(labels):
        m = re.search(rf'"{re.escape(label)}"\s*:\s*"?', raw)
        if not m:
            continue
        start = m.end()
        end = len(raw)
        for next_label in labels[i + 1:]:
            nm = re.search(rf'"{re.escape(next_label)}"\s*:', raw[start:])
            if nm:
                end = start + nm.start()
                break
        value = raw[start:end].rstrip()
        value = re.sub(r'[",}\s]+$', '', value)  # trailing quote/comma/brace left from the value's own punctuation
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        result[label] = value
    return result


class NarrativeModel:
    """QLoRA fine-tuned narrative-extraction model: resolves the latest
    trained adapter via its W&B model artifact and downloads it (the same
    utils.load_model.load_latest_adapter helper train_qlora_narrative.py
    uses to resume training) into local_dir, then loads the frozen base from
    HF with the adapter on top.

    The base loads in 4-bit by default so it fits a 12 GB GPU, matching how
    it was trained. The tokenizer is loaded from the adapter dir, so the
    exact chat template it was trained with is used.

    local_dir   : where the adapter is downloaded to / then loaded from.
                  Use a directory distinct from the NER adapter's (e.g. NOT
                  NER_MODEL_LOCAL_DIR) -- both land at <local_dir>/adapter_model,
                  so sharing a dir with the NER model would overwrite one
                  with the other.
    wandb_project, artifact_name : W&B project / model artifact name to
                  resolve "latest" from; default to WANDB_PROJECT (env) and
                  WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME (env, default
                  "qwen-narrative-adapter") -- see module docstring.
    checkpoint_kind : None (default) loads the final/full-run adapter;
                  "train-best" or "eval-best" loads the best-so-far
                  checkpoint of that kind instead -- registered live during
                  training by utils.save_model.BestCheckpointS3WandbCallback.
                  Useful for a run that's still in progress or was
                  interrupted before finishing.
    wandb_version : the W&B artifact's version alias -- "latest" (default)
                  or a specific pinned version like "v61", to reproduce /
                  test an exact checkpoint rather than whatever is newest.
                  NOT the same thing as `version` below (that's the prompt
                  scheme; this is the W&B artifact version tag).
    version     : which NARRATIVE_VERSIONS entry (utils/prompts.py) the
                  loaded adapter was actually trained on -- selects the
                  system/user prompt and label set predict()/apredict() use.
                  Default "v1": the only version an adapter has been trained
                  on so far. MUST match the adapter you're loading; passing
                  "v2" while loading a v1-trained adapter (or vice versa)
                  reintroduces a train/inference prompt mismatch. Check the
                  W&B artifact's own metadata["version"] if unsure -- e.g.
                  qwen-narrative-adapter-train-best:v61's metadata records
                  "version": "v2".
    """

    def __init__(self, local_dir, wandb_project=None, artifact_name=None,
                checkpoint_kind=None, wandb_version="latest", version="v1",
                base_model_id="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True, dtype="bfloat16", device_map="auto"):
        import os
        from utils.load_model import load_latest_adapter, WANDB_PROJECT

        version_cfg = NARRATIVE_VERSIONS[version]
        self.version = version
        self.system_prompt = version_cfg["system"]
        self.user_template = version_cfg["template"]
        self.labels = list(version_cfg["labels"])

        adapter_dir = load_latest_adapter(
            Path(local_dir),
            wandb_project=wandb_project or WANDB_PROJECT,
            artifact_name=artifact_name or os.getenv(
                "WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME", "qwen-narrative-adapter"),
            checkpoint_kind=checkpoint_kind,
            version=wandb_version,
        )

        kwargs = {"device_map": device_map}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)

        base = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()
        # Adapter uploads contain weights/configuration only. Fall back to the
        # base model tokenizer when tokenizer files were not uploaded with them.
        tokenizer_dir = adapter_dir if any(adapter_dir.glob("tokenizer*")) else base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            trust_remote_code=True,
        )

    def predict(self, abstract, max_new_tokens=400):
        """Predict this instance's version-selected labels for one abstract,
        using the exact prompt training data was built with. Returns a dict
        with all labels always present (empty string for anything the model
        didn't produce / that failed to parse).

        max_new_tokens default: v2's 5-field assistant JSON measured up to
        269 tokens on a 20-abstract sample (see synthetic_narrative_labelling
        _dataset_building.py dry runs); 400 keeps >20% headroom above that.
        Re-check this at full-dataset scale before relying on it for v2."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(abstract=abstract)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]      # strip the prompt
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        try:
            obj = extract_json(raw)
        except Exception as e:
            hit_cap = len(gen) >= max_new_tokens
            print(f"[NarrativeModel.predict] JSON parse failed ({e}); "
                  f"generated {len(gen)}/{max_new_tokens} tokens"
                  + (" -- HIT max_new_tokens, response was likely truncated"
                     " (raise max_new_tokens)" if hit_cap else "") + ".")
            print(f"[NarrativeModel.predict] raw output:\n{raw}")
            obj = _regex_extract_labels(raw, self.labels)
            if obj:
                print(f"[NarrativeModel.predict] recovered via regex fallback: {sorted(obj.keys())}")
        return _normalize_narrative(obj, self.labels)

    async def apredict(self, items, batch_size=8, max_new_tokens=400):
        """Async batched version of predict(). Like FineTunedNERModel.apredict
        (predict_ner_v2.py), this does REAL batched generation: items are
        grouped into chunks of `batch_size`, each chunk tokenized together
        and generated in ONE model.generate() call, so the chunk shares the
        GPU's memory-bandwidth-bound weight reads instead of each item
        paying that cost separately. Each chunk's blocking generate() call
        runs via asyncio.to_thread so it doesn't block the event loop;
        chunks are processed one at a time (GPU is one device).

        items : list of {"id": ..., "abstract": ...}.
        Returns a list, one entry per input item (same order), of
        {"id": ..., "narrative": {TASK, LIMITATION, CONTRIBUTION}}.
        """
        results = []
        with tqdm(total=len(items), desc="Extracting narratives", unit="abstract") as pbar:
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                results.extend(await asyncio.to_thread(self._generate_batch, chunk, max_new_tokens))
                pbar.update(len(chunk))
        return results

    def _generate_batch(self, chunk, max_new_tokens):
        """Tokenize `chunk` (list of {id, abstract}) as one left-padded batch
        and run a single generate() call over it, then parse each
        sequence's completion independently."""
        ids = [it["id"] for it in chunk]
        abstracts = [it["abstract"] for it in chunk]

        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.format(abstract=abstract)},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            for abstract in abstracts
        ]

        # Decoder-only batched generation needs LEFT padding: it right-aligns
        # every prompt's real content so all sequences' generated tokens
        # start at the same index (prompt_len below), regardless of each
        # prompt's original length.
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        finally:
            self.tokenizer.padding_side = original_padding_side

        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for item_id, seq in zip(ids, out):
            gen = seq[prompt_len:]
            raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            try:
                narrative = _normalize_narrative(extract_json(raw), self.labels)
            except Exception as e:
                hit_cap = int((gen != self.tokenizer.pad_token_id).sum()) >= max_new_tokens
                print(f"[NarrativeModel._generate_batch] id={item_id!r} JSON parse failed ({e})"
                      + (" -- may have HIT max_new_tokens, response possibly truncated"
                         " (raise max_new_tokens)" if hit_cap else "") + ".")
                print(f"[NarrativeModel._generate_batch] id={item_id!r} raw output:\n{raw}")
                recovered = _regex_extract_labels(raw, self.labels)
                if recovered:
                    print(f"[NarrativeModel._generate_batch] id={item_id!r} "
                          f"recovered via regex fallback: {sorted(recovered.keys())}")
                narrative = _normalize_narrative(recovered, self.labels)
            results.append({"id": item_id, "narrative": narrative})
        return results


if __name__ == "__main__":
    # python -m prediction.predict_narrative
    import json
    import os
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    local_dir = os.environ.get("NARRATIVE_MODEL_LOCAL_DIR", str(project_root / "model" / "narrative"))
    narrative_model = NarrativeModel(
        local_dir=local_dir,
        checkpoint_kind="train-best",
        wandb_version="latest",
        version="v2",          # must match whatever :latest actually is -- check its metadata["version"] if unsure
    )

    sample_abstract = """We present the Compressive Transformer, an attentive sequence model which\ncompresses past memories for long-range sequence learning. We find the\nCompressive Transformer obtains state-of-the-art language modelling results in\nthe WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc\nrespectively. We also find it can model high-frequency speech effectively and\ncan be used as a memory mechanism for RL, demonstrated on an object matching\ntask. To promote the domain of long-range sequence learning, we propose a new\nopen-vocabulary language modelling benchmark derived from books, PG-19."""
    narrative = narrative_model.predict(sample_abstract)
    print(json.dumps(narrative, indent=2))
