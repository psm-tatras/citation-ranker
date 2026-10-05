"""
Inference for the QLoRA-fine-tuned narrative-extraction model.

Uses the same NARRATIVE_VERSIONS prompt (utils/prompts.py) the adapter was
trained on; `version` must match the loaded adapter.

Env vars: WANDB_PROJECT, WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME (default
"qwen-narrative-adapter"), AWS credentials + AWS_REGION.
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
    """Return a dict with every label present as a string."""
    if not isinstance(obj, dict):
        obj = {}
    return {label: (obj[label] if isinstance(obj.get(label), str) else "")
            for label in labels}


def _regex_extract_labels(raw, labels):
    """Fallback parser for completions that are not valid JSON.

    Each value spans from its key to the next label's key, so a stray
    unescaped quote inside a value does not truncate it. Returns {} if no
    label is found.
    """
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
        value = re.sub(r'[",}\s]+$', '', value)  # trailing quote/comma/brace
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        result[label] = value
    return result


class NarrativeModel:
    """Narrative-extraction model: Qwen base with the narrative LoRA adapter.

    Args:
        local_dir: Adapter download dir (must not be shared with other adapters).
        wandb_project: W&B project; defaults to WANDB_PROJECT.
        artifact_name: W&B artifact name; defaults to
            WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME.
        checkpoint_kind: None for the final adapter, or "train-best"/"eval-best".
        wandb_version: W&B artifact version ("latest" or e.g. "v61").
        version: NARRATIVE_VERSIONS key the adapter was trained on.
        base_model_id, load_in_4bit, dtype, device_map: Base model loading options.
        shared_model, shared_tokenizer: An already-loaded PeftModel/tokenizer to
            attach this adapter to instead of loading a new base.
        adapter_name: PEFT adapter name.
    """

    def __init__(self, local_dir, wandb_project=None, artifact_name=None,
                checkpoint_kind=None, wandb_version="latest", version="v1",
                base_model_id="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True, dtype="bfloat16", device_map="auto",
                shared_model=None, shared_tokenizer=None, adapter_name="narrative"):
        import os
        from utils.load_model import load_latest_adapter, WANDB_PROJECT

        version_cfg = NARRATIVE_VERSIONS[version]
        self.version = version
        self.system_prompt = version_cfg["system"]
        self.user_template = version_cfg["template"]
        self.labels = list(version_cfg["labels"])
        self.adapter_name = adapter_name

        adapter_dir = load_latest_adapter(
            Path(local_dir),
            wandb_project=wandb_project or WANDB_PROJECT,
            artifact_name=artifact_name or os.getenv(
                "WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME", "qwen-narrative-adapter"),
            checkpoint_kind=checkpoint_kind,
            version=wandb_version,
        )

        if shared_model is not None:
            shared_model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
            self.model = shared_model
        else:
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
            self.model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=adapter_name)
        self.model.eval()

        if shared_tokenizer is not None:
            self.tokenizer = shared_tokenizer
        else:
            # Fall back to the base model tokenizer if the adapter has none.
            tokenizer_dir = adapter_dir if any(adapter_dir.glob("tokenizer*")) else base_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
            )

    def predict(self, abstract, max_new_tokens=400):
        """Extract the narrative of one abstract.

        Returns:
            dict: All labels present; "" for any missing or unparsable value.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(abstract=abstract)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.model.set_adapter(self.adapter_name)
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
        """Batched predict() over many abstracts.

        Args:
            items: List of {"id", "abstract"} dicts.

        Returns:
            list[dict]: {"id", "narrative"} per item, in input order.
        """
        results = []
        with tqdm(total=len(items), desc="Extracting narratives", unit="abstract") as pbar:
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                results.extend(await asyncio.to_thread(self._generate_batch, chunk, max_new_tokens))
                pbar.update(len(chunk))
        return results

    def _generate_batch(self, chunk, max_new_tokens):
        """Run one left-padded generate() call over a chunk of {id, abstract}."""
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

        # Decoder-only batched generation needs left padding.
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True).to(self.model.device)
            self.model.set_adapter(self.adapter_name)
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
        version="v2",          # must match the adapter's metadata["version"]
    )

    sample_abstract = """We present the Compressive Transformer, an attentive sequence model which\ncompresses past memories for long-range sequence learning. We find the\nCompressive Transformer obtains state-of-the-art language modelling results in\nthe WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc\nrespectively. We also find it can model high-frequency speech effectively and\ncan be used as a memory mechanism for RL, demonstrated on an object matching\ntask. To promote the domain of long-range sequence learning, we propose a new\nopen-vocabulary language modelling benchmark derived from books, PG-19."""
    narrative = narrative_model.predict(sample_abstract)
    print(json.dumps(narrative, indent=2))
