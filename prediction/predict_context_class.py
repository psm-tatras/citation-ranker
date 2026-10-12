"""
Inference for the QLoRA-fine-tuned citation-relation classifier.

Predicts FUNCTION/M_ASPECT/POLARITY/EVIDENCE/R_MATCH for one citation
context. The prompt is read from the W&B artifact's metadata["prompt"], so it
always matches the loaded adapter.

Env vars: WANDB_PROJECT, WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME
(default "qwen-context-classifier-adapter"), AWS credentials + AWS_REGION.
"""

import asyncio
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .slm_generate import extract_json
from training.train_data_generator.synthetic_context_classification_dataset_building import (
    collapse_ws,
    clean_section,
    clean_narrative,
    build_context_strings,
)


def _normalize_prediction(obj, output_keys):
    """Return a dict with every output key present as a string."""
    if not isinstance(obj, dict):
        obj = {}
    return {key: (obj[key] if isinstance(obj.get(key), str) else "")
            for key in output_keys}


def _regex_extract_labels(raw, output_keys):
    """Fallback parser for completions that are not valid JSON.

    Each value spans from its key to the next key, so a stray unescaped quote
    inside a value does not truncate it. Returns {} if no key is found.
    """
    result = {}
    for i, key in enumerate(output_keys):
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"?', raw)
        if not m:
            continue
        start = m.end()
        end = len(raw)
        for next_key in output_keys[i + 1:]:
            nm = re.search(rf'"{re.escape(next_key)}"\s*:', raw[start:])
            if nm:
                end = start + nm.start()
                break
        value = raw[start:end].rstrip()
        value = re.sub(r'[",}\s]+$', '', value)  # trailing quote/comma/brace
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        result[key] = value
    return result


def _render_user_message(context, narrative_block_template, user_template, max_side_words):
    """Preprocess a raw citation context as in training and render the user prompt.

    Args:
        context: {m_title, m_narrative, r_title, r_narrative, section, co_cited,
            before, center, after} with raw (uncleaned) values.
    """
    m_title = collapse_ws(context.get("m_title", ""))
    r_title = collapse_ws(context.get("r_title", ""))
    m_narrative = clean_narrative(context.get("m_narrative"))
    r_narrative = clean_narrative(context.get("r_narrative"))
    section = clean_section(context.get("section", ""))
    center = collapse_ws(context.get("center", ""))
    before, after, _truncated = build_context_strings(
        context.get("before"), context.get("after"), max_side_words)
    co_cited = context.get("co_cited", False)
    co_cited = co_cited if isinstance(co_cited, str) else ("yes" if co_cited else "no")

    m_block = narrative_block_template.format(**m_narrative)
    r_block = narrative_block_template.format(**r_narrative)
    return user_template.format(
        m_title=m_title, m_narrative=m_block,
        r_title=r_title, r_narrative=r_block,
        section=section, co_cited=co_cited,
        before=before, center=center, after=after,
    )


class ContextClassifierModel:
    """Citation-relation classifier: Qwen base with the classifier LoRA adapter.

    Args:
        local_dir: Adapter download dir (must not be shared with other adapters).
        wandb_project: W&B project; defaults to WANDB_PROJECT.
        artifact_name: W&B artifact name; defaults to
            WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME.
        checkpoint_kind: None for the final adapter, or "train-best"/"eval-best".
        wandb_version: W&B artifact version ("latest" or e.g. "v61").
        max_side_words: Must match the training dataset's --max_side_words.
        base_model_id, load_in_4bit, dtype, device_map: Base model loading options.
        shared_model, shared_tokenizer: An already-loaded PeftModel/tokenizer to
            attach this adapter to instead of loading a new base.
        adapter_name: PEFT adapter name.
    """

    def __init__(self, local_dir, wandb_project=None, artifact_name=None,
                checkpoint_kind=None, wandb_version="latest",
                max_side_words=40,
                base_model_id="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True, dtype="bfloat16", device_map="auto",
                shared_model=None, shared_tokenizer=None, adapter_name="context_classifier"):
        import os
        from utils.load_model import load_latest_adapter, resolve_latest_adapter_artifact, WANDB_PROJECT

        resolved_project = wandb_project or WANDB_PROJECT
        resolved_artifact_name = artifact_name or os.getenv(
            "WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME", "qwen-context-classifier-adapter")
        self.max_side_words = max_side_words
        self.adapter_name = adapter_name

        artifact = resolve_latest_adapter_artifact(
            wandb_project=resolved_project,
            artifact_name=resolved_artifact_name,
            checkpoint_kind=checkpoint_kind,
            version=wandb_version,
        )
        prompt_cfg = artifact.metadata.get("prompt")
        if not prompt_cfg:
            raise ValueError(
                f"W&B artifact '{artifact.name}:{artifact.version}' has no metadata['prompt'] -- "
                "it was not registered by train_qlora_context_classifier.py, or predates the "
                "prompt-embedding change. Cannot determine the inference prompt without it."
            )
        self.version = artifact.metadata.get("version", "unknown")
        self.system_prompt = prompt_cfg["system"]
        self.user_template = prompt_cfg["template"]
        self.narrative_block = prompt_cfg["narrative_block"]
        self.output_keys = list(prompt_cfg["output_keys"])
        self.functions = list(prompt_cfg.get("functions", []))
        print(f"✓ Loaded prompt from '{artifact.name}:{artifact.version}' metadata "
              f"(version={self.version}, output_keys={self.output_keys})")

        adapter_dir = load_latest_adapter(
            Path(local_dir),
            wandb_project=resolved_project,
            artifact_name=resolved_artifact_name,
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

    def predict(self, context, max_new_tokens=200):
        """Classify one citation context (see _render_user_message for its shape).

        Returns:
            dict: All output keys present; "" for any missing or unparsable value.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": _render_user_message(
                context, self.narrative_block, self.user_template, self.max_side_words)},
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
            print(f"[ContextClassifierModel.predict] JSON parse failed ({e}); "
                  f"generated {len(gen)}/{max_new_tokens} tokens"
                  + (" -- HIT max_new_tokens, response was likely truncated"
                     " (raise max_new_tokens)" if hit_cap else "") + ".")
            print(f"[ContextClassifierModel.predict] raw output:\n{raw}")
            obj = _regex_extract_labels(raw, self.output_keys)
            if obj:
                print(f"[ContextClassifierModel.predict] recovered via regex fallback: {sorted(obj.keys())}")
        return _normalize_prediction(obj, self.output_keys)

    async def apredict(self, items, batch_size=8, max_new_tokens=200):
        """Batched predict() over many citation contexts.

        Args:
            items: List of {"id", "context"} dicts.

        Returns:
            list[dict]: {"id", "prediction"} per item, in input order.
        """
        results = []
        with tqdm(total=len(items), desc="Classifying citation relations", unit="context") as pbar:
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                results.extend(await asyncio.to_thread(self._generate_batch, chunk, max_new_tokens))
                pbar.update(len(chunk))
        return results

    def _generate_batch(self, chunk, max_new_tokens):
        """Run one left-padded generate() call over a chunk of {id, context}."""
        ids = [it["id"] for it in chunk]
        contexts = [it["context"] for it in chunk]

        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": _render_user_message(
                        context, self.narrative_block, self.user_template, self.max_side_words)},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            for context in contexts
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
                prediction = _normalize_prediction(extract_json(raw), self.output_keys)
            except Exception as e:
                hit_cap = int((gen != self.tokenizer.pad_token_id).sum()) >= max_new_tokens
                print(f"[ContextClassifierModel._generate_batch] id={item_id!r} JSON parse failed ({e})"
                      + (" -- may have HIT max_new_tokens, response possibly truncated"
                         " (raise max_new_tokens)" if hit_cap else "") + ".")
                print(f"[ContextClassifierModel._generate_batch] id={item_id!r} raw output:\n{raw}")
                recovered = _regex_extract_labels(raw, self.output_keys)
                if recovered:
                    print(f"[ContextClassifierModel._generate_batch] id={item_id!r} "
                          f"recovered via regex fallback: {sorted(recovered.keys())}")
                prediction = _normalize_prediction(recovered, self.output_keys)
            results.append({"id": item_id, "prediction": prediction})
        return results


if __name__ == "__main__":
    # python -m prediction.predict_context_class
    import json
    import os
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    local_dir = os.environ.get("CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR", str(project_root / "model" / "context_classifier"))
    context_classifier_model = ContextClassifierModel(
        local_dir=local_dir,
        checkpoint_kind="train-best",
        wandb_version="latest",
    )

    sample_context = {
        "m_title": "A Data Source for Reasoning Embodied Agents",
        "m_narrative": {
            "TASK": "machine reasoning with neural models, particularly using embodied agents and database-encoded world states",
            "LIMITATION": "baseline models struggle with answering certain questions about the world-state",
            "CONTRIBUTION": "a new data generator for machine reasoning that integrates with an embodied agent",
            "METHOD": "templated text queries and answers matched with world-states encoded into a database",
            "RESULT": "baseline models can answer some questions about the world-state but struggle with others",
        },
        "r_title": "Adam: A Method for Stochastic Optimization",
        "r_narrative": {
            "TASK": "first-order gradient-based optimization of stochastic objective functions",
            "LIMITATION": "",
            "CONTRIBUTION": "Adam: an algorithm for first-order gradient-based optimization of stochastic objective functions",
            "METHOD": "adaptive estimates of lower-order moments, invariant to diagonal rescaling of the gradients",
            "RESULT": "computationally efficient, little memory, well suited to noisy and/or sparse gradients",
        },
        "section": "Model and Training Details",
        "co_cited": False,
        "before": [],
        "center": "All our models are trained using Adam [Citation] for 5,000 epochs, where each epoch is over a chunk of 10,000 training samples.",
        "after": ["Since we are generating the data, we vary the training samples from 1k to 1M, and use a validation set of 10k samples."],
    }
    with context_classifier_model.model.disable_adapter():
        prediction = context_classifier_model.predict(sample_context)
    print(json.dumps(prediction, indent=2))
