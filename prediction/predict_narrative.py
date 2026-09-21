"""
Inference for the QLoRA-fine-tuned narrative-extraction model
(training/train_qlora_narrative.py): given an abstract, predict the
3-label narrative decomposition -- TASK, LIMITATION, CONTRIBUTION -- using
the EXACT same prompt (NARRATIVE_SYSTEM_v1 / NARRATIVE_USER_TEMPLATE_v1) the
model was trained on, so inference matches training.

Mirrors prediction/predict_ner_v2.py's FineTunedNERModel: resolves and
downloads the latest adapter via its W&B model artifact
(utils.load_model.load_latest_adapter), loads the frozen Qwen base with the
adapter on top, and offers both a single-abstract predict() and a real
batched-generation apredict() for many abstracts at once.

NOTE: training/train_qlora_narrative.py currently registers its adapter
under the SAME W&B artifact name as the NER adapter
(WANDB_ADAPTER_ARTIFACT_NAME, default "qwen-ner-adapter" -- see
utils/load_model.py), since it imports that constant rather than defining
its own. NarrativeModel follows that same resolution by default so it loads
whatever train_qlora_narrative.py actually registers -- pass an explicit
`artifact_name` here if/when the training script is given its own artifact
name.
"""

import asyncio
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .slm_generate import extract_json
from utils.prompts import NARRATIVE_SYSTEM_v1, NARRATIVE_USER_TEMPLATE_v1, NARRATIVE_LABELS_v1


def _normalize_narrative(obj):
    """Ensure every NARRATIVE_LABELS_v1 key is present as a string, so callers
    get a predictable shape regardless of what the model actually returned."""
    if not isinstance(obj, dict):
        obj = {}
    return {label: (obj[label] if isinstance(obj.get(label), str) else "")
            for label in NARRATIVE_LABELS_v1}


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
    wandb_project, artifact_name : W&B project / model artifact name to
                  resolve "latest" from; default to utils.load_model's
                  WANDB_PROJECT / WANDB_ADAPTER_ARTIFACT_NAME (see module
                  docstring for the current naming caveat).
    """

    def __init__(self, local_dir, wandb_project=None, artifact_name=None,
                base_model_id="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True, dtype="bfloat16", device_map="auto"):
        from utils.load_model import load_latest_adapter, WANDB_PROJECT, WANDB_ADAPTER_ARTIFACT_NAME

        adapter_dir = load_latest_adapter(
            Path(local_dir),
            wandb_project=wandb_project or WANDB_PROJECT,
            artifact_name=artifact_name or WANDB_ADAPTER_ARTIFACT_NAME,
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

    def predict(self, abstract, max_new_tokens=256):
        """Predict {TASK, LIMITATION, CONTRIBUTION} for one abstract, using
        the exact prompt training data was built with. Returns a dict with
        all three keys always present (empty string for anything the model
        didn't produce / that failed to parse)."""
        messages = [
            {"role": "system", "content": NARRATIVE_SYSTEM_v1},
            {"role": "user", "content": NARRATIVE_USER_TEMPLATE_v1.format(abstract=abstract)},
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
        except Exception:
            obj = {}
        return _normalize_narrative(obj)

    async def apredict(self, items, batch_size=8, max_new_tokens=256):
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
                    {"role": "system", "content": NARRATIVE_SYSTEM_v1},
                    {"role": "user", "content": NARRATIVE_USER_TEMPLATE_v1.format(abstract=abstract)},
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
            try:
                gen = seq[prompt_len:]
                raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                narrative = _normalize_narrative(extract_json(raw))
            except Exception:
                narrative = _normalize_narrative({})
            results.append({"id": item_id, "narrative": narrative})
        return results


if __name__ == "__main__":
    # python -m prediction.predict_narrative
    import json
    import os
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    local_dir = os.environ.get("NER_MODEL_LOCAL_DIR", str(project_root / "model"))
    narrative_model = NarrativeModel(local_dir=local_dir)

    sample_abstract = (
        "We present a new method for extracting citation contexts from scientific "
        "abstracts using a fine-tuned language model. Prior approaches relied on "
        "hand-crafted rules that failed to generalize across domains. Our approach "
        "achieves state-of-the-art accuracy on a held-out benchmark of 500 papers."
    )
    narrative = narrative_model.predict(sample_abstract)
    print(json.dumps(narrative, indent=2))
