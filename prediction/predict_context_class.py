"""
Inference for the QLoRA-fine-tuned citation-relation-classification model
(training/train_qlora_context_classifier.py): given a citation context
(main paper M, cited paper R, the sentence where R is cited), predict
FUNCTION/M_ASPECT/POLARITY/EVIDENCE/R_MATCH using the EXACT same prompt the
model was trained on, so inference matches training.

Unlike prediction/predict_narrative.py (which resolves its prompt against a
local utils.prompts.NARRATIVE_VERSIONS registry by a caller-supplied
`version` string), the prompt here is loaded DIRECTLY FROM the W&B model
artifact's own metadata -- train_qlora_context_classifier.py embeds the full
CITATION_CLASSIFIER_SLM_VERSIONS[version] config (system prompt, user
template, narrative block, label sets) in metadata["prompt"] when it
registers the adapter (both the final artifact and every live
train-best/eval-best checkpoint). So there is no `version` parameter to get
wrong here: whichever adapter gets loaded, its own prompt comes along with
it, and train/inference can never silently disagree the way
predict_narrative.py's module docstring warns about for NARRATIVE_VERSIONS.

Mirrors prediction/predict_ner_v2.py's FineTunedNERModel /
prediction/predict_narrative.py's NarrativeModel: resolves and downloads the
latest adapter via its W&B model artifact (utils.load_model.load_latest_adapter),
loads the frozen Qwen base with the adapter on top, and offers both a
single-context predict() and a real batched-generation apredict() for many
citation contexts at once.

The context-classifier adapter is registered under its OWN W&B artifact name
(train_qlora_context_classifier.py's WANDB_ADAPTER_ARTIFACT_NAME, sourced
from the WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME env var, default
"qwen-context-classifier-adapter") -- separate from the NER and narrative
adapters -- so ContextClassifierModel resolves "latest" against that name by
default, NOT utils.load_model's NER-flavored WANDB_ADAPTER_ARTIFACT_NAME
constant (do not import/use that one here). Required env vars to load
successfully:
  WANDB_PROJECT                                  -- must match the project
                                                     the adapter was
                                                     registered in
  WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME -- optional; defaults to
                                                     "qwen-context-classifier-adapter"
                                                     (matches
                                                     train_qlora_context_classifier.py's
                                                     default)
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
from training.train_data_generator.synthetic_context_classification_dataset_building import (
    collapse_ws,
    clean_section,
    clean_narrative,
    build_context_strings,
)


def _normalize_prediction(obj, output_keys):
    """Ensure every key in `output_keys` is present as a string, so callers
    get a predictable shape regardless of what the model actually returned."""
    if not isinstance(obj, dict):
        obj = {}
    return {key: (obj[key] if isinstance(obj.get(key), str) else "")
            for key in output_keys}


def _regex_extract_labels(raw, output_keys):
    """Last-resort fallback for when extract_json can't parse the raw
    completion at all -- mirrors predict_narrative.py's _regex_extract_labels
    (generic over any ordered list of JSON string keys, so duplicated here
    rather than imported to avoid coupling this module to predict_narrative's
    internals). See that module for why a stray unescaped quote inside a
    value defeats strict brace/quote-balanced JSON parsing and this is
    needed: anchors each key's value on the START of the next key rather
    than a closing quote, so a stray quote in the middle of a value doesn't
    truncate it.

    Returns {} if none of output_keys could be found at all (raw is too
    garbled to be worth trusting)."""
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
        value = re.sub(r'[",}\s]+$', '', value)  # trailing quote/comma/brace left from the value's own punctuation
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        result[key] = value
    return result


def _render_user_message(context, narrative_block_template, user_template, max_side_words):
    """Preprocess a raw citation-context dict the SAME way
    synthetic_context_classification_dataset_building.py's build_records()
    does (collapse_ws/clean_section/clean_narrative/build_context_strings),
    then render it against the adapter's own prompt templates.

    context: {m_title, m_narrative, r_title, r_narrative, section, co_cited,
              before, center, after} -- m_narrative/r_narrative are RAW
              narrative dicts (missing/empty fields allowed, as in
              find_citation_context() output); before/after are lists of RAW
              sentences; center is the RAW citing sentence; co_cited is a
              bool or "yes"/"no" string; section is the RAW section heading
              (may contain {{cite:...}}-style markers).
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
    """QLoRA fine-tuned citation-relation-classification model: resolves the
    latest trained adapter via its W&B model artifact and downloads it (the
    same utils.load_model.load_latest_adapter helper
    train_qlora_context_classifier.py uses to resume training) into
    local_dir, then loads the frozen base from HF with the adapter on top.

    The prompt (system/template/narrative_block/output_keys) is read from
    THAT SAME ARTIFACT's metadata["prompt"] -- not from a locally-chosen
    utils.prompts.CITATION_CLASSIFIER_SLM_VERSIONS version -- so whichever
    adapter gets loaded, its exact training prompt comes with it.

    The base loads in 4-bit by default so it fits a 12 GB GPU, matching how
    it was trained. The tokenizer is loaded from the adapter dir, so the
    exact chat template it was trained with is used.

    local_dir   : where the adapter is downloaded to / then loaded from.
                  Use a directory distinct from the NER/narrative adapters'
                  (e.g. NOT NER_MODEL_LOCAL_DIR / NARRATIVE_MODEL_LOCAL_DIR)
                  -- both land at <local_dir>/adapter_model, so sharing a dir
                  with another model would overwrite one with the other.
    wandb_project, artifact_name : W&B project / model artifact name to
                  resolve "latest" from; default to WANDB_PROJECT (env) and
                  WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME (env,
                  default "qwen-context-classifier-adapter") -- see module
                  docstring.
    checkpoint_kind : None (default) loads the final/full-run adapter;
                  "train-best" or "eval-best" loads the best-so-far
                  checkpoint of that kind instead -- registered live during
                  training by utils.save_model.BestCheckpointS3WandbCallback.
                  Useful for a run that's still in progress or was
                  interrupted before finishing.
    wandb_version : the W&B artifact's version alias -- "latest" (default)
                  or a specific pinned version like "v61", to reproduce /
                  test an exact checkpoint rather than whatever is newest.
    max_side_words : must match whatever --max_side_words the adapter's
                  training dataset was actually built with (default 40,
                  matching synthetic_context_classification_dataset_building.py's
                  CLI default) -- a mismatch here doesn't break JSON parsing,
                  but silently shows the model more or less citation context
                  per side than it saw in training.
    shared_model, shared_tokenizer, adapter_name : for sharing ONE loaded
                  Qwen base across multiple fine-tuned heads (see
                  prediction/predict_main.py) instead of each wrapper class
                  loading its own full copy of the 7B base model. Leave both
                  shared_model/shared_tokenizer as None (default) to load a
                  standalone base, exactly as before. To share: pass the
                  OTHER wrapper's already-loaded `.model` (a PeftModel) as
                  shared_model (and, to skip a redundant tokenizer load, its
                  `.tokenizer` as shared_tokenizer) -- this adapter is then
                  attached to that SAME PeftModel under `adapter_name`
                  (PEFT's multi-adapter support) rather than a second base
                  being loaded. adapter_name defaults to "context_classifier";
                  predict()/apredict() call
                  self.model.set_adapter(adapter_name) before every
                  generate() call, so the right adapter is active even if a
                  sibling wrapper sharing the same model switched it in
                  between.
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

        # Load the prompt from the artifact's OWN metadata (embedded by
        # train_qlora_context_classifier.py) rather than resolving a version
        # against utils.prompts locally -- see class docstring.
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
            # Attach this adapter to an ALREADY-LOADED base/PeftModel instead
            # of loading a second copy of the 7B base -- see shared_model's
            # docstring above.
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
            # Adapter uploads contain weights/configuration only. Fall back to the
            # base model tokenizer when tokenizer files were not uploaded with them.
            tokenizer_dir = adapter_dir if any(adapter_dir.glob("tokenizer*")) else base_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
            )

    def predict(self, context, max_new_tokens=200):
        """Predict FUNCTION/M_ASPECT/POLARITY/EVIDENCE/R_MATCH for one
        citation context, using the exact prompt the loaded adapter was
        trained with. Returns a dict with all output_keys always present
        (empty string for anything the model didn't produce / that failed
        to parse).

        context: see _render_user_message's docstring for the expected shape.

        max_new_tokens default: 200 is a conservative estimate for this
        5-field JSON (EVIDENCE capped at 12 words); unlike
        NarrativeModel.predict's 400 (measured on a real sample), this has
        not yet been benchmarked against actual generations -- re-check once
        this adapter has been run on a real batch and lower/raise as needed.
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
        """Async batched version of predict(). Like FineTunedNERModel.apredict
        (predict_ner_v2.py) / NarrativeModel.apredict (predict_narrative.py),
        this does REAL batched generation: items are grouped into chunks of
        `batch_size`, each chunk tokenized together and generated in ONE
        model.generate() call, so the chunk shares the GPU's
        memory-bandwidth-bound weight reads instead of each item paying that
        cost separately. Each chunk's blocking generate() call runs via
        asyncio.to_thread so it doesn't block the event loop; chunks are
        processed one at a time (GPU is one device).

        items : list of {"id": ..., "context": <citation-context dict, see
                _render_user_message>}.
        Returns a list, one entry per input item (same order), of
        {"id": ..., "prediction": {FUNCTION, M_ASPECT, POLARITY, EVIDENCE, R_MATCH}}.
        """
        results = []
        with tqdm(total=len(items), desc="Classifying citation relations", unit="context") as pbar:
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                results.extend(await asyncio.to_thread(self._generate_batch, chunk, max_new_tokens))
                pbar.update(len(chunk))
        return results

    def _generate_batch(self, chunk, max_new_tokens):
        """Tokenize `chunk` (list of {id, context}) as one left-padded batch
        and run a single generate() call over it, then parse each
        sequence's completion independently."""
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
    prediction = context_classifier_model.predict(sample_context)
    print(json.dumps(prediction, indent=2))
