"""
vLLM version of prediction/predict_main.py: narrative extraction (Layer 1) and
citation-context classification (Layer 3) served from ONE bf16 Qwen2.5-7B
base with BOTH LoRA adapters attached via vLLM multi-LoRA.

Differences from CitationPipelinePredictor (HF transformers):
  - No set_adapter(): each request carries its own LoRARequest, so prompts
    for both adapters (and from many main papers) can share a batch.
  - Continuous batching: batch sizes are not needed; vLLM schedules all
    submitted prompts (bounded by max_num_seqs / KV-cache memory).
  - bf16 base (not 4-bit NF4): the standard way to serve QLoRA adapters.
    Outputs can differ slightly from the 4-bit HF path -- validate first.

Prompt building, adapter download (W&B -> S3), JSON parsing and fallbacks
are reused from predict_narrative.py / predict_context_class.py, so the
prompts are identical to the HF path.

Usage:
    predictor = CitationPipelinePredictorVLLM(
        narrative_local_dir="model/narrative",
        context_classifier_local_dir="model/context_classifier",
    )
    m_narrative = predictor.predict_narrative(m_abstract)
    results = predictor.predict_papers([paper_job, ...])   # many M at once
"""

import os
import traceback
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.prompts import NARRATIVE_VERSIONS
from .slm_generate import extract_json
from .predict_narrative import (
    _normalize_narrative,
    _regex_extract_labels as _regex_extract_narrative,
)
from .predict_context_class import (
    _normalize_prediction,
    _regex_extract_labels as _regex_extract_prediction,
    _render_user_message,
)


class CitationPipelinePredictorVLLM:
    """Narrative and context-classifier adapters on one vLLM engine.

    Args:
        narrative_local_dir, context_classifier_local_dir: Adapter download
            dirs (must differ).
        base_model_id: HF id of the base model the adapters were trained on.
        narrative_version: NARRATIVE_VERSIONS key the narrative adapter was trained on.
        narrative_*, context_*: Per-adapter W&B overrides (project, artifact
            name, checkpoint kind, artifact version) -- same as the HF predictor.
        context_max_side_words: Must match the classifier's training --max_side_words.
        dtype: Base model dtype for vLLM.
        max_lora_rank: >= the largest adapter rank (both adapters are r=16).
        max_model_len: Max prompt + generated tokens per request.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_num_seqs: Max sequences decoded concurrently.
        enforce_eager: Disable CUDA graphs (less memory, slower).
    """

    def __init__(self, narrative_local_dir, context_classifier_local_dir,
                 base_model_id="Qwen/Qwen2.5-7B-Instruct",
                 narrative_version="v1",
                 narrative_wandb_project=None, narrative_artifact_name=None,
                 narrative_checkpoint_kind=None, narrative_wandb_version="latest",
                 context_wandb_project=None, context_artifact_name=None,
                 context_checkpoint_kind=None, context_wandb_version="latest",
                 context_max_side_words=40,
                 dtype="bfloat16", max_lora_rank=16, max_model_len=8192,
                 gpu_memory_utilization=0.90, max_num_seqs=128, enforce_eager=False):
        from utils.load_model import load_latest_adapter, resolve_latest_adapter_artifact, WANDB_PROJECT

        # ---- Narrative adapter + prompt (same source as NarrativeModel) ----
        version_cfg = NARRATIVE_VERSIONS[narrative_version]
        self.narrative_version = narrative_version
        self.narrative_system = version_cfg["system"]
        self.narrative_template = version_cfg["template"]
        self.narrative_labels = list(version_cfg["labels"])
        narrative_dir = load_latest_adapter(
            Path(narrative_local_dir),
            wandb_project=narrative_wandb_project or WANDB_PROJECT,
            artifact_name=narrative_artifact_name or os.getenv(
                "WANDB_NARRATIVE_ADAPTER_ARTIFACT_NAME", "qwen-narrative-adapter"),
            checkpoint_kind=narrative_checkpoint_kind,
            version=narrative_wandb_version,
        )

        # ---- Context-classifier adapter + prompt from W&B metadata (same as ContextClassifierModel) ----
        ctx_project = context_wandb_project or WANDB_PROJECT
        ctx_artifact_name = context_artifact_name or os.getenv(
            "WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME", "qwen-context-classifier-adapter")
        artifact = resolve_latest_adapter_artifact(
            wandb_project=ctx_project, artifact_name=ctx_artifact_name,
            checkpoint_kind=context_checkpoint_kind, version=context_wandb_version,
        )
        prompt_cfg = artifact.metadata.get("prompt")
        if not prompt_cfg:
            raise ValueError(
                f"W&B artifact '{artifact.name}:{artifact.version}' has no metadata['prompt'] -- "
                "cannot determine the inference prompt without it.")
        self.context_version = artifact.metadata.get("version", "unknown")
        self.context_system = prompt_cfg["system"]
        self.context_template = prompt_cfg["template"]
        self.context_narrative_block = prompt_cfg["narrative_block"]
        self.context_output_keys = list(prompt_cfg["output_keys"])
        self.context_max_side_words = context_max_side_words
        print(f"✓ Loaded prompt from '{artifact.name}:{artifact.version}' metadata "
              f"(version={self.context_version}, output_keys={self.context_output_keys})")
        context_dir = load_latest_adapter(
            Path(context_classifier_local_dir),
            wandb_project=ctx_project, artifact_name=ctx_artifact_name,
            checkpoint_kind=context_checkpoint_kind, version=context_wandb_version,
        )

        self.narrative_lora = LoRARequest("narrative", 1, str(narrative_dir))
        self.context_lora = LoRARequest("context_classifier", 2, str(context_dir))

        # ---- Tokenizer (chat template) and decoding settings matching the HF path ----
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        # HF generate(do_sample=False) still applies the model's generation_config
        # repetition_penalty (1.05 for Qwen2.5-Instruct) and stops on all of its
        # eos ids; vLLM ignores generation_config when SamplingParams are given
        # explicitly, so carry both over to keep outputs comparable.
        gen_cfg = GenerationConfig.from_pretrained(base_model_id)
        self.repetition_penalty = float(getattr(gen_cfg, "repetition_penalty", None) or 1.0)
        eos = gen_cfg.eos_token_id
        eos = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
        self.stop_token_ids = sorted(set(eos) | {self.tokenizer.eos_token_id})
        self.max_model_len = max_model_len

        self.llm = LLM(
            model=base_model_id,
            dtype=dtype,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------ prompts
    def _narrative_prompt(self, abstract):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.narrative_system},
             {"role": "user", "content": self.narrative_template.format(abstract=abstract)}],
            tokenize=False, add_generation_prompt=True)

    def _context_prompt(self, context):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.context_system},
             {"role": "user", "content": _render_user_message(
                 context, self.context_narrative_block, self.context_template,
                 self.context_max_side_words)}],
            tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------- generation
    def _generate(self, requests, use_tqdm=True):
        """Run one vLLM generate() over mixed-adapter requests.

        Args:
            requests: list of (prompt_text, LoRARequest, max_new_tokens).

        Returns:
            list of (raw_text, truncated) per request, in input order;
            (None, False) for a request whose prompt alone exceeds max_model_len.
        """
        results = [(None, False)] * len(requests)
        prompts, params, loras, positions = [], [], [], []
        for i, (prompt, lora, max_new_tokens) in enumerate(requests):
            n_prompt = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            budget = self.max_model_len - n_prompt
            if budget < 16:
                print(f"[vLLM] request {i} skipped: prompt is {n_prompt} tokens "
                      f"(max_model_len={self.max_model_len})")
                continue
            prompts.append(prompt)
            params.append(SamplingParams(
                temperature=0.0, max_tokens=min(max_new_tokens, budget),
                repetition_penalty=self.repetition_penalty,
                stop_token_ids=self.stop_token_ids))
            loras.append(lora)
            positions.append(i)
        if not prompts:
            return results
        outputs = self.llm.generate(prompts, params, lora_request=loras, use_tqdm=use_tqdm)
        for pos, out in zip(positions, outputs):
            comp = out.outputs[0]
            results[pos] = (comp.text.strip(), comp.finish_reason == "length")
        return results

    def _parse_narrative(self, item_id, raw, truncated):
        if raw is None:
            return _normalize_narrative({}, self.narrative_labels)
        try:
            return _normalize_narrative(extract_json(raw), self.narrative_labels)
        except Exception as e:
            print(f"[vLLM narrative] id={item_id!r} JSON parse failed ({e})"
                  + (" -- HIT max_new_tokens, response truncated" if truncated else "") + ".")
            print(f"[vLLM narrative] id={item_id!r} raw output:\n{raw}")
            recovered = _regex_extract_narrative(raw, self.narrative_labels)
            if recovered:
                print(f"[vLLM narrative] id={item_id!r} recovered via regex fallback: {sorted(recovered.keys())}")
            return _normalize_narrative(recovered, self.narrative_labels)

    def _parse_relation(self, item_id, raw, truncated):
        if raw is None:
            return _normalize_prediction({}, self.context_output_keys)
        try:
            return _normalize_prediction(extract_json(raw), self.context_output_keys)
        except Exception as e:
            print(f"[vLLM relation] id={item_id!r} JSON parse failed ({e})"
                  + (" -- HIT max_new_tokens, response truncated" if truncated else "") + ".")
            print(f"[vLLM relation] id={item_id!r} raw output:\n{raw}")
            recovered = _regex_extract_prediction(raw, self.context_output_keys)
            if recovered:
                print(f"[vLLM relation] id={item_id!r} recovered via regex fallback: {sorted(recovered.keys())}")
            return _normalize_prediction(recovered, self.context_output_keys)

    # ---------------------------------------------------- Layer 1: narratives
    def predict_narratives(self, items, max_new_tokens=2048, use_tqdm=True):
        """Batched narrative extraction.

        Args:
            items: list of {"id", "abstract"}.

        Returns:
            list[dict]: {"id", "narrative"} per item, in input order.
        """
        raws = self._generate(
            [(self._narrative_prompt(it["abstract"]), self.narrative_lora, max_new_tokens) for it in items],
            use_tqdm=use_tqdm)
        return [{"id": it["id"], "narrative": self._parse_narrative(it["id"], raw, trunc)}
                for it, (raw, trunc) in zip(items, raws)]

    def predict_narrative(self, abstract, max_new_tokens=400):
        """Extract the narrative of one abstract -> {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}."""
        return self.predict_narratives([{"id": 0, "abstract": abstract}],
                                       max_new_tokens=max_new_tokens, use_tqdm=False)[0]["narrative"]

    # ------------------------------------------------ Layer 3: relations
    def predict_relations(self, items, max_new_tokens=512, use_tqdm=True):
        """Batched citation-relation classification.

        Args:
            items: list of {"id", "context"} (context shape as in ContextClassifierModel).

        Returns:
            list[dict]: {"id", "prediction"} per item, in input order.
        """
        raws = self._generate(
            [(self._context_prompt(it["context"]), self.context_lora, max_new_tokens) for it in items],
            use_tqdm=use_tqdm)
        return [{"id": it["id"], "prediction": self._parse_relation(it["id"], raw, trunc)}
                for it, (raw, trunc) in zip(items, raws)]

    def predict_citation_context(self, m_title, m_narrative, r_title, r_narrative,
                                 cite_ctx, relation_max_new_tokens=200):
        """Classify each citation context of one reference (same API as the HF predictor).

        Returns:
            list[dict]: One {section, sentence_index, ref_id, relation} per entry.
        """
        items = [{"id": i, "context": {
            "m_title": m_title, "m_narrative": m_narrative,
            "r_title": r_title, "r_narrative": r_narrative,
            "section": entry.get("section", ""), "co_cited": entry.get("co_cited", False),
            "before": entry.get("before"), "center": entry.get("center", ""),
            "after": entry.get("after"),
        }} for i, entry in enumerate(cite_ctx)]
        preds = self.predict_relations(items, max_new_tokens=relation_max_new_tokens, use_tqdm=False)
        return [{"section": entry.get("section", ""),
                 "sentence_index": entry.get("sentence_index"),
                 "ref_id": entry.get("ref_id"),
                 "relation": p["prediction"]} for entry, p in zip(cite_ctx, preds)]

    # ------------------------------------------- many main papers at once
    def predict_papers(self, jobs, narrative_max_new_tokens=2048, relation_max_new_tokens=512):
        """Layer 1 + Layer 3 for several main papers in two vLLM calls.

        Call 1: narratives of every M abstract and every usable reference
        abstract across all jobs. Call 2: every citation context across all
        jobs. Per-paper logic (usable references, co_cited, output shape) is
        the same as CitationPipelinePredictor.predict_references().

        Args:
            jobs: list of {"m_id", "m_title", "m_abstract", "ref_contexts",
                "ref_papers_data"} -- ref_contexts / ref_papers_data as in
                CitationPipelinePredictor.predict_references().

        Returns:
            list[dict]: per job, {"m_aspects": dict, "references": dict}.
        """
        from utils.seku_crd_corpus_utils import get_title_and_abstract

        # ---- usable references per job ----
        valid = []  # per job: {r_id: (r_title, r_abstract)}
        for job in jobs:
            refs = {}
            for r_id, paper in job["ref_papers_data"].items():
                if paper is None or r_id not in job["ref_contexts"]:
                    continue
                try:
                    refs[r_id] = get_title_and_abstract(paper)
                except Exception:
                    print(f"can not process {r_id}, issue {traceback.format_exc()}")
            print(f"[{job['m_id']}] {len(refs)}/{len(job['ref_papers_data'])} references usable")
            valid.append(refs)

        # ---- Call 1: narratives (M abstracts + reference abstracts) ----
        narrative_items = []
        for j, job in enumerate(jobs):
            narrative_items.append({"id": (j, None), "abstract": job["m_abstract"]})
            for r_id, (_, r_abstract) in valid[j].items():
                narrative_items.append({"id": (j, r_id), "abstract": r_abstract})
        narratives = {r["id"]: r["narrative"] for r in self.predict_narratives(
            narrative_items, max_new_tokens=narrative_max_new_tokens)}

        # ---- Call 2: relations for every citation context ----
        relation_items = []
        for j, job in enumerate(jobs):
            key_to_refs = defaultdict(set)
            for r_id in valid[j]:
                for ctx in job["ref_contexts"][r_id]["citation_context"]:
                    key_to_refs[(ctx.get("section"), ctx.get("sentence_index"))].add(r_id)
            m_narrative = narratives[(j, None)]
            for r_id, (r_title, _) in valid[j].items():
                for i, entry in enumerate(job["ref_contexts"][r_id]["citation_context"]):
                    section = entry.get("section", "")
                    co_cited = len(key_to_refs[(section, entry.get("sentence_index"))]) > 1
                    relation_items.append({"id": (j, r_id, i), "context": {
                        "m_title": job["m_title"], "m_narrative": m_narrative,
                        "r_title": r_title, "r_narrative": narratives[(j, r_id)],
                        "section": section, "co_cited": co_cited,
                        "before": entry.get("before"), "center": entry.get("center", ""),
                        "after": entry.get("after"),
                    }})
        relations = {r["id"]: r["prediction"] for r in self.predict_relations(
            relation_items, max_new_tokens=relation_max_new_tokens)}

        # ---- assemble, same shape as predict_references() ----
        results = []
        for j, job in enumerate(jobs):
            references = {}
            for r_id, (r_title, r_abstract) in valid[j].items():
                r_context = job["ref_contexts"][r_id]["citation_context"]
                references[r_id] = [{r_id: {
                    "abstract": r_abstract,
                    "title": r_title,
                    "m_context": r_context,
                    "aspect": narratives[(j, r_id)],
                    "facet_relevance": [relations[(j, r_id, i)] for i in range(len(r_context))],
                }}]
            results.append({"m_aspects": narratives[(j, None)], "references": references})
        return results

    async def predict_references(self, m_title, m_narrative, ref_contexts, ref_papers_data,
                                 narrative_batch_size=None, relation_batch_size=None,
                                 narrative_max_new_tokens=2048, relation_max_new_tokens=512):
        """Drop-in for CitationPipelinePredictor.predict_references() (one M,
        M narrative already computed). Batch sizes are ignored -- vLLM schedules
        continuously."""
        from utils.seku_crd_corpus_utils import get_title_and_abstract

        valid_refs = {}
        for r_id, paper in ref_papers_data.items():
            if paper is None or r_id not in ref_contexts:
                continue
            try:
                valid_refs[r_id] = get_title_and_abstract(paper)
            except Exception:
                print(f"can not process {r_id}, issue {traceback.format_exc()}")
        print(f"{len(valid_refs)}/{len(ref_papers_data)} references usable for batching")

        r_narratives = {r["id"]: r["narrative"] for r in self.predict_narratives(
            [{"id": r_id, "abstract": a} for r_id, (_, a) in valid_refs.items()],
            max_new_tokens=narrative_max_new_tokens)}

        key_to_refs = defaultdict(set)
        for r_id in valid_refs:
            for ctx in ref_contexts[r_id]["citation_context"]:
                key_to_refs[(ctx.get("section"), ctx.get("sentence_index"))].add(r_id)
        relation_items = []
        for r_id, (r_title, _) in valid_refs.items():
            for i, entry in enumerate(ref_contexts[r_id]["citation_context"]):
                section = entry.get("section", "")
                relation_items.append({"id": f"{r_id}|{i}", "context": {
                    "m_title": m_title, "m_narrative": m_narrative,
                    "r_title": r_title, "r_narrative": r_narratives[r_id],
                    "section": section,
                    "co_cited": len(key_to_refs[(section, entry.get("sentence_index"))]) > 1,
                    "before": entry.get("before"), "center": entry.get("center", ""),
                    "after": entry.get("after"),
                }})
        relations = {r["id"]: r["prediction"] for r in self.predict_relations(
            relation_items, max_new_tokens=relation_max_new_tokens)}

        references = {}
        for r_id, (r_title, r_abstract) in valid_refs.items():
            r_context = ref_contexts[r_id]["citation_context"]
            references[r_id] = [{r_id: {
                "abstract": r_abstract,
                "title": r_title,
                "m_context": r_context,
                "aspect": r_narratives[r_id],
                "facet_relevance": [relations[f"{r_id}|{i}"] for i in range(len(r_context))],
            }}]
        return references


if __name__ == "__main__":
    # python -m prediction.predict_main_vllm
    import json
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    predictor = CitationPipelinePredictorVLLM(
        narrative_local_dir=os.environ.get(
            "NARRATIVE_MODEL_LOCAL_DIR", str(project_root / "model" / "narrative")),
        context_classifier_local_dir=os.environ.get(
            "CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR", str(project_root / "model" / "context_classifier")),
        narrative_checkpoint_kind="train-best",
        narrative_wandb_version="latest",
        narrative_version="v2",
        context_checkpoint_kind="train-best",
        context_wandb_version="latest",
    )

    m_narrative = predictor.predict_narrative(
        """We present the Compressive Transformer, an attentive sequence model which\ncompresses past memories for long-range sequence learning. We find the\nCompressive Transformer obtains state-of-the-art language modelling results in\nthe WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc\nrespectively.""")
    r_narrative = predictor.predict_narrative(
        """We introduce Adam, an algorithm for first-order gradient-based optimization\nof stochastic objective functions, based on adaptive estimates of lower-order\nmoments. The method is straightforward to implement, is computationally\nefficient, has little memory requirements.""")

    cite_ctx = [{
        "section": "Model and Training Details",
        "ref_id": "b1",
        "sentence_index": 12,
        "before": [],
        "center": "All our models are trained using Adam [Citation] for 5,000 epochs, where each epoch is over a chunk of 10,000 training samples.",
        "after": ["Since we are generating the data, we vary the training samples from 1k to 1M, and use a validation set of 10k samples."],
    }]
    relations = predictor.predict_citation_context(
        m_title="A Data Source for Reasoning Embodied Agents",
        m_narrative=m_narrative,
        r_title="Adam: A Method for Stochastic Optimization",
        r_narrative=r_narrative,
        cite_ctx=cite_ctx,
    )
    print(json.dumps({"m_narrative": m_narrative, "r_narrative": r_narrative,
                      "relations": relations}, indent=2))
