"""
End-to-end citation-relation prediction: narrative extraction followed by
citation-context classification, sharing one Qwen2.5-7B base model between
both LoRA adapters.

Usage:
    predictor = CitationPipelinePredictor(
        narrative_local_dir="model/narrative",
        context_classifier_local_dir="model/context_classifier",
    )
    m_narrative = predictor.predict_narrative(m_abstract)
    r_narrative = predictor.predict_narrative(r_abstract)
    relations = predictor.predict_citation_context(
        m_title, m_narrative, r_title, r_narrative, cite_ctx)
"""

import gc
import traceback
from collections import defaultdict

import torch

from .predict_narrative import NarrativeModel
from .predict_context_class import ContextClassifierModel


class CitationPipelinePredictor:
    """Narrative and context-classifier adapters on a single shared base model.

    Args:
        narrative_local_dir: Download dir for the narrative adapter.
        context_classifier_local_dir: Download dir for the context-classifier
            adapter (must differ from narrative_local_dir).
        base_model_id, load_in_4bit, dtype, device_map: Base model loading options.
        narrative_version: NARRATIVE_VERSIONS key the narrative adapter was trained on.
        narrative_*, context_*: Per-adapter W&B overrides (project, artifact
            name, checkpoint kind, artifact version).
        context_max_side_words: Must match the classifier's training --max_side_words.
    """

    def __init__(self, narrative_local_dir, context_classifier_local_dir,
                 base_model_id="Qwen/Qwen2.5-7B-Instruct",
                 load_in_4bit=True, dtype="bfloat16", device_map="auto",
                 narrative_version="v1",
                 narrative_wandb_project=None, narrative_artifact_name=None,
                 narrative_checkpoint_kind=None, narrative_wandb_version="latest",
                 context_wandb_project=None, context_artifact_name=None,
                 context_checkpoint_kind=None, context_wandb_version="latest",
                 context_max_side_words=40):
        self.narrative_model = NarrativeModel(
            local_dir=narrative_local_dir,
            wandb_project=narrative_wandb_project,
            artifact_name=narrative_artifact_name,
            checkpoint_kind=narrative_checkpoint_kind,
            wandb_version=narrative_wandb_version,
            version=narrative_version,
            base_model_id=base_model_id,
            load_in_4bit=load_in_4bit, dtype=dtype, device_map=device_map,
            adapter_name="narrative",
        )
        self.context_model = ContextClassifierModel(
            local_dir=context_classifier_local_dir,
            wandb_project=context_wandb_project,
            artifact_name=context_artifact_name,
            checkpoint_kind=context_checkpoint_kind,
            wandb_version=context_wandb_version,
            max_side_words=context_max_side_words,
            base_model_id=base_model_id,
            load_in_4bit=load_in_4bit, dtype=dtype, device_map=device_map,
            shared_model=self.narrative_model.model,
            shared_tokenizer=self.narrative_model.tokenizer,
            adapter_name="context_classifier",
        )
    # This is Layer 1 - narrative extraction from paper either M or R
    def predict_narrative(self, abstract, max_new_tokens=400):
        """Extract the narrative of one abstract.

        Returns:
            dict: {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
        """
        return self.narrative_model.predict(abstract, max_new_tokens=max_new_tokens)
    # This is Layer 2 - identify citation context in M
    def predict_citation_context(self, m_title, m_narrative, r_title, r_narrative,
                                  cite_ctx, relation_max_new_tokens=200):
        """Classify each citation context of one reference.

        Args:
            m_title, m_narrative: Main paper title and narrative.
            r_title, r_narrative: Cited paper title and narrative.
            cite_ctx: List of {section, ref_id, sentence_index, before, center,
                after} dicts for one reference; co_cited defaults to False.

        Returns:
            list[dict]: One {section, sentence_index, ref_id, relation} per entry.
        """
        results = []
        for entry in cite_ctx:
            context = {
                "m_title": m_title, "m_narrative": m_narrative,
                "r_title": r_title, "r_narrative": r_narrative,
                "section": entry.get("section", ""),
                "co_cited": entry.get("co_cited", False),
                "before": entry.get("before"), "center": entry.get("center", ""),
                "after": entry.get("after"),
            }
            relation = self.context_model.predict(context, max_new_tokens=relation_max_new_tokens)
            results.append({
                "section": entry.get("section", ""),
                "sentence_index": entry.get("sentence_index"),
                "ref_id": entry.get("ref_id"),
                "relation": relation,
            })
        return results
    # This is Layer 3 - for each R in M 1) generate narrative of R, 2) classify R's relation with M
    async def predict_references(self, m_title, m_narrative, ref_contexts, ref_papers_data,
                                   narrative_batch_size=8, relation_batch_size=8,
                                   narrative_max_new_tokens=2048,
                                   relation_max_new_tokens=512):
        """Batched narrative + relation prediction for every reference of a main paper M.

        Args:
            m_title, m_narrative: Main paper title and narrative.
            ref_contexts: {r_id: {"citation_context": [...], "abstract": ...}},
                e.g. find_citation_context(...)[1][m_id]["references"].
            ref_papers_data: {r_id: paper_data_or_None}, e.g. from
                utils.citation_context.get_papers_from_s3(in_corpus_refs_ids).
            narrative_batch_size, relation_batch_size: batch sizes for the
                narrative (Layer 1) and relation (Layer 3) batched calls --
                separate since Layer 3 has one item per CITING SENTENCE
                (often several per reference), so it can need a smaller
                batch than Layer 1 (one item per reference) to fit the same
                GPU budget.
            narrative_max_new_tokens, relation_max_new_tokens: passed
                through to the respective batched calls.

        Returns:
            dict: {r_id: [{r_id: {abstract, title, m_context, aspect,
                   facet_relevance}}]}, one entry per usable reference.
        """
        from utils.seku_crd_corpus_utils import get_title_and_abstract

        valid_refs = {}  # r_id -> (r_title, r_abstract)
        for r_id, ref_paper_data in ref_papers_data.items():
            if ref_paper_data is None or r_id not in ref_contexts:
                continue
            try:
                r_title, r_abstract = get_title_and_abstract(ref_paper_data)
                valid_refs[r_id] = (r_title, r_abstract)
            except Exception:
                print(f"can not process {r_id}, issue {traceback.format_exc()}")
        print(f"{len(valid_refs)}/{len(ref_papers_data)} references usable for batching")

        narrative_items = [{"id": r_id, "abstract": r_abstract} for r_id, (_, r_abstract) in valid_refs.items()]
        narrative_results = await self.narrative_model.apredict(
            narrative_items, batch_size=narrative_batch_size, max_new_tokens=narrative_max_new_tokens)
        r_narratives = {r["id"]: r["narrative"] for r in narrative_results}

        # Narrative phase is done and its output already collapsed into
        # r_narratives above -- narrative_results/narrative_items and the KV
        # cache/activations PyTorch's allocator is still holding "reserved"
        # for them are dead weight before the relation phase even starts
        # allocating its own batch. Release it back now rather than letting
        # both phases' peaks stack on a 12GB GPU.
        del narrative_results, narrative_items
        gc.collect()
        torch.cuda.empty_cache()

        # co_cited: cross-reference (section, sentence_index) across ALL
        # references of M, since every reference's contexts are available at
        # once here (unlike predict_citation_context(), which only sees one
        # reference and defaults co_cited to False).
        key_to_refs = defaultdict(set)
        for r_id in valid_refs:
            for ctx in ref_contexts[r_id]["citation_context"]:
                key_to_refs[(ctx.get("section"), ctx.get("sentence_index"))].add(r_id)

        relation_items = []
        for r_id, (r_title, r_abstract) in valid_refs.items():
            r_narrative = r_narratives[r_id]
            for i, entry in enumerate(ref_contexts[r_id]["citation_context"]):
                section = entry.get("section", "")
                sentence_index = entry.get("sentence_index")
                co_cited = len(key_to_refs[(section, sentence_index)]) > 1
                context = {
                    "m_title": m_title, "m_narrative": m_narrative,
                    "r_title": r_title, "r_narrative": r_narrative,
                    "section": section, "co_cited": co_cited,
                    "before": entry.get("before"), "center": entry.get("center", ""),
                    "after": entry.get("after"),
                }
                relation_items.append({"id": f"{r_id}|{i}", "context": context})

        relation_results = await self.context_model.apredict(
            relation_items, batch_size=relation_batch_size, max_new_tokens=relation_max_new_tokens)
        relations_by_item_id = {r["id"]: r["prediction"] for r in relation_results}

        references = {}
        for r_id, (r_title, r_abstract) in valid_refs.items():
            r_context = ref_contexts[r_id]["citation_context"]
            r_class = [relations_by_item_id[f"{r_id}|{i}"] for i in range(len(r_context))]
            references[r_id] = [{r_id: {
                "abstract": r_abstract,
                "title": r_title,
                "m_context": r_context,
                "aspect": r_narratives[r_id],
                "facet_relevance": r_class,
            }}]
        return references


if __name__ == "__main__":
    # python -m prediction.predict_main
    import json
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    narrative_local_dir = os.environ.get(
        "NARRATIVE_MODEL_LOCAL_DIR", str(project_root / "model" / "narrative"))
    context_classifier_local_dir = os.environ.get(
        "CONTEXT_CLASSIFIER_MODEL_LOCAL_DIR", str(project_root / "model" / "context_classifier"))

    predictor = CitationPipelinePredictor(
        narrative_local_dir=narrative_local_dir,
        context_classifier_local_dir=context_classifier_local_dir,
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
    print(json.dumps(relations, indent=2))
