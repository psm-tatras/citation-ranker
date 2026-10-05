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

    def predict_narrative(self, abstract, max_new_tokens=400):
        """Extract the narrative of one abstract.

        Returns:
            dict: {TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT}
        """
        return self.narrative_model.predict(abstract, max_new_tokens=max_new_tokens)

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
