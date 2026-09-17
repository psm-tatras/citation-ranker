"""
predict_ner_v3.py
=================

Unified inference pipeline for TWO TASKS using shared Qwen 7B base model:

  Task A (NER):           Fine-tuned with LoRA adapter
  Task B (Type Classification): Zero-shot (no adapter)

Key difference from v2:
  - Load base model ONCE in memory
  - Reuse for both tasks
  - Reduce memory footprint + latency

Classes:
  UnifiedQwenPipeline     -> Loads base model once, manages both tasks
  - predict_ner()         -> NER extraction (with LoRA adapter)
  - predict_contribution_type() -> Type classification (zero-shot)

Usage:
  pipeline = UnifiedQwenPipeline(
      ner_adapter_dir="./qwen-ner-adapter",
      base_model_id="Qwen/Qwen2.5-7B-Instruct"
  )
  
  # Task A: NER
  entities = pipeline.predict_ner(abstract)
  
  # Task B: Type classification (from CONTRIBUTION entity)
  contrib_text = next(e["text"] for e in entities if e["label"] == "CONTRIBUTION")
  contrib_type = pipeline.predict_contribution_type(contrib_text)
"""

import asyncio
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Import your existing helpers (adjust imports as needed)
from .slm_generate import extract_json, resolve_offsets
from utils.prompts import (
    NER_SYSTEM,
    NER_USER_TMPL,
    TYPE_SYSTEM_PROMPT,
    TYPE_USER_PROMPT_TEMPLATE,
    CONTRIBUTION_TYPES,
)


# ============================================================================
# Type definitions
# ============================================================================

@dataclass
class Entity:
    """NER entity: text span + label + character offsets."""
    text: str
    label: str
    start: int
    end: int

    def to_dict(self):
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class TypePrediction:
    """Type classification result."""
    contribution_text: str
    predicted_type: str
    confidence: Optional[float] = None
    raw_response: str = ""

    def to_dict(self):
        return {
            "contribution_text": self.contribution_text[:100] + "..." if len(self.contribution_text) > 100 else self.contribution_text,
            "predicted_type": self.predicted_type,
            "confidence": self.confidence,
        }


# NOTE: Prompts imported from prompts.py (shared with training)
# TYPE_SYSTEM_PROMPT, TYPE_USER_PROMPT_TEMPLATE, CONTRIBUTION_TYPES
# NER prompts are used via NER_SYSTEM_PROMPT from pipeline startup


# ============================================================================
# Unified Qwen Pipeline
# ============================================================================

class UnifiedQwenPipeline:
    """
    Loads Qwen 7B base model ONCE, reuses for both NER and type classification.

    Memory efficiency:
      - Base model loaded once (~14GB in 4-bit)
      - NER adapter loaded on top (~200MB)
      - Type classification runs zero-shot (no adapter loading/unloading)

    Parameters:
    -----------
    ner_adapter_dir : str
        Path to NER LoRA adapter (from train_ner.py / train_qlora.py)

    base_model_id : str
        Base Qwen model ID (default: Qwen/Qwen2.5-7B-Instruct)

    load_in_4bit : bool
        Load in 4-bit quantization (default: True for 12GB GPU)

    dtype : str
        Compute dtype: "bfloat16" or "float16" (default: bfloat16)

    device_map : str
        Device mapping (default: "auto")
    """

    def __init__(
        self,
        ner_adapter_dir: str,
        base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.base_model_id = base_model_id
        self.ner_adapter_dir = Path(ner_adapter_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[UnifiedQwenPipeline] Loading base model: {base_model_id}")
        print(f"[UnifiedQwenPipeline] Device: {self.device} | 4-bit: {load_in_4bit}")

        # === Load base model ONCE ===
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

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, **kwargs
        )
        self.base_model.eval()

        # === Load tokenizer (from adapter if available, else from base) ===
        tokenizer_dir = (
            ner_adapter_dir
            if any(self.ner_adapter_dir.glob("tokenizer*"))
            else base_model_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # === Load NER adapter on top of base ===
        print(f"[UnifiedQwenPipeline] Loading NER adapter: {ner_adapter_dir}")
        self.ner_model = PeftModel.from_pretrained(self.base_model, str(ner_adapter_dir))
        self.ner_model.eval()

        # === For type classification, we'll use the base model directly (zero-shot) ===
        # No adapter loading needed

        print("[UnifiedQwenPipeline] ✓ Initialization complete")

    # ========================================================================
    # Task A: NER (with LoRA adapter)
    # ========================================================================

    def predict_ner(
        self,
        abstract: str,
        max_new_tokens: int = 512,
    ) -> List[Entity]:
        """
        Extract named entities from abstract using fine-tuned NER model.

        Parameters:
        -----------
        abstract : str
            Paper abstract text

        max_new_tokens : int
            Max output tokens (default: 512)

        Returns:
        --------
        List[Entity]
            Extracted entities with text, label, start, end offsets
        """
        # Build prompt (NER_SYSTEM and NER_USER_TMPL imported from prompts.py)
        messages = [
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": NER_USER_TMPL.format(abstract=abstract)},
        ]

        # Generate
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.ner_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse JSON output
        entities_dict = self._parse_ner_response(response, abstract)

        # Convert to Entity objects with offsets
        entities = []
        for label, texts in entities_dict.items():
            if not texts:
                continue
            for text in texts:
                # Find character offsets in abstract
                start = abstract.find(text)
                if start != -1:
                    end = start + len(text)
                    entities.append(
                        Entity(text=text, label=label, start=start, end=end)
                    )

        return entities

    @staticmethod
    def _parse_ner_response(response: str, abstract: str) -> Dict[str, List[str]]:
        """
        Extract JSON from model response. Reuses slm_generate helpers for robustness.

        Returns:
        --------
        Dict[str, List[str]]
            Keys are NER labels, values are lists of extracted texts
        """
        # Try to find JSON block in response
        try:
            json_str = extract_json(response)
            result = json_str if isinstance(json_str, dict) else {}
        except Exception:
            result = {}

        # Normalize: convert values to lists
        normalized = {}
        for label in [
            "PROBLEM", "OBJECTIVE", "METHOD", "CONTRIBUTION", "FINDING",
            "IMPROVEMENT", "DATASET", "METRIC", "KEY_CONCEPT"
        ]:
            if label in result:
                val = result[label]
                if isinstance(val, str):
                    normalized[label] = [val] if val.strip() else []
                elif isinstance(val, list):
                    normalized[label] = [v for v in val if isinstance(v, str) and v.strip()]
                else:
                    normalized[label] = []
            else:
                normalized[label] = []

        return normalized

    # ========================================================================
    # Task B: Type Classification (zero-shot, no adapter)
    # ========================================================================

    def predict_contribution_type(
        self,
        contribution_text: str,
        max_new_tokens: int = 10,
        return_raw: bool = False,
    ) -> TypePrediction:
        """
        Classify contribution type using zero-shot Qwen (no adapter).

        Parameters:
        -----------
        contribution_text : str
            CONTRIBUTION aspect text from NER

        max_new_tokens : int
            Max output tokens (default: 10, enough for one word)

        return_raw : bool
            If True, include raw model response in result

        Returns:
        --------
        TypePrediction
            Predicted type with contribution text and confidence
        """
        # Build prompt
        messages = [
            {"role": "system", "content": TYPE_SYSTEM_PROMPT},
            {"role": "user", "content": TYPE_USER_PROMPT_TEMPLATE.format(
                contribution_text=contribution_text
            )},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.ner_model.generate(  # Use ner_model (base with adapter)
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract type from response
        predicted_type = self._parse_type_response(response)

        result = TypePrediction(
            contribution_text=contribution_text,
            predicted_type=predicted_type,
            confidence=None,  # Could add logit-based confidence later
            raw_response=response if return_raw else "",
        )

        return result

    @staticmethod
    def _parse_type_response(response: str) -> str:
        """
        Extract type label from model response.

        Returns:
        --------
        str
            One of: algorithm, dataset, theory, experiment, framework, generalization, software
        """
        response_lower = response.lower().strip()

        # Extract just the type word (should be in response)
        for type_name in CONTRIBUTION_TYPES:
            if type_name in response_lower:
                return type_name

        # Fallback: default to "algorithm" if parsing fails
        return "algorithm"

    # ========================================================================
    # Batch processing
    # ========================================================================

    def predict_paper(
        self,
        abstract: str,
        return_types: bool = True,
    ) -> Dict:
        """
        Full pipeline: extract NER entities, classify CONTRIBUTION type.

        Parameters:
        -----------
        abstract : str
            Paper abstract

        return_types : bool
            If True, classify contribution types for CONTRIBUTION entities

        Returns:
        --------
        Dict
            {
              "entities": [Entity.to_dict(), ...],
              "contribution_types": [TypePrediction.to_dict(), ...],  # if return_types
            }
        """
        # Task A: Extract entities
        entities = self.predict_ner(abstract)

        result = {
            "entities": [e.to_dict() for e in entities],
            "contribution_types": [],
        }

        # Task B: Classify CONTRIBUTION types
        if return_types:
            contributions = [e for e in entities if e.label == "CONTRIBUTION"]
            for contrib in contributions:
                type_pred = self.predict_contribution_type(contrib.text)
                result["contribution_types"].append(type_pred.to_dict())

        return result

    async def predict_paper_batch(
        self,
        abstracts: List[str],
        return_types: bool = True,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Batch process multiple papers.

        Note: LLM inference is not truly async (GPU-bound), but wrapper provides
        consistent interface. For true parallelism, consider multiprocessing.

        Parameters:
        -----------
        abstracts : List[str]
            List of paper abstracts

        return_types : bool
            Classify contribution types

        batch_size : int
            Batch size (currently processes one at a time due to GPU)

        show_progress : bool
            Show tqdm progress bar

        Returns:
        --------
        List[Dict]
            Results for each abstract
        """
        iterator = tqdm(abstracts, disable=not show_progress)
        results = []

        for abstract in iterator:
            result = self.predict_paper(abstract, return_types=return_types)
            results.append(result)

        return results


# ============================================================================
# Utilities
# ============================================================================

def load_pipeline(
    ner_adapter_dir: str,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    **kwargs,
) -> UnifiedQwenPipeline:
    """
    Convenience function to load the unified pipeline.

    Parameters:
    -----------
    ner_adapter_dir : str
        Path to NER LoRA adapter

    base_model_id : str
        Base model ID

    **kwargs : dict
        Additional arguments passed to UnifiedQwenPipeline

    Returns:
    --------
    UnifiedQwenPipeline
        Ready-to-use pipeline
    """
    return UnifiedQwenPipeline(
        ner_adapter_dir=ner_adapter_dir,
        base_model_id=base_model_id,
        **kwargs,
    )


# ============================================================================
# Post-processing: Add type classification to JSON
# ============================================================================

def postprocess_json_with_types(
    pipeline: UnifiedQwenPipeline,
    json_data: Dict,
) -> Dict:
    """
    Post-process JSON: classify CONTRIBUTION types in main paper and references.
    
    Input JSON structure:
    {
      "paper_id": "xxx",
      "title": "xxx",
      "abstract": "xxx",
      "nes": {"CONTRIBUTION": "text", "METHOD": "text", ...},
      "references": {
        "ref_id": {
          "nes": {"CONTRIBUTION": "text", ...},
          "title": "xxx",
          "abstract": "xxx",
          ...
        }
      }
    }
    
    Output: Same JSON with added "aspect_type" key:
    {
      ...
      "aspect_type": {"CONTRIBUTION": "algorithm"},
      "references": {
        "ref_id": {
          ...
          "aspect_type": {"CONTRIBUTION": "dataset"}
        }
      }
    }
    
    Parameters:
    -----------
    pipeline : UnifiedQwenPipeline
        Loaded pipeline
    
    json_data : Dict
        Paper metadata with extracted NER entities
    
    Returns:
    --------
    Dict
        Modified JSON with aspect_type keys added
    """
    # Make a copy to avoid modifying original
    result = json_data.copy()
    main_paper_id = list(result.keys())[0]
    result = result[main_paper_id]
    
    # === Check main paper ===

    main_nes = result.get("nes", {})
    main_contribution = main_nes.get("CONTRIBUTION")
    
    if not main_contribution:
        # No contribution in main, return as-is
        return result,False
    
    # Classify type for main paper
    type_pred = pipeline.predict_contribution_type(main_contribution)
    # result["aspect_type"] = {"CONTRIBUTION": type_pred.predicted_type}
    result["main_aspect_type"] = {"CONTRIBUTION": type_pred.predicted_type}
    
    # === Process references ===
    references = result.get("references", {})
    if references:
        for ref_id, ref_data in references.items():
            ref_nes = ref_data.get("nes", {})
            ref_contribution = ref_nes.get("CONTRIBUTION")
            
            if ref_contribution:
                # Classify type for this reference
                ref_type_pred = pipeline.predict_contribution_type(ref_contribution)
                # ref_data["aspect_type"] = {"CONTRIBUTION": ref_type_pred.predicted_type}
                result["references"][ref_id]["aspect_type"] = {"CONTRIBUTION": ref_type_pred.predicted_type}
    repack_result = {main_paper_id:result}
    return repack_result,True


async def add_contribution_types_to_json(
    json_file_path: str,
    pipeline: UnifiedQwenPipeline,
    output_file_path: Optional[str] = None,
) -> List[Dict]:
    """
    Post-process JSON file: add CONTRIBUTION type classification to main and references.
    
    Reads JSON (line-delimited, one object per line), classifies CONTRIBUTION types,
    adds "aspect_type" key, and saves result.
    
    Only processes papers where main has CONTRIBUTION extracted.
    For each such paper, also processes references with CONTRIBUTION.
    
    Parameters:
    -----------
    json_file_path : str
        Path to input JSON file (JSONL format: one JSON object per line)
    
    pipeline : UnifiedQwenPipeline
        Loaded pipeline
    
    output_file_path : str, optional
        Path to save modified JSON. If None, overwrites input file.
    
    Returns:
    --------
    List[Dict]
        List of modified paper objects
    
    Example:
    --------
    >>> pipeline = UnifiedQwenPipeline(...)
    >>> papers = await add_contribution_types_to_json(
    ...     "papers.jsonl",
    ...     pipeline,
    ...     output_file_path="papers_with_types.jsonl"
    ... )
    """
    import json
    from pathlib import Path
    
    input_path = Path(json_file_path)
    if output_file_path is None:
        output_path = input_path
    else:
        output_path = Path(output_file_path)
    
    # Read input JSONL
    papers = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    paper = json.loads(line)
                    papers.append(paper)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Skipped malformed JSON line: {e}")
                    continue
    
    print(f"[postprocess] Loaded {len(papers)} papers from {input_path}")
    
    # Process each paper
    modified_papers = []
    for i, paper in enumerate(papers):
        modified_paper = postprocess_json_with_types(pipeline, paper)
        modified_papers.append(modified_paper)
        
        if (i + 1) % 100 == 0:
            print(f"[postprocess] Processed {i + 1}/{len(papers)}")
    
    # Write output JSONL
    with open(output_path, "w") as f:
        for paper in modified_papers:
            f.write(json.dumps(paper) + "\n")
    
    print(f"[postprocess] Saved {len(modified_papers)} papers to {output_path}")
    
    return modified_papers


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Load pipeline (loads base model once)
    pipeline = UnifiedQwenPipeline(
        ner_adapter_dir="./path/to/qwen-ner-adapter",  # Your adapter path
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
    )

    # Example abstract
    abstract = """
    We present the Compressive Transformer, an attentive sequence model which
    compresses past memories for long-range sequence learning. We find the
    Compressive Transformer obtains state-of-the-art language modelling results in
    the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc
    respectively. We also find it can model high-frequency speech effectively and
    can be used as a memory mechanism for RL, demonstrated on an object matching
    task. To promote the domain of long-range sequence learning, we propose a new
    open-vocabulary language modelling benchmark derived from books, PG-19.
    """

    # Task A: Extract entities
    print("\n=== Task A: NER ===")
    entities = pipeline.predict_ner(abstract)
    for e in entities:
        print(f"  {e.label}: {e.text[:80]}...")

    # Task B: Classify CONTRIBUTION types
    print("\n=== Task B: Type Classification ===")
    contributions = [e for e in entities if e.label == "CONTRIBUTION"]
    for contrib in contributions:
        type_pred = pipeline.predict_contribution_type(contrib.text)
        print(f"  Type: {type_pred.predicted_type}")
        print(f"  Text: {type_pred.contribution_text[:100]}...")

    # Full pipeline
    print("\n=== Full Pipeline ===")
    result = pipeline.predict_paper(abstract, return_types=True)
    print(f"  Entities: {len(result['entities'])}")
    print(f"  Contribution types: {len(result['contribution_types'])}")
    
    # === Example: Post-process JSON file ===
    print("\n=== Post-processor Example ===")
    # example_json = {
    #     "paper_id": "1911.05507",
    #     "nes": {"CONTRIBUTION": "Compressive Transformer"},
    #     "references": {
    #         "ref1": {"nes": {"CONTRIBUTION": "PG-19 benchmark"}},
    #     }
    # }
    # modified = postprocess_json_with_types(pipeline, example_json)
    # print(f"  Main type: {modified.get('aspect_type', {}).get('CONTRIBUTION')}")
    # print(f"  Ref type: {modified['references']['ref1'].get('aspect_type', {}).get('CONTRIBUTION')}")