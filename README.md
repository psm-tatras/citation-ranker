# Seku CRD Citation Processor

Extracts research-role entities (PROBLEM, METHOD, CONTRIBUTION, FINDING, ...) from
scientific paper abstracts with a QLoRA-fine-tuned Qwen NER model, then scores how
relevant each cited reference is to a main paper by embedding and comparing the
extracted entity text per aspect. Output is a per-paper JSON report that can be
queried for the top-N most relevant references on a given aspect.

## Pipeline overview

1. **Train the NER adapter** (`training/`) — fine-tune Qwen2.5-7B-Instruct with
   QLoRA on span-annotated abstracts, tracked in W&B, weights shipped to S3.
2. **Generate citation relevance reports** (`seku_citation_relevance_main.py`) —
   for a sample of main papers: pull the paper + its in-corpus references from
   S3, run NER on all of them, embed each aspect's extracted text (SciBERT),
   and score every reference's relevance to the main paper per aspect via
   cosine similarity. Writes one JSON report per main paper.
3. **Query the reports** (`utils/query_n_search.py`) — given a report, rank its
   references by relevance score for one aspect and dump the top N.

## Project structure

- `training/`
  - `convert_ner_format_to_qwen_sft.py` — converts span annotations
    (`ner_spans.jsonl`) into Qwen chat-format instruction/completion pairs.
  - `train_qlora.py` — QLoRA fine-tuning of Qwen2.5-7B-Instruct for NER
    (4-bit, LoRA, `trl.SFTTrainer`, W&B tracking, S3 upload).
  - `train_data_generator/` — synthetic NER dataset generation notebook/script.
- `prediction/`
  - `predict_ner.py` — original free-function inference helpers (NER, context
    windowing, entity highlighting).
  - `predict_ner_v2.py` — class-based equivalents: `NERModel`,
    `FineTunedNERModel` (QLoRA adapter, with batched-async `apredict`),
    `SummarizerModel`, `ContextAugmenter`.
  - `predict_ner_v3.py` — `UnifiedQwenPipeline`: loads the Qwen base model
    once and reuses it for both NER (LoRA adapter) and zero-shot CONTRIBUTION
    type classification, to cut memory/latency vs. loading two models.
  - `slm_generate.py` — JSON extraction / offset-resolution helpers shared by
    the inference and SFT-conversion code, so train and inference parsing
    always match.
- `utils/`
  - `seku_crd_corpus_utils.py` — paper index loading and async S3 fetch
    helpers for the citation corpus.
  - `embedding_utils.py` — `EmbeddingModel` (batched HF encoder embeddings)
    and `reference_relevance_computation` (cosine similarity vs. a base text).
  - `load_model.py` — resolves and downloads the latest NER adapter via its
    W&B model artifact (shared by training-resume and inference).
  - `prompts.py` — shared label set (`LABELS`) and prompt templates for NER
    and type classification.
  - `register_ner_dataset_wandb.py` — registers a converted SFT dataset
    directory as a W&B dataset artifact.
  - `query_n_search.py` — `aspect_query`: ranks a report's references by one
    aspect's relevance score and writes the top N to a file.
- `seku_citation_relevance_main.py` — the end-to-end report-generation driver
  described above.
- `playground/` — exploratory notebooks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For GPU work, install the CUDA-matched PyTorch build before the rest of
`requirements.txt`. The real `.env` is git-ignored; never commit credentials.

`.env.example` covers the AWS/W&B basics. Running `seku_citation_relevance_main.py`
also needs these (add them to your own `.env`):

```
NER_MODEL_LOCAL_DIR=   # local dir the NER adapter is downloaded/loaded from
RESULT_DIR=            # where per-paper JSON reports + processed/failed logs are written
MAX_REFERENCES=32      # cap on references processed per main paper
MAX_MAIN_PAPERS=2000   # how many papers to load from the paper index
TEST_VOLUME=5          # how many main papers to sample per run

EMBEDDING_MODEL_NAME=allenai/scibert_scivocab_uncased
EMBEDDING_MODEL_LOCAL_DIR=   # local dir holding the embedding model's own files
EMBEDDING_BATCH_SIZE=8
```

`WANDB_ADAPTER_ARTIFACT_NAME` (used alongside `WANDB_PROJECT`) is also required —
it names the W&B model artifact `load_model.py` resolves "latest" from.

## Running things

VS Code launch configs are in `.vscode/launch.json`:

- **Convert NER spans -> Qwen SFT**
- **Register NER dataset in W&B**
- **Generate Citation Relevance Report** — runs `seku_citation_relevance_main.py`
- **Query Top-N References by Aspect** — runs `utils/query_n_search.py`

Or from the command line (module form, so relative imports resolve):

```bash
python -m training.convert_ner_format_to_qwen_sft --input_dir <dir> --val_frac 0.15
python seku_citation_relevance_main.py
python -m utils.query_n_search --filename <report.json> --aspect CONTRIBUTION \
    --top_n 10 --output_dir <dir> [--aspect_only]
```

`seku_citation_relevance_main.py` resumes on rerun: it skips main paper IDs
already recorded in `RESULT_DIR/processed_main.jsonl`, and logs failures to
`RESULT_DIR/failed_main.jsonl`.

## Known gaps

- `predict_ner.py` / `predict_ner_v2.py` reference a seq2seq contribution
  generator (`train_contribution_baseline.py`) that hasn't been migrated into
  this repo yet — `SummarizerModel`/`load_summarizer_model` will load a model
  dir if you point them at one, but there's no training script here for it.
