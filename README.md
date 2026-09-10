# Seku CRD Citation Processor

Tools for extracting research-role entities from scientific abstracts and training QLoRA adapters for Qwen-based NER.

## Project structure

- `training/convert_ner_format_to_qwen_sft.py` converts span annotations to Qwen chat-format JSONL.
- `training/train_qlora.py` fine-tunes a Qwen model with QLoRA and optionally integrates W&B and S3.
- `prediction/predict_ner.py` provides NER, context-window, and contribution-generation helpers.
- `utils/` contains shared utilities.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the AWS and W&B values in `.env` before running corpus or training jobs.
The real `.env` file is ignored by Git; never commit credentials.

For GPU training, install the PyTorch build appropriate for the CUDA version before installing the remaining dependencies.

## Current status

The scripts were migrated from an older project. Some inference paths still refer to modules that have not yet been migrated, including `slm_generate` and `train_contribution_ner_augmented`. Those modules must be added before the corresponding inference functions can run.
