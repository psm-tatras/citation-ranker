#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen2.5-7B-Instruct citation relation
classification (FUNCTION/M_ASPECT/POLARITY/EVIDENCE/R_MATCH).
Prompt selected via --version (utils/prompts.py CITATION_CLASSIFIER_SLM_VERSIONS):
  v1 (default): the lean SLM prompt, CONFIDENCE removed from the DeepSeek
                 teacher prompt (LAYER_2_RELATION_USER_TEMPLATE_v2).
With W&B dataset integration and S3 model upload.

This uses:
  - 4-bit quantization (bitsandbytes)
  - LoRA adapters (peft)
  - SFTTrainer (trl) for supervised fine-tuning
  - W&B for experiment tracking and dataset lineage
  - S3 for model storage (IAM role auth)

Unlike train_qlora_narrative.py, the registered W&B model artifact's metadata
also carries the FULL CITATION_CLASSIFIER_SLM_VERSIONS[version] prompt config
(system prompt, user template, label sets) -- not just the version string --
so the artifact is self-describing: a later inference script can pull the
adapter and know exactly which prompt to call it with, even if
utils/prompts.py has moved on to a newer version by then.

Loss is computed on the assistant (label JSON) tokens only, via
DataCollatorForCompletionOnlyLM. Before training, every train/val example is
tokenized and the run aborts if any exceeds --max-seq-length (truncation
previously cut every example before its citation context and target) or lacks
the "<|im_start|>assistant\n" response template (it would be fully masked and
contribute no loss).

Memory profile: ~8-11 GB for 7B model with batch size 2-4
"""

import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
import os
import json
import boto3
import datetime
import wandb
from dotenv import load_dotenv

from utils.load_model import (
    WANDB_PROJECT,
    load_latest_adapter,
)
from utils.save_model import BestCheckpointS3WandbCallback, lookup_best_loss

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — values can be provided in .env or overridden by CLI options
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET_NAME", "")
# Context classifier's own top-level directory within the shared bucket (NOT
# a separate bucket) -- e.g. s3://psm-iiser/context-classifier/. Each run's
# own timestamp directory (computed below) sits directly under this prefix.
S3_KEY_PREFIX = os.getenv("S3_KEY_PREFIX_CONTEXT_CLASSIFIER", "context-classifier")

# NOT utils.load_model.WANDB_ADAPTER_ARTIFACT_NAME -- that's shared with
# train_qlora.py (NER) and train_qlora_narrative.py. Context classifier needs
# its own name/env var so all three never collide regardless of how
# WANDB_ADAPTER_ARTIFACT_NAME is set.
WANDB_ADAPTER_ARTIFACT_NAME = os.getenv(
    "WANDB_CONTEXT_CLASSIFIER_ADAPTER_ARTIFACT_NAME", "qwen-context-classifier-adapter")

WANDB_DATASET_ARTIFACT = os.getenv("WANDB_DATASET_ARTIFACT", "dataset-context-classification-56main:latest")
WANDB_DATASET_PROJECT = os.getenv("WANDB_DATASET_PROJECT", "")
WANDB_ARTIFACT_TRAIN_FILE = os.getenv("WANDB_ARTIFACT_TRAIN_FILE", "citation_contexts_layer2_train_slm_v1.jsonl")
WANDB_ARTIFACT_VAL_FILE = os.getenv("WANDB_ARTIFACT_VAL_FILE", "citation_contexts_layer2_val_slm_v1.jsonl")

TRAINING_MODE = "new"  # "new" = fresh LoRA adapter; "resume" = load adapter weights from S3 first


def setup_quantization_config():
    """Create 4-bit quantization config for memory efficiency."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def setup_lora_config(lora_rank=16, lora_alpha=32, lora_dropout=0.05):
    """Create LoRA adapter config."""
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Qwen2.5 attention layers
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def upload_to_s3(local_dir: Path, s3_bucket: str, s3_key_prefix: str) -> str:
    """
    Upload adapter model to S3 using IAM role authentication.

    Args:
        local_dir: Local directory containing adapter files
        s3_bucket: S3 bucket name (e.g., 'my-ml-bucket')
        s3_key_prefix: S3 key prefix (e.g., 'models/qwen-context-classifier/171024-143022')

    Returns:
        S3 URI of the uploaded model (e.g., 's3://bucket/prefix')
    """
    s3_client = boto3.client("s3")

    uploaded_count = 0
    for local_file in local_dir.rglob("*"):
        if not local_file.is_file():
            continue

        # Construct S3 key: prefix + relative path from local_dir.parent, so
        # the uploaded "adapter_model/" folder name is preserved in the key
        # (matching train_qlora.py's upload_to_s3 and what
        # utils.load_model.load_latest_adapter() expects: files nested under
        # {prefix}/adapter_model/, not directly under {prefix}/).
        relative_path = local_file.relative_to(local_dir.parent)
        s3_key = f"{s3_key_prefix.rstrip('/')}/{relative_path}".replace("\\", "/")

        s3_client.upload_file(
            Filename=str(local_file),
            Bucket=s3_bucket,
            Key=s3_key,
        )
        uploaded_count += 1

    s3_uri = f"s3://{s3_bucket}/{s3_key_prefix}"
    print(f"  Uploaded {uploaded_count} files to {s3_uri}")
    return s3_uri


def load_data(data_file: Path):
    """Load JSONL dataset with chat format messages."""
    dataset = load_dataset("json", data_files=str(data_file))["train"]
    return dataset


def _contains_subsequence(ids, sub):
    """True if token list `sub` occurs contiguously in `ids`."""
    k = len(sub)
    return any(ids[i:i + k] == sub for i in range(len(ids) - k + 1))


def check_examples(dataset, formatting_func, tokenizer, max_seq_length, response_template_ids, name):
    """Tokenize every example as the trainer will; abort if any would be truncated
    or lacks the assistant response template (completion-only loss would then
    mask the whole example and it would train on nothing)."""
    lengths, missing = [], []
    for i, ex in enumerate(dataset):
        ids = tokenizer(formatting_func(ex), add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if not _contains_subsequence(ids[:max_seq_length], response_template_ids):
            missing.append(i)

    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p99 = sorted_lengths[min(n - 1, int(0.99 * n))]
    over = sum(1 for length in lengths if length > max_seq_length)
    print(f"  {name}: tokens max={sorted_lengths[-1]} p99={p99} median={sorted_lengths[n // 2]} "
          f"(max_seq_length={max_seq_length}); missing response template: {len(missing)}")
    if over:
        raise ValueError(
            f"{over}/{n} {name} examples exceed max_seq_length={max_seq_length} "
            f"(max {sorted_lengths[-1]} tokens) and would be truncated before the assistant "
            f"target. Raise --max-seq-length or check the dataset's prompt version.")
    if missing:
        raise ValueError(
            f"{len(missing)}/{n} {name} examples lack the response template "
            f"{tokenizer.decode(response_template_ids)!r} (first indices: {missing[:10]}). "
            f"Check the chat template / tokenizer and that every example has an assistant turn.")


def train_qlora_context_classifier(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_file: Path = Path("training_data_train.jsonl"),
    val_file: Path = Path("training_data_val.jsonl"),
    output_dir: Path = Path("checkpoints/qwen_context_classifier"),
    epochs: int = 3,
    batch_size: int = 2,
    grad_accumulation: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 2048,
    save_steps: int = 100,
    s3_bucket: str = S3_BUCKET,
    s3_key_prefix: str = S3_KEY_PREFIX,
    wandb_project: str = WANDB_PROJECT,
    dataset_artifact: str = WANDB_DATASET_ARTIFACT,
    dataset_wandb_project: str = WANDB_DATASET_PROJECT,
    train_artifact_file: str = WANDB_ARTIFACT_TRAIN_FILE,
    val_artifact_file: str = WANDB_ARTIFACT_VAL_FILE,
    mode: str = TRAINING_MODE,
    run_name: str = None,
    resume_checkpoint_kind: str = "final",
    version: str = "v1",
):
    """
    Fine-tune Qwen2.5-7B with QLoRA on citation relation classification.
    Prompt is selected via `version` (see its docstring below).
    Uploads best model to S3 and registers in W&B, with the FULL SLM prompt
    config embedded in the registered artifact's metadata.

    Args:
        model_name: HuggingFace model ID
        train_file: JSONL file with training data
        val_file: JSONL file with validation data
        output_dir: Directory to save checkpoints and adapter
        epochs: Training epochs
        batch_size: Per-device batch size
        grad_accumulation: Gradient accumulation steps
        learning_rate: Learning rate for LoRA params
        warmup_steps: Number of warmup steps
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha
        max_seq_length: Max token length
        save_steps: Save checkpoint every N steps
        s3_bucket: S3 bucket name (e.g., 'my-ml-bucket')
        s3_key_prefix: root S3 key prefix runs are stored under
        wandb_project: W&B project this run logs to
        dataset_artifact: W&B dataset artifact reference (e.g., 'dataset-layer2-v2:latest')
        dataset_wandb_project: W&B project the dataset artifact lives in, if different
        train_artifact_file: name of the training-data file inside the artifact
        val_artifact_file: name of the validation-data file inside the artifact
        mode: 'new' starts LoRA training from scratch; 'resume' loads adapter from S3
        run_name: W&B run name; if None, W&B auto-generates one
        resume_checkpoint_kind: which artifact 'resume' loads from -- 'final'
            (default) is the plain WANDB_ADAPTER_ARTIFACT_NAME, only registered
            when a run completes fully; 'train-best'/'eval-best' load the
            best-so-far checkpoint of that kind instead, registered live
            during training by utils.save_model.BestCheckpointS3WandbCallback
            -- use one of these to resume a run that was interrupted before
            it could register a 'final' artifact.
        version: which CITATION_CLASSIFIER_SLM_VERSIONS entry (utils/prompts.py)
            this model is intended to be called with at inference -- training
            itself just trains on whatever "messages" are in the downloaded
            JSONL, regardless of this value; `version` only controls what
            gets recorded in W&B run config / checkpoint metadata / the final
            adapter's metadata (including the full prompt text). MUST match
            the prompt dataset_artifact's messages were actually rendered
            with, or every W&B record for this run will misdescribe the model.
    """
    from utils.prompts import CITATION_CLASSIFIER_SLM_VERSIONS
    prompt_cfg = CITATION_CLASSIFIER_SLM_VERSIONS[version]
    functions = list(prompt_cfg["functions"])
    # JSON/W&B-safe: tuples aren't valid in wandb config/metadata payloads.
    prompt_metadata = {k: (list(v) if isinstance(v, tuple) else v) for k, v in prompt_cfg.items()}

    # trl is only needed for actual training, so pulled in here
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    print("=" * 70)
    print("QLoRA Fine-Tuning: Qwen2.5-7B-Instruct for Citation Relation Classification")
    print(f"Version: {version}  Functions: {', '.join(functions)}")
    print("=" * 70)

    # Initialize W&B
    print("\n[0/6] Initializing W&B...")
    run = wandb.init(
        project=wandb_project,
        name=run_name or None,
        job_type="train",
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accumulation": grad_accumulation,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "max_seq_length": max_seq_length,
            "version": version,
            "functions": functions,
            "prompt": prompt_metadata,
        }
    )

    # Download dataset artifact from W&B
    dataset_artifact_ref = (
        f"{dataset_wandb_project}/{dataset_artifact}" if dataset_wandb_project else dataset_artifact
    )
    print(f"\n[1/6] Downloading dataset artifact '{dataset_artifact_ref}' from W&B...")
    try:
        artifact = wandb.use_artifact(dataset_artifact_ref)
        print(f"  Version: {artifact.version}")
        print(f"  Description: {artifact.description or '(no description)'}")
        needed_files = {train_artifact_file, val_artifact_file}
        downloaded = {}
        for name in needed_files:
            local_path = artifact.get_entry(name).download()
            downloaded[name] = local_path
            print(f"  Downloaded: {name} -> {local_path}")
        train_file = Path(downloaded[train_artifact_file])
        val_file = Path(downloaded[val_artifact_file])
        print(f"✓ Using training file: {train_file}")
        print(f"✓ Using validation file: {val_file}")
    except Exception as e:
        print(f"✗ Failed to download dataset artifact: {e}")
        raise

    # Load model
    print("\n[2/6] Loading model...")
    quantization_config = setup_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Model loaded: {model_name}")

    # Load tokenizer
    print("[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    def formatting_func(example):
        """Format chat messages into a single text string for training."""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Setup LoRA
    print("\n[3/6] Setting up LoRA adapters...")
    if mode == "resume":
        resumed_adapter_dir = output_dir / "resumed_adapter"
        adapter_path = load_latest_adapter(
            resumed_adapter_dir,
            wandb_project=wandb_project,
            artifact_name=WANDB_ADAPTER_ARTIFACT_NAME,
            wandb_run=run,
            checkpoint_kind=None if resume_checkpoint_kind == "final" else resume_checkpoint_kind,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        print("=" * 70)
        print(f"✓ WEIGHTS LOADED FROM S3 (checkpoint_kind={resume_checkpoint_kind}): {adapter_path}")
        print("=" * 70)
    else:
        lora_config = setup_lora_config(lora_rank=lora_rank, lora_alpha=lora_alpha)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("\n[4/6] Loading training data...")
    train_dataset = load_data(train_file)
    val_dataset = load_data(val_file)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    # Loss only on the assistant turn: everything up to and including Qwen's
    # "<|im_start|>assistant\n" is masked to -100. Token ids (not the string)
    # so the template matches regardless of surrounding tokenization.
    response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    check_examples(train_dataset, formatting_func, tokenizer, max_seq_length, response_template_ids, "train")
    check_examples(val_dataset, formatting_func, tokenizer, max_seq_length, response_template_ids, "val")
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Training arguments
    print("\n[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=save_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to=["wandb", "tensorboard"],
        push_to_hub=False,
        run_name=f"qwen-context-classifier-lora-rank{lora_rank}",
    )

    # Saves+uploads a new best train/eval checkpoint to S3 + W&B immediately
    # whenever loss improves, so a premature stop (crash, KeyboardInterrupt,
    # preemption) never loses progress -- same rationale as train_qlora_narrative.py.
    #
    # On resume, seed the "best so far" baselines from whatever's already
    # recorded in W&B (regardless of which checkpoint_kind we just loaded
    # WEIGHTS from) -- otherwise these would reset to +inf and the very next
    # logged loss would immediately count as "best" and trigger a save even
    # if it's actually worse than the checkpoint being resumed from.
    if mode == "resume":
        initial_best_train_loss = lookup_best_loss(
            wandb_project, WANDB_ADAPTER_ARTIFACT_NAME, "train-best", "train_loss")
        initial_best_eval_loss = lookup_best_loss(
            wandb_project, WANDB_ADAPTER_ARTIFACT_NAME, "eval-best", "eval_loss")
        print(f"Resume baselines: train_loss={initial_best_train_loss}, eval_loss={initial_best_eval_loss}")
    else:
        initial_best_train_loss = float("inf")
        initial_best_eval_loss = float("inf")

    # Computed ONCE up front (not just at the end for the final upload), so
    # train-best/eval-best checkpoints saved DURING training land under the
    # SAME per-run directory as the final adapter, instead of a bare shared
    # path that every run of this model would overwrite into.
    run_timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    run_s3_prefix = f"{s3_key_prefix.rstrip('/')}/{run_timestamp}" if s3_key_prefix else run_timestamp

    best_checkpoint_callback = BestCheckpointS3WandbCallback(
        tokenizer=tokenizer,
        s3_bucket=s3_bucket,
        s3_key_prefix_root=run_s3_prefix,
        wandb_run=run,
        adapter_artifact_name=WANDB_ADAPTER_ARTIFACT_NAME,
        artifact_metadata_extra={
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "version": version,
            "functions": functions,
            "prompt": prompt_metadata,
            "dataset_artifact": dataset_artifact_ref,
        },
        tmp_root=str(output_dir / "_best_tmp"),
        initial_best_train_loss=initial_best_train_loss,
        initial_best_eval_loss=initial_best_eval_loss,
    )

    # Create trainer (standard SFTTrainer)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        packing=False,  # Chat format doesn't benefit from packing
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=[best_checkpoint_callback],
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    trainer.train()

    # Save adapter
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    model.save_pretrained(output_dir / "adapter_model")
    tokenizer.save_pretrained(output_dir / "adapter_model")
    print(f"✓ Adapter saved to: {output_dir}/adapter_model")
    print(f"✓ Tokenizer saved to: {output_dir}/adapter_model")

    # Print adapter size
    adapter_size = sum(
        f.stat().st_size for f in (output_dir / "adapter_model").rglob("*") if f.is_file()
    ) / 1e6
    print(f"✓ Adapter size: {adapter_size:.1f} MB")

    # Upload to S3 -- run_s3_prefix was already computed above, before
    # training started, so this lands under the SAME per-run directory as
    # the train-best/eval-best checkpoints saved during training.
    print("\n" + "=" * 70)
    print("Uploading to S3...")
    print("=" * 70)
    s3_uri = None
    try:
        s3_uri = upload_to_s3(
            local_dir=output_dir / "adapter_model",
            s3_bucket=s3_bucket,
            s3_key_prefix=run_s3_prefix
        )
    except Exception as e:
        print(f"✗ S3 upload failed: {e}")

    # Log adapter and S3 location as artifact to W&B -- metadata includes the
    # FULL prompt config (system/template/label sets), not just `version`, so
    # the artifact is self-describing even if utils/prompts.py moves on.
    print("\nRegistering model in W&B...")
    adapter_artifact = wandb.Artifact(
        name=WANDB_ADAPTER_ARTIFACT_NAME,
        type="model",
        description=f"QLoRA adapter (rank={lora_rank}) fine-tuned on citation relation classification "
                    f"({version}, functions: {'/'.join(functions)})",
        metadata={
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "version": version,
            "functions": functions,
            "prompt": prompt_metadata,
            "dataset_artifact": dataset_artifact_ref,
            "s3_uri": s3_uri if s3_uri else "upload-failed",
            "s3_bucket": s3_bucket,
            "s3_key_prefix": run_s3_prefix,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

    # Deliberately NOT adapter_artifact.add_dir(...) -- the adapter files
    # already live in S3 (s3_uri above); this artifact is a metadata-only
    # pointer to that S3 location, not a second copy of the weights in W&B.
    #
    # A tiny pointer file IS attached (not the weights): W&B's "already
    # exists with the same content, no new version created" dedup check is
    # based on the file manifest, not the metadata dict -- with zero files,
    # every registration has an identical (empty) manifest, so W&B would
    # silently keep reusing the very first version forever and this run's
    # actual metadata would never get persisted. Writing the metadata itself
    # as a small JSON file gives each version genuinely different content.
    with adapter_artifact.new_file("pointer.json", mode="w") as f:
        json.dump(adapter_artifact.metadata, f, indent=2, default=str)

    # Local reference file only (not uploaded to W&B).
    if s3_uri:
        s3_info_file = output_dir / "S3_LOCATION.txt"
        with open(s3_info_file, 'w') as f:
            f.write(f"S3 URI: {s3_uri}\n")
            f.write(f"Bucket: {s3_bucket}\n")
            f.write(f"Key Prefix: {run_s3_prefix}\n")
            f.write(f"Uploaded: {datetime.datetime.now().isoformat()}\n")

    run.log_artifact(adapter_artifact)
    print(f"✓ Model registered in W&B (metadata only -- weights live in S3)")

    # Finalize W&B run
    run.finish()

    print("\n" + "=" * 70)
    print("Training & Upload Complete!")
    print("=" * 70)
    if s3_uri:
        print(f"✓ Model uploaded to: {s3_uri}")
    print(f"✓ Model registered in W&B")
    print(f"✓ Dataset artifact linked: {dataset_artifact_ref}")
    print("\nNext steps:")
    print(f"1. Check W&B dashboard: https://wandb.ai/your-username/{wandb_project}")
    if s3_uri:
        print(f"2. Download from S3 for inference:")
        print(f"   aws s3 cp {s3_uri} ./adapter-model/ --recursive")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for citation relation classification with W&B & S3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--train-file", type=Path, default=Path("training_data_train.jsonl"), help="Local training file")
    parser.add_argument("--val-file", type=Path, default=Path("training_data_val.jsonl"), help="Local validation file")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen_context_classifier"), help="Checkpoint directory")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT, help="W&B project this run logs to")
    parser.add_argument("--dataset-artifact", default=WANDB_DATASET_ARTIFACT, help="W&B dataset artifact reference")
    parser.add_argument("--dataset-wandb-project", default=WANDB_DATASET_PROJECT, help="W&B project the dataset artifact lives in (default: same)")
    parser.add_argument("--train-artifact-file", default=WANDB_ARTIFACT_TRAIN_FILE, help="Training file name inside artifact")
    parser.add_argument("--val-artifact-file", default=WANDB_ARTIFACT_VAL_FILE, help="Validation file name inside artifact")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--mode", choices=["new", "resume"], default=TRAINING_MODE,
                         help="'new' trains from scratch; 'resume' loads adapter weights from S3 first")
    parser.add_argument("--resume-checkpoint-kind", choices=["final", "train-best", "eval-best"], default="final",
                         help="Which artifact --mode resume loads: 'final' (default) is only registered when a "
                              "run completes fully; use 'train-best'/'eval-best' to resume a run that was "
                              "interrupted before finishing")
    parser.add_argument("--version", choices=["v1"], default="v1",
                         help="Which CITATION_CLASSIFIER_SLM_VERSIONS prompt (utils/prompts.py) this model is "
                              "intended to be called with at inference -- MUST match the prompt --dataset-artifact's "
                              "messages were actually rendered with, or W&B will misdescribe this run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)

    args = parser.parse_args()

    if not S3_BUCKET:
        print("⚠ Warning: S3_BUCKET is not set (check your .env)")

    train_qlora_context_classifier(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        dataset_artifact=args.dataset_artifact,
        dataset_wandb_project=args.dataset_wandb_project,
        train_artifact_file=args.train_artifact_file,
        val_artifact_file=args.val_artifact_file,
        mode=args.mode,
        run_name=args.run_name,
        resume_checkpoint_kind=args.resume_checkpoint_kind,
        version=args.version,
    )

# Example run. Run from the project root (~/seku-crd-citation-processor) as a
# module -- NOT `python training/train_qlora_context_classifier.py`, which
# only puts training/ on the import path and fails with "ModuleNotFoundError:
# No module named 'utils'".
# (Equivalently: `PYTHONPATH=. python training/train_qlora_context_classifier.py ...`)
#
# python -m training.train_qlora_context_classifier \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --output-dir checkpoints/qwen_context_classifier \
#     --wandb-project psm-iiser-citation-relevancy \
#     --dataset-artifact dataset-context-classification-56main:latest \
#     --dataset-wandb-project psm-iiser-citation-relevancy \
#     --train-artifact-file citation_contexts_layer2_train_slm_v1.jsonl \
#     --val-artifact-file citation_contexts_layer2_val_slm_v1.jsonl \
#     --version v1 \
#     --run-name Run-context-classifier-v1 \
#     --mode new \
#     --epochs 2 \
#     --batch-size 2 \
#     --grad-accumulation 4 \
#     --learning-rate 5e-5 \
#     --warmup-steps 200 \
#     --lora-rank 16 \
#     --lora-alpha 32 \
#     --max-seq-length 2048 \
#     --save-steps 50
#
# --train-file/--val-file are ignored once --dataset-artifact download
# succeeds (it overwrites train_file/val_file with the downloaded paths) --
# they're only used as a fallback label before that point, so any value works.
# For --mode resume, drop --dataset-artifact and friends if you want the same
# dataset artifact as the run being resumed, or override them to switch datasets.

# Resuming a run that was interrupted before it could register a 'final'
# artifact (KeyboardInterrupt, crash, preemption): add --resume-checkpoint-kind
# train-best (most recent progress) or eval-best (best generalization, updates
# once per eval) -- otherwise --mode resume silently falls back to 'final',
# which may be stale or not exist at all.
#
# python -m training.train_qlora_context_classifier \
#     --wandb-project psm-iiser-citation-relevancy \
#     --dataset-artifact dataset-context-classification-56main:latest \
#     --dataset-wandb-project psm-iiser-citation-relevancy \
#     --train-artifact-file citation_contexts_layer2_train_slm_v1.jsonl \
#     --val-artifact-file citation_contexts_layer2_val_slm_v1.jsonl \
#     --version v1 \
#     --run-name Run-context-classifier-v1-resume-from-train-best \
#     --mode resume \
#     --resume-checkpoint-kind train-best \
#     --epochs 2 \
#     --batch-size 2 \
#     --grad-accumulation 4 \
#     --learning-rate 5e-5 \
#     --warmup-steps 200 \
#     --save-steps 50