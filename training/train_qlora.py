#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen2.5-7B-Instruct NER extraction.
With W&B dataset integration and S3 model upload.

This uses:
  - 4-bit quantization (bitsandbytes)
  - LoRA adapters (peft)
  - SFTTrainer (trl) for supervised fine-tuning
  - W&B for experiment tracking and dataset lineage
  - S3 for model storage (IAM role auth)
  
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
    WANDB_ADAPTER_ARTIFACT_NAME,
    load_latest_adapter,
)
from utils.save_model import BestCheckpointS3WandbCallback

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — values can be provided in .env or overridden by CLI options
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET_NAME", "")
# Deliberately NOT falling back to AWS_DIRECTORY -- that var is already used
# elsewhere in this repo (utils/seku_crd_corpus_utils.py) for the citation
# corpus index prefix, which is unrelated to model checkpoints.
S3_KEY_PREFIX = os.getenv("S3_KEY_PREFIX") or os.getenv("AWS_MODEL_DIRECTORY", "")

WANDB_DATASET_ARTIFACT = os.getenv("WANDB_DATASET_ARTIFACT", "dataset-ner-3k:latest")
# The dataset artifact can live in a different W&B project than the one this
# run logs to (e.g. registered separately via register_ner_dataset_wandb.py
# --project). Empty means "same project as WANDB_PROJECT" (unqualified lookup).
WANDB_DATASET_PROJECT = os.getenv("WANDB_DATASET_PROJECT", "")

# Names of the individual files to pull out of the dataset artifact (not the
# whole artifact). Set in env / .env, e.g. WANDB_ARTIFACT_TRAIN_FILE=ner_bio_train.jsonl
WANDB_ARTIFACT_TRAIN_FILE = os.getenv("WANDB_ARTIFACT_TRAIN_FILE", "ner_bio.jsonl")
WANDB_ARTIFACT_VAL_FILE = os.getenv("WANDB_ARTIFACT_VAL_FILE", "ner_bio.jsonl")

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


def get_default_class_weights():
    """
    Return inverse-frequency class weights based on dataset distribution.
    
    Dataset stats (from synthetic_dgft_abstracts_6000_ner_spans.jsonl):
    KEY_CONCEPT: 44.2%, METHOD: 13.3%, METRIC: 9.9%, CONTRIBUTION: 9.8%,
    FINDING: 9.6%, DATASET: 3.9%, PROBLEM: 3.7%, OBJECTIVE: 3.2%, IMPROVEMENT: 2.4%
    
    Weights are inverse frequency, normalized so the most frequent class has weight 1.0.
    This upweights rare labels (IMPROVEMENT 21.7x, OBJECTIVE 13.7x, PROBLEM 12.0x).
    
    Returns:
        dict: {label -> weight}
    """
    return {
        "KEY_CONCEPT": 1.0,      # 44.2% (baseline, most frequent)
        "METHOD": 3.3,           # 13.3% (upweight by 3.3x)
        "METRIC": 4.5,           # 9.9% (upweight by 4.5x)
        "CONTRIBUTION": 4.5,     # 9.8% (upweight by 4.5x)
        "FINDING": 4.6,          # 9.6% (upweight by 4.6x)
        "DATASET": 11.3,         # 3.9% (upweight by 11.3x)
        "PROBLEM": 12.0,         # 3.7% (upweight by 12.0x)
        "OBJECTIVE": 13.7,       # 3.2% (upweight by 13.7x)
        "IMPROVEMENT": 21.7,     # 2.4% (upweight by 21.7x — rarest label)
    }


def upload_to_s3(local_dir: Path, s3_bucket: str, s3_key_prefix: str) -> str:
    """
    Upload adapter model to S3 using IAM role authentication.
    
    Args:
        local_dir: Local directory containing adapter files
        s3_bucket: S3 bucket name (e.g., 'my-ml-bucket')
        s3_key_prefix: S3 key prefix (e.g., 'models/qwen-ner/')
    
    Returns:
        S3 URI (e.g., 's3://my-ml-bucket/models/qwen-ner/adapter_model/')
    """
    print(f"\n[Upload] Uploading adapter to S3...")
    print(f"  Bucket: {s3_bucket}")
    print(f"  Prefix: {s3_key_prefix}")
    
    # Initialize S3 client (uses IAM role from EC2)
    s3_client = boto3.client('s3')
    
    uploaded_files = 0
    total_size_mb = 0
    
    try:
        # Recursively upload all files from adapter directory
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Construct S3 key
                relative_path = file_path.relative_to(local_dir.parent)
                s3_key = f"{s3_key_prefix.rstrip('/')}/{relative_path}".replace("\\", "/")
                
                # Upload file
                file_size_mb = file_path.stat().st_size / 1e6
                print(f"  Uploading: {s3_key} ({file_size_mb:.2f} MB)")
                
                s3_client.upload_file(
                    str(file_path),
                    s3_bucket,
                    s3_key
                )
                uploaded_files += 1
                total_size_mb += file_size_mb
        
        s3_uri = f"s3://{s3_bucket}/{s3_key_prefix}"
        print(f"✓ Upload complete: {uploaded_files} files ({total_size_mb:.2f} MB)")
        print(f"✓ S3 URI: {s3_uri}")
        return s3_uri
    
    except Exception as e:
        print(f"✗ S3 upload failed: {e}")
        raise


def load_data(data_file: Path, max_samples: int = None):
    """Load instruction/completion JSONL data."""
    print(f"Loading data from {data_file}...")
    dataset = load_dataset("json", data_files=str(data_file))["train"]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples (limited to {max_samples})")
    else:
        print(f"Using all {len(dataset)} samples")
    
    return dataset


def train_qlora(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_file: Path = Path("training_data_train.jsonl"),
    val_file: Path = Path("training_data_val.jsonl"),
    output_dir: Path = Path("checkpoints/qwen_ner"),
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
    class_weights: dict = None,
):
    """
    Fine-tune Qwen2.5-7B with QLoRA on research abstract NER.
    Uploads best model to S3 and registers in W&B.
    
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
        s3_key_prefix: root S3 key prefix runs are stored under (e.g.
            'models/qwen-ner/'). Each run uploads to its own timestamped
            subfolder, s3_key_prefix/DDMMYY-HHMMSS/, so past runs are never
            overwritten.
        wandb_project: W&B project this run logs to (the training run + adapter artifact)
        dataset_artifact: W&B dataset artifact reference (e.g., 'dataset-ner-3k:latest')
        class_weights: dict mapping label names to weight values for class balancing.
            If None, uses default inverse-frequency weights. Example:
            {"IMPROVEMENT": 21.7, "OBJECTIVE": 13.7, "KEY_CONCEPT": 1.0}
        dataset_wandb_project: W&B project the dataset artifact lives in, if different
            from wandb_project (e.g. a shared datasets project registered via
            register_ner_dataset_wandb.py --project). Empty = same as wandb_project.
        train_artifact_file: name of the training-data file inside the artifact
        val_artifact_file: name of the validation-data file inside the artifact
        mode: 'new' starts LoRA training from scratch; 'resume' finds the most
            recent run under s3_bucket/s3_key_prefix and continues training it
        run_name: W&B run name; if None, W&B auto-generates one (e.g. 'young-microwave-2')
    """
    # trl is only needed for actual training (not for resolving/downloading
    # the adapter via utils.load_model), so it's kept out of module-level
    # imports and pulled in here.
    from trl import SFTTrainer

    print("=" * 70)
    print("QLoRA Fine-Tuning: Qwen2.5-7B-Instruct for Research Abstract NER")
    print("=" * 70)

    # Initialize class weights (use defaults if not provided)
    if class_weights is None:
        class_weights = get_default_class_weights()
        print("\n✓ Using default inverse-frequency class weights")
    else:
        print(f"\n✓ Using custom class weights")
    
    print(f"  Class weights (top 5 by upweight factor):")
    for label, weight in sorted(class_weights.items(), key=lambda x: -x[1])[:5]:
        print(f"    {label}: {weight:.1f}x")

    # Initialize W&B
    print("\n[0/6] Initializing W&B...")
    run = wandb.init(
        project=wandb_project,
        name=run_name or None,  # empty string (e.g. blank CLI/launch-config input) -> auto-generate
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
            "class_weights": class_weights,
        }
    )
    
    # Download only the specific files we need from the dataset artifact
    # (not the whole artifact — some artifacts carry many unrelated files).
    # Qualify with dataset_wandb_project when the dataset artifact lives in a
    # different W&B project than this run logs to (e.g. a shared datasets
    # project); otherwise wandb.use_artifact() looks in this run's own project.
    dataset_artifact_ref = (
        f"{dataset_wandb_project}/{dataset_artifact}" if dataset_wandb_project else dataset_artifact
    )
    print(f"Downloading dataset artifact '{dataset_artifact_ref}' from W&B...")
    try:
        artifact = wandb.use_artifact(dataset_artifact_ref)
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
        print(f"✗ Could not download dataset artifact '{dataset_artifact_ref}': {e}")
        run.finish(exit_code=1)
        raise
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model in 4-bit
    print("\n[1/5] Loading model in 4-bit...")
    quantization_config = setup_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # Disable cache for training
    )
    print(f"✓ Model loaded: {model.num_parameters():,} parameters")
    
    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def formatting_func(example):
        # This trl version (ConstantLengthDataset) calls formatting_func per
        # example and appends the result into a buffer it later tokenizes, so
        # it must return a BARE STRING (returning a list nests it -> the
        # tokenizer gets list-of-lists and raises TextEncodeInput TypeError).
        # example["messages"] is one conversation (list of role/content dicts).
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,   # full turn incl. assistant, for training
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Setup LoRA
    print("\n[3/5] Setting up LoRA adapters...")
    if mode == "resume":
        # Shared with predict_ner.load_ft_ner_model: resolves the latest
        # trained adapter via its W&B artifact (rather than listing S3
        # ourselves) and downloads it.
        resumed_adapter_dir = output_dir / "resumed_adapter"
        adapter_path = load_latest_adapter(
            resumed_adapter_dir,
            wandb_project=wandb_project,
            artifact_name=WANDB_ADAPTER_ARTIFACT_NAME,
            wandb_run=run,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        print("=" * 70)
        print(f"✓ WEIGHTS LOADED FROM S3: {adapter_path}")
        print("=" * 70)
    else:
        lora_config = setup_lora_config(lora_rank=lora_rank, lora_alpha=lora_alpha)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\n[4/5] Loading training data...")
    train_dataset = load_data(train_file)
    val_dataset = load_data(val_file)
    
    # Training arguments
    print("\n[5/5] Setting up training...")
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
        save_total_limit=2,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if available
        gradient_checkpointing=True,  # big activation-memory savings for 7B on 24GB
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required with PEFT
        ddp_find_unused_parameters=False,
        report_to=["wandb", "tensorboard"],
        push_to_hub=False,
        run_name=f"qwen-ner-lora-rank{lora_rank}",
    )
    
    # Create weighted loss function for class balancing
    def compute_weighted_loss(logits, labels, class_weights_dict):
        """
        Compute weighted cross-entropy loss for NER token classification.
        
        For causal LM fine-tuning on chat format, labels are token IDs.
        We extract NER label tokens from the JSON output and weight by class.
        
        For simplicity: use standard CE loss but upweight samples that contain
        rare labels (IMPROVEMENT, OBJECTIVE, PROBLEM, DATASET).
        """
        # Rare label IDs: extract token IDs for JSON representations of these labels
        rare_labels = {"IMPROVEMENT", "OBJECTIVE", "PROBLEM", "DATASET"}
        
        # Get token IDs for rare label names in the tokenizer vocab
        rare_token_ids = set()
        for label in rare_labels:
            # Tokenize the label as it appears in JSON output
            token_ids = tokenizer.encode(f'"{label}"', add_special_tokens=False)
            rare_token_ids.update(token_ids)
        
        # Standard CE loss (fallback)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction='none'
        )
        
        # If the sample contains rare labels, upweight the loss
        sample_has_rare = (labels.view(-1).unsqueeze(0) == torch.tensor(list(rare_token_ids), device=labels.device).unsqueeze(1)).any(dim=0)
        weights = torch.where(sample_has_rare, torch.tensor(2.0, device=labels.device), torch.tensor(1.0, device=labels.device))
        
        weighted_loss = (ce_loss * weights).mean()
        return weighted_loss
    
    # Custom trainer class that applies class weighting
    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**{k: v for k, v in inputs.items() if k != 'class_weights'})
            loss = compute_weighted_loss(outputs.logits, inputs['labels'], class_weights)
            return (loss, outputs) if return_outputs else loss
    
    # Saves+uploads a new best train/eval checkpoint to S3 + W&B immediately
    # whenever loss improves, so a premature stop (crash, KeyboardInterrupt,
    # preemption) never loses progress.
    best_checkpoint_callback = BestCheckpointS3WandbCallback(
        tokenizer=tokenizer,
        s3_bucket=s3_bucket,
        s3_key_prefix_root=s3_key_prefix,
        wandb_run=run,
        adapter_artifact_name=WANDB_ADAPTER_ARTIFACT_NAME,
        artifact_metadata_extra={
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "dataset_artifact": dataset_artifact_ref,
        },
        tmp_root=str(output_dir / "_best_tmp"),
    )

    # Create trainer with weighted loss
    trainer = WeightedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        packing=True,  # Concatenate sequences for efficiency
        max_seq_length=max_seq_length,
        formatting_func=formatting_func,
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
    # Saved alongside the adapter (not output_dir directly) so it's included
    # in the S3 upload below, which only uploads output_dir/adapter_model/.
    tokenizer.save_pretrained(output_dir / "adapter_model")
    print(f"✓ Adapter saved to: {output_dir}/adapter_model")
    print(f"✓ Tokenizer saved to: {output_dir}/adapter_model")
    
    # Print adapter size
    adapter_size = sum(
        f.stat().st_size for f in (output_dir / "adapter_model").rglob("*") if f.is_file()
    ) / 1e6
    print(f"✓ Adapter size: {adapter_size:.1f} MB")
    
    # Upload to S3, into its own timestamped run folder so this (and every
    # future) run's weights land alongside, never overwriting, past runs.
    run_timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    run_s3_prefix = f"{s3_key_prefix.rstrip('/')}/{run_timestamp}"
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
    
    # Log adapter and S3 location as artifact to W&B
    print("\nRegistering model in W&B...")
    adapter_artifact = wandb.Artifact(
        name=WANDB_ADAPTER_ARTIFACT_NAME,
        type="model",
        description=f"QLoRA adapter (rank={lora_rank}) fine-tuned on NER dataset",
        metadata={
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "dataset_artifact": dataset_artifact_ref,
            "s3_uri": s3_uri if s3_uri else "upload-failed",
            "s3_bucket": s3_bucket,
            "s3_key_prefix": run_s3_prefix,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )
    
    # Add adapter files to artifact
    adapter_artifact.add_dir(str(output_dir / "adapter_model"))
    
    # Also add a file with S3 location for easy reference
    if s3_uri:
        s3_info_file = output_dir / "S3_LOCATION.txt"
        with open(s3_info_file, 'w') as f:
            f.write(f"S3 URI: {s3_uri}\n")
            f.write(f"Bucket: {s3_bucket}\n")
            f.write(f"Key Prefix: {run_s3_prefix}\n")
            f.write(f"Uploaded: {datetime.datetime.now().isoformat()}\n")
        adapter_artifact.add_file(str(s3_info_file))
    
    run.log_artifact(adapter_artifact)
    print(f"✓ Model registered in W&B with S3 location")
    
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
        print("3. Run inference:")
        print(f"   python3 inference_qlora.py --model {model_name} --adapter ./adapter-model")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for NER with W&B & S3")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--train-file", type=Path, default=Path("training_data_train.jsonl"), help="Local training file")
    parser.add_argument("--val-file", type=Path, default=Path("training_data_val.jsonl"), help="Local validation file")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen_ner"), help="Checkpoint directory")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT, help="W&B project this run logs to (the training run + adapter artifact)")
    parser.add_argument("--dataset-artifact", default=WANDB_DATASET_ARTIFACT, help="W&B dataset artifact reference (e.g. 'dataset-ner-3k:latest')")
    parser.add_argument("--dataset-wandb-project", default=WANDB_DATASET_PROJECT, help="W&B project the dataset artifact lives in, if different from --wandb-project (default: same project)")
    parser.add_argument("--train-artifact-file", default=WANDB_ARTIFACT_TRAIN_FILE, help="Training file name inside the W&B artifact")
    parser.add_argument("--val-artifact-file", default=WANDB_ARTIFACT_VAL_FILE, help="Validation file name inside the W&B artifact")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--mode", choices=["new", "resume"], default=TRAINING_MODE,
                         help="'new' trains from scratch; 'resume' loads adapter weights from S3 first")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--class-weights",
        type=json.loads,
        default=None,
        help="JSON dict of label->weight for class balancing. "
             'Example: \'{"IMPROVEMENT":21.7,"OBJECTIVE":13.7,"KEY_CONCEPT":1.0}\'. '
             "If not provided, uses default inverse-frequency weights."
    )
    
    args = parser.parse_args()

    if not S3_BUCKET:
        print("⚠ Warning: S3_BUCKET is not set (check your .env)")

    train_qlora(
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
        class_weights=args.class_weights,
    )
