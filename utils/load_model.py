"""
Shared helper for locating and downloading the latest fine-tuned NER
adapter.

Every training run (training/train_qlora.py) uploads its adapter to its own
timestamped S3 folder and registers a W&B model artifact pointing at it. This
module is the single place that resolves "latest adapter" by asking W&B for
that artifact -- rather than each caller (training resume, inference) guessing
independently, e.g. by listing S3 -- and downloads it locally. Used by both
training/train_qlora.py (resume mode) and prediction/predict_ner.py
(load_ft_ner_model).
"""

import os
from pathlib import Path

import boto3
import wandb
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "my-ner-project")
WANDB_ADAPTER_ARTIFACT_NAME = os.getenv("WANDB_ADAPTER_ARTIFACT_NAME", "qwen-ner-adapter")


def download_from_s3(s3_bucket: str, s3_key_prefix: str, local_dir: Path) -> Path:
    """
    Download all files under an S3 key prefix into local_dir, preserving
    their relative paths (IAM role authentication).

    Args:
        s3_bucket: S3 bucket name
        s3_key_prefix: S3 key prefix the files were uploaded under
        local_dir: Local directory to download files into

    Returns:
        local_dir, populated with the downloaded files
    """
    print(f"\n[Download] Downloading from S3...")
    print(f"  Bucket: {s3_bucket}")
    print(f"  Prefix: {s3_key_prefix}")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client('s3')

    paginator = s3_client.get_paginator('list_objects_v2')
    downloaded_files = 0
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_key_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            relative_path = key[len(s3_key_prefix):].lstrip('/')
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading: {key} -> {local_path}")
            s3_client.download_file(s3_bucket, key, str(local_path))
            downloaded_files += 1

    if downloaded_files == 0:
        raise FileNotFoundError(f"No files found at s3://{s3_bucket}/{s3_key_prefix}")

    print(f"✓ Download complete: {downloaded_files} files")
    return local_dir


def resolve_latest_adapter_artifact(wandb_project: str = WANDB_PROJECT,
                                     artifact_name: str = WANDB_ADAPTER_ARTIFACT_NAME,
                                     wandb_run=None):
    """
    Look up the most recent trained-adapter W&B model artifact.

    Args:
        wandb_project: W&B project the artifact was logged to
        artifact_name: W&B artifact name (e.g. 'qwen-ner-adapter')
        wandb_run: an active wandb run (e.g. train_qlora's own `run`) to use
            for the lookup, so it's recorded as an input to that run. If
            None, uses the read-only wandb.Api() (no active run needed --
            this is the path predict_ner takes).

    Returns:
        The W&B Artifact object (its .metadata holds s3_bucket/s3_key_prefix)
    """
    reference = f"{artifact_name}:latest"
    if wandb_run is not None:
        return wandb_run.use_artifact(reference)
    return wandb.Api().artifact(f"{wandb_project}/{reference}")


def load_latest_adapter(local_dir: Path,
                         s3_bucket: str = None,
                         wandb_project: str = WANDB_PROJECT,
                         artifact_name: str = WANDB_ADAPTER_ARTIFACT_NAME,
                         wandb_run=None) -> Path:
    """
    Resolve the latest trained adapter via its W&B artifact and download it.

    Args:
        local_dir: directory to download the adapter into
        s3_bucket: overrides the bucket recorded in the artifact's metadata
        wandb_project, artifact_name, wandb_run: see resolve_latest_adapter_artifact

    Returns:
        Path to the downloaded adapter directory (local_dir/adapter_model),
        containing adapter_config.json / adapter weights.
    """
    print(f"Looking up '{artifact_name}:latest' in W&B project '{wandb_project}'...")
    try:
        artifact = resolve_latest_adapter_artifact(wandb_project, artifact_name, wandb_run)
    except Exception as e:
        raise RuntimeError(
            f"Could not find a previous '{artifact_name}' model artifact in "
            f"W&B project '{wandb_project}': {e}"
        )

    bucket = s3_bucket or artifact.metadata.get("s3_bucket")
    key_prefix = artifact.metadata.get("s3_key_prefix")
    if not bucket or not key_prefix:
        raise ValueError(
            f"W&B artifact '{artifact.name}' is missing s3_bucket/s3_key_prefix "
            "metadata (its S3 upload may have failed)"
        )

    download_from_s3(bucket, key_prefix, local_dir)
    # upload_to_s3 (train_qlora.py) preserves the local "adapter_model/"
    # subfolder in the S3 key, so the downloaded weights land one level
    # deeper than local_dir.
    adapter_path = Path(local_dir) / "adapter_model"
    print(f"✓ Loaded adapter '{artifact.name}' from s3://{bucket}/{key_prefix}")
    return adapter_path
