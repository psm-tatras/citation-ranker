"""Resolve a fine-tuned adapter via its W&B artifact and download it from S3."""

import os
from pathlib import Path

import boto3
import wandb
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "my-ner-project")
WANDB_ADAPTER_ARTIFACT_NAME = os.getenv("WANDB_ADAPTER_ARTIFACT_NAME", "qwen-ner-adapter")


def download_from_s3(s3_bucket: str, s3_key_prefix: str, local_dir: Path) -> Path:
    """Download every file under an S3 prefix into local_dir, keeping relative paths.

    Returns:
        local_dir
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
                                     wandb_run=None,
                                     checkpoint_kind: str = None,
                                     version: str = "latest"):
    """Look up an adapter's W&B model artifact.

    Args:
        wandb_project: W&B project the artifact was logged to.
        artifact_name: Base artifact name, e.g. "qwen-ner-adapter".
        wandb_run: Active run to record the lookup on; None uses wandb.Api().
        checkpoint_kind: None for the final adapter, or "train-best"/"eval-best".
        version: Artifact version alias, e.g. "latest" or "v61".

    Returns:
        The W&B Artifact (metadata holds s3_bucket/s3_key_prefix).
    """
    name = f"{artifact_name}-{checkpoint_kind}" if checkpoint_kind else artifact_name
    reference = f"{name}:{version}"
    if wandb_run is not None:
        return wandb_run.use_artifact(reference)
    return wandb.Api().artifact(f"{wandb_project}/{reference}")


def load_latest_adapter(local_dir: Path,
                         wandb_project: str = WANDB_PROJECT,
                         artifact_name: str = WANDB_ADAPTER_ARTIFACT_NAME,
                         wandb_run=None,
                         checkpoint_kind: str = None,
                         version: str = "latest") -> Path:
    """Resolve an adapter's W&B artifact and download its files from S3.

    Args:
        local_dir: Directory to download the adapter into.
        wandb_project, artifact_name, wandb_run, checkpoint_kind, version:
            See resolve_latest_adapter_artifact.

    Returns:
        Path to local_dir/adapter_model.
    """
    resolved_name = f"{artifact_name}-{checkpoint_kind}" if checkpoint_kind else artifact_name
    print(f"Looking up '{resolved_name}:{version}' in W&B project '{wandb_project}'...")
    try:
        artifact = resolve_latest_adapter_artifact(wandb_project, artifact_name, wandb_run, checkpoint_kind, version)
    except Exception as e:
        raise RuntimeError(
            f"Could not find a '{resolved_name}:{version}' model artifact in "
            f"W&B project '{wandb_project}': {e}"
        )

    bucket = artifact.metadata.get("s3_bucket")
    key_prefix = artifact.metadata.get("s3_key_prefix")
    if not bucket or not key_prefix:
        raise ValueError(
            f"W&B artifact '{artifact.name}' is missing s3_bucket/s3_key_prefix "
            "metadata (its S3 upload may have failed)"
        )

    download_from_s3(bucket, key_prefix, local_dir)
    # The upload keeps the "adapter_model/" subfolder in the S3 key.
    adapter_path = Path(local_dir) / "adapter_model"
    print(f"✓ Loaded adapter '{artifact.name}' from s3://{bucket}/{key_prefix}")
    return adapter_path
