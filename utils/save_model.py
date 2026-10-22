"""
Shared callback for continuously checkpointing the best train-loss and best
eval-loss adapters to S3 + W&B DURING training, not just after trainer.train()
returns.

Previously, training scripts (train_qlora.py, train_qlora_narrative.py) only
uploaded/registered the adapter once, at the very end of a successful run --
so a premature stop (KeyboardInterrupt, OOM, spot-instance preemption, crash)
meant losing all progress, as happened with the interrupted narrative run
recovered manually on 2026-09-18. BestCheckpointS3WandbCallback saves+uploads
a new best-so-far checkpoint immediately whenever train loss or eval loss
improves, so progress always survives, no matter how the run ends.

S3 layout:
  - "overwrite" kinds (train, by default -- it updates far more often than
    eval) use a FIXED path, re-uploaded in place each time, and W&B prunes
    every older version after logging a new one -- so both S3 and W&B only
    ever hold the single current best, no accumulation:
      {s3_key_prefix_root}/train_best/adapter_model/...
  - other kinds (eval, by default -- naturally much less frequent, once per
    eval) keep a full timestamped history in both S3 and W&B:
      {s3_key_prefix_root}/eval_best/<DDMMYY-HHMMSS>/adapter_model/...

W&B: two independently-versioned artifact collections, so ":latest" on each
always resolves to the best-so-far of that specific kind:
  {adapter_artifact_name}-train-best  (pruned to 1 version -- effectively overwritten)
  {adapter_artifact_name}-eval-best   (full version history kept)

Each W&B artifact is METADATA ONLY (s3_uri/s3_bucket/s3_key_prefix + the
training metrics) -- the adapter files themselves are never uploaded to W&B,
only to S3, to avoid duplicating (and paying to store) the weights twice.
"""

import datetime
import json
from pathlib import Path

import boto3
import wandb
from transformers import TrainerCallback

from utils.load_model import resolve_latest_adapter_artifact


def lookup_best_loss(wandb_project: str, artifact_name: str,
                      checkpoint_kind: str, metric_name: str) -> float:
    """
    Read the loss value recorded in an existing best-checkpoint artifact's
    metadata (e.g. {adapter_name}-train-best's "train_loss" field), to use as
    the BASELINE a resumed run's new checkpoints must beat -- rather than
    resetting to +inf and immediately treating the very first logged loss as
    "best" regardless of whether it's actually better than what's already
    saved. Returns +inf if no such artifact exists yet (e.g. resuming before
    any eval has happened), so the first genuine result is still accepted.

    Args:
        wandb_project: W&B project the artifact was logged to
        artifact_name: base adapter artifact name (e.g. 'qwen-narrative-adapter')
        checkpoint_kind: 'train-best' or 'eval-best'
        metric_name: metadata key to read ('train_loss' or 'eval_loss')
    """
    try:
        artifact = resolve_latest_adapter_artifact(
            wandb_project=wandb_project,
            artifact_name=artifact_name,
            checkpoint_kind=checkpoint_kind,
        )
        value = artifact.metadata.get(metric_name)
        return float(value) if value is not None else float("inf")
    except Exception:
        return float("inf")


def upload_dir_to_s3(local_dir: Path, s3_bucket: str, s3_key_prefix: str) -> str:
    """
    Upload all files under local_dir to S3, nested under s3_key_prefix such
    that files land at {s3_key_prefix}/{local_dir.name}/... -- the layout
    utils.load_model.load_latest_adapter() expects (and what train_qlora.py's
    own upload_to_s3 produces).

    Args:
        local_dir: local directory to upload (e.g. .../adapter_model)
        s3_bucket: S3 bucket name
        s3_key_prefix: destination prefix; local_dir's own name is appended

    Returns:
        S3 URI of the destination prefix
    """
    local_dir = Path(local_dir)
    s3_client = boto3.client("s3")
    uploaded = 0
    for local_file in local_dir.rglob("*"):
        if not local_file.is_file():
            continue
        relative_path = local_file.relative_to(local_dir.parent)
        s3_key = f"{s3_key_prefix.rstrip('/')}/{relative_path}".replace("\\", "/")
        s3_client.upload_file(str(local_file), s3_bucket, s3_key)
        uploaded += 1
    s3_uri = f"s3://{s3_bucket}/{s3_key_prefix}"
    print(f"  Uploaded {uploaded} files to {s3_uri}")
    return s3_uri


class BestCheckpointS3WandbCallback(TrainerCallback):
    """
    Whenever train loss or eval loss hits a new best, immediately:
      1. saves the adapter + tokenizer to a local tmp dir
      2. uploads it to S3 under train_best/ or eval_best/
      3. logs it as a new version of a dedicated W&B artifact collection

    So the best-so-far model is always safely persisted, independent of
    whether/how the run eventually finishes.
    """

    def __init__(self, tokenizer, s3_bucket: str, s3_key_prefix_root: str,
                 wandb_run, adapter_artifact_name: str,
                 artifact_metadata_extra: dict = None,
                 adapter_extra_files: dict = None,
                 tmp_root: str = "checkpoints/_best_tmp",
                 overwrite_kinds=frozenset({"train"}),
                 initial_best_train_loss: float = float("inf"),
                 initial_best_eval_loss: float = float("inf")):
        self.tokenizer = tokenizer
        self.s3_bucket = s3_bucket
        self.s3_key_prefix_root = s3_key_prefix_root.rstrip("/")
        self.wandb_run = wandb_run
        self.adapter_artifact_name = adapter_artifact_name
        self.artifact_metadata_extra = artifact_metadata_extra or {}
        # {filename: JSON-serializable obj} written next to the adapter weights
        # (so they ship to S3 with it) and attached to the W&B artifact -- e.g.
        # the context classifier's prompt.json, so the adapter is usable
        # without consulting W&B metadata.
        self.adapter_extra_files = adapter_extra_files or {}
        self.tmp_root = Path(tmp_root)
        self.overwrite_kinds = overwrite_kinds
        # For a fresh run these stay +inf (any first result counts as best).
        # For a resumed run, the caller passes in the loss already recorded
        # against the checkpoint being resumed from (see lookup_best_loss),
        # so only a genuine improvement over that baseline triggers a save.
        self.best_train_loss = initial_best_train_loss
        self.best_eval_loss = initial_best_eval_loss

    def _prune_old_versions(self, artifact_name: str, keep_version: str):
        """Delete every other version in this W&B artifact collection, so
        only keep_version (the one just logged) remains -- the net effect of
        "overwrite" using the standard artifact API (:latest still works via
        load_latest_adapter() exactly as before, since there's only ever one
        version present)."""
        try:
            api = wandb.Api()
            collection_path = f"{self.wandb_run.entity}/{self.wandb_run.project}/{artifact_name}"
            for art in api.artifacts("model", collection_path):
                if art.version != keep_version:
                    try:
                        art.delete()
                    except Exception as e:
                        print(f"    (could not delete old {artifact_name}:{art.version}: {e})")
        except Exception as e:
            print(f"    (could not prune old versions of {artifact_name}: {e})")

    def _save_and_upload(self, model, kind, metric_name, metric_value, state):
        # A transient S3/W&B hiccup here must not crash the training run --
        # that would throw away hours of GPU work over a checkpoint upload.
        try:
            overwrite = kind in self.overwrite_kinds
            adapter_dir = self.tmp_root / kind / "adapter_model"
            model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)
            for filename, content in self.adapter_extra_files.items():
                with open(adapter_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)

            if overwrite:
                # Fixed path, re-uploaded in place -- no timestamp, no accumulation.
                s3_prefix = f"{self.s3_key_prefix_root}/{kind}_best"
            else:
                timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
                s3_prefix = f"{self.s3_key_prefix_root}/{kind}_best/{timestamp}"
            s3_uri = upload_dir_to_s3(adapter_dir, self.s3_bucket, s3_prefix)

            artifact_name = f"{self.adapter_artifact_name}-{kind}-best"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=(
                    f"Best {kind} checkpoint so far ({metric_name}={metric_value:.4f}, "
                    f"step={state.global_step}, epoch={state.epoch:.2f})"
                ),
                metadata={
                    **self.artifact_metadata_extra,
                    "s3_uri": s3_uri,
                    "s3_bucket": self.s3_bucket,
                    "s3_key_prefix": s3_prefix,
                    "checkpoint_kind": kind,
                    metric_name: metric_value,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            )
            # Deliberately NOT artifact.add_dir(...) -- weights already live in
            # S3 (s3_uri above); this is a metadata-only pointer to that S3
            # location, not a second copy of the weights in W&B.
            #
            # A tiny pointer file IS attached (not the weights) because W&B's
            # "already exists with the same content, no new version created"
            # dedup check is based on the artifact's file manifest, not its
            # metadata dict -- with zero files, every save has an identical
            # (empty) manifest, so W&B silently reuses the same version
            # forever and the new metadata (loss/step/timestamp) never
            # actually gets persisted. Writing the metadata itself as a
            # few-hundred-byte JSON file gives each version genuinely
            # different content, so a new version is created every time.
            with artifact.new_file("pointer.json", mode="w") as f:
                json.dump(artifact.metadata, f, indent=2, default=str)
            for filename in self.adapter_extra_files:
                artifact.add_file(str(adapter_dir / filename), name=filename)
            self.wandb_run.log_artifact(artifact)
            print(f"✓ New best {kind} ({metric_name}={metric_value:.4f}) -> {s3_uri}  (W&B: {artifact_name}, metadata only)")

            if overwrite:
                # Wait for the version to be assigned server-side, then prune
                # every other version so this collection stays at exactly 1.
                artifact.wait()
                self._prune_old_versions(artifact_name, artifact.version)
        except Exception as e:
            print(f"✗ Failed to save/upload best-{kind} checkpoint ({metric_name}={metric_value:.4f}): {e}")
            print("  Continuing training -- this checkpoint just won't be persisted.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Training-step logs carry "loss"; eval logs carry "eval_loss" instead
        # (handled separately in on_evaluate), so this only fires for train loss.
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return
        train_loss = logs["loss"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            model = kwargs.get("model")
            if model is not None:
                self._save_and_upload(model, "train", "train_loss", train_loss, state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero or not metrics or "eval_loss" not in metrics:
            return
        eval_loss = metrics["eval_loss"]
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            model = kwargs.get("model")
            if model is not None:
                self._save_and_upload(model, "eval", "eval_loss", eval_loss, state)
