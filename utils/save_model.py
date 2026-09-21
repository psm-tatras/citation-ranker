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

S3 layout (train and eval checkpoints kept in separate subtrees):
  {s3_key_prefix_root}/train_best/<DDMMYY-HHMMSS>/adapter_model/...
  {s3_key_prefix_root}/eval_best/<DDMMYY-HHMMSS>/adapter_model/...

W&B: two independently-versioned artifact collections, so ":latest" on each
always resolves to the best-so-far of that specific kind:
  {adapter_artifact_name}-train-best
  {adapter_artifact_name}-eval-best
"""

import datetime
from pathlib import Path

import boto3
import wandb
from transformers import TrainerCallback


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
                 tmp_root: str = "checkpoints/_best_tmp"):
        self.tokenizer = tokenizer
        self.s3_bucket = s3_bucket
        self.s3_key_prefix_root = s3_key_prefix_root.rstrip("/")
        self.wandb_run = wandb_run
        self.adapter_artifact_name = adapter_artifact_name
        self.artifact_metadata_extra = artifact_metadata_extra or {}
        self.tmp_root = Path(tmp_root)
        self.best_train_loss = float("inf")
        self.best_eval_loss = float("inf")

    def _save_and_upload(self, model, kind, metric_name, metric_value, state):
        # A transient S3/W&B hiccup here must not crash the training run --
        # that would throw away hours of GPU work over a checkpoint upload.
        try:
            adapter_dir = self.tmp_root / kind / "adapter_model"
            model.save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)

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
            artifact.add_dir(str(adapter_dir))
            self.wandb_run.log_artifact(artifact)
            print(f"✓ New best {kind} ({metric_name}={metric_value:.4f}) -> {s3_uri}  (W&B: {artifact_name})")
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
