"""
Register an NER SFT dataset directory (output of
training/convert_ner_format_to_qwen_sft.py) as a W&B dataset artifact, so
training/train_qlora.py can pull it by name/version instead of a local path.

Every file in --input_dir is added to the artifact as-is.

Usage
-----
    python -m utils.register_ner_dataset_wandb \
        --wandb-project my-ner-project \
        --dataset-artifact dataset-ner-3k \
        --description "9-label NER SFT set, NER_USER_TMPL prompt" \
        --input_dir data/dataset-ner-synthetic-dgft-6000
"""

import argparse
import os

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-project", required=True,
                     help="W&B project to register the dataset artifact in.")
    ap.add_argument("--dataset-artifact", required=True,
                     help="Dataset artifact name (e.g. 'dataset-ner-3k').")
    ap.add_argument("--description", required=True,
                     help="Description shown on the artifact's version page.")
    ap.add_argument("--input_dir", required=True,
                     help="Directory whose files are all added to the artifact.")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"{args.input_dir!r} is not a directory")
    files = sorted(f for f in os.listdir(args.input_dir)
                    if os.path.isfile(os.path.join(args.input_dir, f)))
    if not files:
        raise SystemExit(f"No files found in {args.input_dir!r}")
    print(f"Registering {len(files)} file(s) from {args.input_dir}:")
    for f in files:
        print(f"  {f}")

    run = wandb.init(project=args.wandb_project, job_type="register-dataset")
    artifact = wandb.Artifact(
        name=args.dataset_artifact,
        type="dataset",
        description=args.description,
        metadata={"source_dir": os.path.abspath(args.input_dir), "files": files},
    )
    artifact.add_dir(args.input_dir)
    run.log_artifact(artifact)
    artifact.wait()  # block until the upload commits, so .version below is populated
    run.finish()

    ref = f"{args.dataset_artifact}:{artifact.version}"
    print(f"\n✓ Registered dataset artifact '{ref}' in project '{args.wandb_project}'")


if __name__ == "__main__":
    main()
