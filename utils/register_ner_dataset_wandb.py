"""
Register a synthetic-dataset directory (NER SFT, narrative, or Layer-2
citation-relation) as a W&B dataset artifact, so training scripts can pull it
by name/version instead of a local path.

Every file in --input_dir is added to the artifact as-is.

Optionally, pass --prompt_version (+ --prompt_registry) to embed the exact
data-generator prompt used to build the dataset in the artifact's metadata --
the full VERSIONS entry from utils/prompts.py (system prompt, user template,
label sets), so the artifact is self-describing even if the prompt is edited
or a new version is added later.

Usage
-----
    python -m utils.register_ner_dataset_wandb \
        --wandb-project my-ner-project \
        --dataset-artifact dataset-ner-3k \
        --description "9-label NER SFT set, NER_USER_TMPL prompt" \
        --input_dir data/dataset-ner-synthetic-dgft-6000

With the data-generator prompt embedded in metadata (e.g. a Layer-2 v2
dataset built with --version v2):
    python -m utils.register_ner_dataset_wandb \
        --wandb-project my-project \
        --dataset-artifact dataset-layer2-v2 \
        --description "Layer-2 citation relation v2: + FUTURE function" \
        --input_dir data/dataset-layer2/citation_contexts_layer2_v2 \
        --prompt_registry LAYER_2_VERSIONS --prompt_version v2
"""

import argparse
import os

import wandb

import utils.prompts as prompts_module


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
    ap.add_argument("--prompt_registry", default="LAYER_2_VERSIONS",
                     choices=["LAYER_2_VERSIONS", "NARRATIVE_VERSIONS", "CITATION_CLASSIFIER_SLM_VERSIONS"],
                     help="Which utils.prompts VERSIONS registry --prompt_version is a key "
                          "into (default: LAYER_2_VERSIONS). Only used if --prompt_version is set.")
    ap.add_argument("--prompt_version", default=None,
                     help="Version key (e.g. 'v1', 'v2') to look up in --prompt_registry and "
                          "embed verbatim (system prompt, user template, label sets) in the "
                          "artifact's metadata under 'prompt'. Default: not embedded.")
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

    metadata = {"source_dir": os.path.abspath(args.input_dir), "files": files}
    if args.prompt_version:
        registry = getattr(prompts_module, args.prompt_registry)
        if args.prompt_version not in registry:
            raise SystemExit(f"{args.prompt_version!r} not in utils.prompts.{args.prompt_registry} "
                              f"(available: {sorted(registry)})")
        version_cfg = registry[args.prompt_version]
        metadata["prompt"] = {
            "registry": args.prompt_registry,
            "version": args.prompt_version,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in version_cfg.items()},
        }
        print(f"Embedding utils.prompts.{args.prompt_registry}[{args.prompt_version!r}] "
              f"in artifact metadata.")

    run = wandb.init(project=args.wandb_project, job_type="register-dataset")
    artifact = wandb.Artifact(
        name=args.dataset_artifact,
        type="dataset",
        description=args.description,
        metadata=metadata,
    )
    artifact.add_dir(args.input_dir)
    run.log_artifact(artifact)
    artifact.wait()  # block until the upload commits, so .version below is populated
    run.finish()

    ref = f"{args.dataset_artifact}:{artifact.version}"
    print(f"\n✓ Registered dataset artifact '{ref}' in project '{args.wandb_project}'")


if __name__ == "__main__":
    main()
