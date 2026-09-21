# W&B Dataset Registry Setup

## What This Does
Registers your NER dataset in W&B as a **versioned artifact**. This creates an audit trail:
- Which model trained on which dataset version
- Dataset lineage (metadata, file hashes, creation timestamp)
- Easy reference when you come back months later: "What exactly was in that dataset?"

## Free Tier Support
✓ **Yes, fully supported on free plan.** W&B Artifacts are free. You get:
- Unlimited artifact versions
- Full lineage tracking
- Integration with training runs
- Web dashboard to explore datasets

## Setup (One-Time)

### 1. Install wandb
```bash
pip install wandb
```

### 2. Authenticate
```bash
wandb login
# Paste your API key when prompted (from https://wandb.ai/settings/keys)
```

## Usage

### Register Your Dataset
```bash
python register_ner_dataset_wandb.py \
  --data-dir "/Work/Experiments/citation-processor-ranker/data/SEKU/processed/ner_data_3k" \
  --project "PSM-IISER-NER"
```

**Expected output:**
```
Initializing W&B (project: my-ner-project)
Analyzing dataset...
Creating artifact: dataset-ner-3k
Adding files to artifact...
Logging artifact to W&B...
Logging statistics...

======================================================================
✓ Dataset registered successfully!
  Artifact URI: username/my-ner-project/dataset-ner-3k:v0
  Records: 2952
  Total Entities: 59196
  Size: 22.09 MB
======================================================================

Next step: Link this artifact to training runs:
  artifact = wandb.use_artifact('username/my-ner-project/dataset-ner-3k:v0')
======================================================================
```

## What Gets Uploaded

### Files in Artifact
- `ner_spans.jsonl` – char-offset format (22 MB)
- `ner_bio.jsonl` – BIO token format (13 MB)
- `ner_labels.json` – label mapping (1.1 KB)
- `DATASET_CARD.md` – human-readable summary

### Metadata Logged
```
{
  "records": 2952,
  "total_entities": 59196,
  "num_labels": 9,
  "labels": ["CONTRIBUTION", "DATASET", "FINDING", "IMPROVEMENT", "KEY_CONCEPT", "METHOD", "METRIC", "OBJECTIVE", "PROBLEM"],
  "source": "DeepSeek LLM",
  "created": "20250908_120000"
}
```

## Using the Dataset in Training Code

### Link Artifact to Training Run
```python
import wandb

# Initialize your training run
run = wandb.init(project='my-ner-project', job_type='train')

# Link the dataset artifact
artifact = wandb.use_artifact('username/my-ner-project/dataset-ner-3k:v0')
data_dir = artifact.download()

# Load dataset
with open(f'{data_dir}/ner_bio.jsonl') as f:
    train_data = [json.loads(line) for line in f]

# Train model...
run.log({'accuracy': 0.78})
run.finish()
```

**Result in W&B:**
- Training run is now linked to dataset artifact
- You can see: "model trained on dataset-ner-3k:v0"
- If you change dataset → creates v1, v2, etc.
- Full lineage history visible on the web dashboard

## Versioning

Each time you run the script:
```
dataset-ner-3k:v0  (first registration)
dataset-ner-3k:v1  (second registration, if you change files)
dataset-ner-3k:v2  (etc.)
```

To register an updated dataset:
1. Edit your data files (e.g., add more records to `ner_spans.jsonl`)
2. Run the script again
3. W&B automatically creates a new version

To check versions:
```bash
wandb artifact list username/my-ner-project/dataset-ner-3k
```

## Exploring Your Dataset in W&B Web Dashboard

After registration:
1. Go to https://wandb.ai/username/my-ner-project
2. Click **Artifacts** tab → **dataset-ner-3k**
3. You'll see:
   - Label distribution chart
   - File sizes and metadata
   - Version history
   - Which runs used this artifact

## Free Tier Limits
- **Storage:** 100 GB total (plenty for NER datasets)
- **Artifact versions:** Unlimited
- **Runs:** Unlimited

Your 22 MB dataset is tiny relative to these limits.

## Next Steps

### Option 1: Log Dataset Splits
If you later create train/val/test splits:
```bash
# Create separate artifacts
python register_ner_dataset_wandb.py --data-dir ./ner_data_3k_train --project my-ner-project
python register_ner_dataset_wandb.py --data-dir ./ner_data_3k_val --project my-ner-project
```

### Option 2: Compare Multiple Dataset Versions
```
dataset-ner-3k:v0  (original, 2952 records)
dataset-ner-3k:v1  (after augmentation, 3500 records)
dataset-ner-3k:v2  (after cleaning, 2900 records)
```

View lineage on the dashboard → understand which model performed better on which version.

### Option 3: Add to README
Pin the artifact URI in your project README:
```
Training uses: `wandb.use_artifact('username/my-ner-project/dataset-ner-3k:v0')`
```

## Troubleshooting

### "No module named wandb"
```bash
pip install wandb --upgrade
```

### "Could not find files at path"
Verify data directory contains:
```bash
ls -la ./ner_data_3k/
# Should show:
# - ner_spans.jsonl
# - ner_bio.jsonl
# - ner_labels.json
```

### "401 Unauthorized"
Run `wandb login` again and paste fresh API key.

### "Project not found"
The project will be created automatically if it doesn't exist. Ensure you have permission in that workspace.

## For Collaboration

Share dataset with team:
1. Run the script (uploads artifact)
2. Share the URI with teammates: `username/my-ner-project/dataset-ner-3k:v0`
3. They can download: 
   ```python
   import wandb
   artifact = wandb.use_artifact('username/my-ner-project/dataset-ner-3k:v0')
   data_dir = artifact.download()
   ```

No need to share files via email or drive — W&B handles versioning and access control.
