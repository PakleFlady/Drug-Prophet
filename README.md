# Drug Prophet Baseline

Baseline work for the Kaggle competition **Mettler Toledo Industrial AI Challenge: Drug Prophet**.

The task is to forecast the final stable penicillin concentration from early-window fermentation time-series observations.

## Contents

- `scripts/baseline_ridge.py` - dependency-light baseline pipeline using pandas, NumPy, grouped validation, and dual Ridge regression.
- `baseline_notes.md` - modeling notes and current validation interpretation.
- `outputs/submission_baseline.csv` - current baseline submission file.

## Data

Competition data is not committed to this repository. Download the Kaggle data and place it at:

```text
mt-ai-innovation-challenge-drug-prophet/kaggle_upload/
```

Expected files:

- `train_series.csv`
- `train_labels.csv`
- `test_series.csv`
- `sample_submission.csv`

## Run

Use the Python environment available on your machine, then run:

```bash
python scripts/baseline_ridge.py
```

The script writes feature matrices, validation diagnostics, and a submission file under `outputs/`.

## Current Baseline

The selected candidate is a conservative blend:

```text
90% global median + 10% Ridge(spc_last_missing)
```

Grouped-by-batch cross-validation MAE:

```text
4.7433
```

This is a sanity baseline, not a final model. The validation setup is intentionally grouped by `batch_id` to avoid leakage across different windows from the same fermentation batch.
