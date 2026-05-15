# Drug Prophet Baseline Notes

## Problem framing

This is an early-window regression task. Each `sample_id` contains a variable-length time-series segment from one fermentation batch, and the target is the final stable penicillin concentration `target_stable_mean`.

The same batch appears as multiple observation windows in training (`5`, `10`, `20`, and `35` percent), so validation must be grouped by `batch_id`. Random sample-level splitting leaks batch information and gives an overly optimistic score.

## Current baseline

Script: `scripts/baseline_ridge.py`

The pipeline:

1. Read `train_series.csv`, `train_labels.csv`, `test_series.csv`, and `sample_submission.csv`.
2. Aggregate each time-series sample into one row of features:
   - metadata: control strategy, window size, step count, time span
   - process and spectrum summaries: mean, std, min, max, median, first, last, delta, range, missing fraction, slope
3. Evaluate with 5-fold batch-grouped cross validation.
4. Compare conservative candidates:
   - global median target
   - Ridge regression on different feature variants
   - small blends of the median and Ridge predictions
5. Train the selected candidate on all visible training data and write `outputs/submission_baseline.csv`.

## Selected candidate

The current selected candidate is:

`90% global median + 10% Ridge(spc_last_missing)`

Ridge uses the last observed PCA Raman features plus their missingness indicators, with compact metadata. This is deliberately conservative because there are only 45 independent training batches.

Current grouped CV MAE:

`4.7433`

The pure global-median baseline is close:

`4.7548`

That small gap means the current model signal is weak. Treat this as a sanity baseline, not a final solution.

## Next improvement directions

1. Build batch-level validation diagnostics and inspect the worst batches.
2. Add physically meaningful process features: oxygen transfer behavior, feed totals, pH/temperature stability, OUR/CER trajectory, and final-window deltas.
3. Separate feature sets for process variables and Raman variables, then blend only when validation supports it.
4. Try stronger tabular models if a compliant dependency is available locally: LightGBM, XGBoost, CatBoost, or scikit-learn models.
5. Prepare reproducible final package because technical review is 40% of the final score.
