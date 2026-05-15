from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("mt-ai-innovation-challenge-drug-prophet") / "kaggle_upload"
DEFAULT_OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 20260516


def make_group_folds(groups: pd.Series, n_splits: int = 5, seed: int = RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold-like splits without depending on scikit-learn."""
    unique_groups = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    fold_groups = np.array_split(unique_groups, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    group_values = groups.to_numpy()
    for valid_groups in fold_groups:
        valid_mask = np.isin(group_values, valid_groups)
        valid_idx = np.flatnonzero(valid_mask)
        train_idx = np.flatnonzero(~valid_mask)
        folds.append((train_idx, valid_idx))
    return folds


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"sample_id", "control_type"}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _slope(values: pd.Series, times: pd.Series) -> float:
    mask = values.notna() & times.notna()
    if mask.sum() < 2:
        return np.nan
    x = times[mask].to_numpy(dtype=float)
    y = values[mask].to_numpy(dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def build_sample_features(series: pd.DataFrame) -> pd.DataFrame:
    """Collapse each variable-length early window into one row per sample_id."""
    feature_cols = numeric_feature_columns(series)
    stats_cols = [
        col
        for col in feature_cols
        if col not in {"batch_id", "control_type_id", "window_pct"}
    ]

    rows: list[dict[str, float | int | str]] = []
    grouped = series.sort_values(["sample_id", "step_idx"]).groupby("sample_id", sort=False)

    for sample_id, group in grouped:
        row: dict[str, float | int | str] = {"sample_id": sample_id}

        row["batch_id"] = int(group["batch_id"].iloc[0])
        row["control_type_id"] = int(group["control_type_id"].iloc[0])
        row["window_pct"] = int(group["window_pct"].iloc[0])
        row["n_steps"] = int(len(group))
        row["t_rel_max"] = float(group["t_rel"].max())
        row["time_h_max"] = float(group["Time..h."].max())

        control_type = str(group["control_type"].iloc[0])
        for value in ["recipe", "operator", "apc"]:
            row[f"control_type__{value}"] = 1.0 if control_type == value else 0.0

        times = group["t_rel"]
        for col in stats_cols:
            values = group[col]
            row[f"{col}__mean"] = float(values.mean(skipna=True))
            row[f"{col}__std"] = float(values.std(skipna=True))
            row[f"{col}__min"] = float(values.min(skipna=True))
            row[f"{col}__max"] = float(values.max(skipna=True))
            row[f"{col}__median"] = float(values.median(skipna=True))
            row[f"{col}__missing_frac"] = float(values.isna().mean())

            non_null = values.dropna()
            if len(non_null) > 0:
                first = float(non_null.iloc[0])
                last = float(non_null.iloc[-1])
                row[f"{col}__first"] = first
                row[f"{col}__last"] = last
                row[f"{col}__delta"] = last - first
                row[f"{col}__range"] = float(non_null.max() - non_null.min())
            else:
                row[f"{col}__first"] = np.nan
                row[f"{col}__last"] = np.nan
                row[f"{col}__delta"] = np.nan
                row[f"{col}__range"] = np.nan

            row[f"{col}__slope"] = _slope(values, times)

        rows.append(row)

    features = pd.DataFrame(rows)
    return features.sort_values("sample_id").reset_index(drop=True)


def prepare_matrix(
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    train_x = train_features[feature_cols].copy()
    valid_x = valid_features[feature_cols].copy()

    all_nan_cols = [col for col in feature_cols if train_x[col].isna().all()]
    kept_cols = [col for col in feature_cols if col not in all_nan_cols]
    train_x = train_x[kept_cols]
    valid_x = valid_x[kept_cols]

    medians = train_x.median(axis=0)
    train_x = train_x.fillna(medians).fillna(0.0)
    valid_x = valid_x.fillna(medians).fillna(0.0)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0).replace(0.0, 1.0).fillna(1.0)

    train_np = ((train_x - means) / stds).to_numpy(dtype=float)
    valid_np = ((valid_x - means) / stds).to_numpy(dtype=float)

    constant_cols = [col for col in kept_cols if stds[col] == 1.0 and train_x[col].nunique(dropna=False) <= 1]
    return train_np, valid_np, {"kept_cols": kept_cols, "all_nan_cols": all_nan_cols, "constant_cols": constant_cols}


def fit_ridge_dual(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y_mean = float(y.mean())
    y_centered = y - y_mean

    kernel = x @ x.T
    lhs = kernel + alpha * np.eye(kernel.shape[0])
    dual_coef = np.linalg.solve(lhs, y_centered)
    coef = x.T @ dual_coef
    return coef, y_mean


def predict_ridge(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return x @ coef + intercept


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_alphas(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
    alphas: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = features.merge(labels, on="sample_id", how="inner")
    folds = make_group_folds(train_df["batch_id"], n_splits=5)

    result_rows: list[dict[str, float | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for alpha in alphas:
        oof = np.full(len(train_df), np.nan)
        fold_scores = []
        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            fold_train = train_df.iloc[train_idx]
            fold_valid = train_df.iloc[valid_idx]
            x_train, x_valid, _ = prepare_matrix(fold_train, fold_valid, feature_cols)
            y_train = fold_train["target_stable_mean"].to_numpy(dtype=float)

            coef, intercept = fit_ridge_dual(x_train, y_train, alpha)
            pred = predict_ridge(x_valid, coef, intercept)
            oof[valid_idx] = pred

            fold_mae = mae(fold_valid["target_stable_mean"].to_numpy(dtype=float), pred)
            fold_scores.append(fold_mae)
            result_rows.append(
                {
                    "alpha": alpha,
                    "fold": fold_id,
                    "fold_mae": fold_mae,
                    "n_train": len(train_idx),
                    "n_valid": len(valid_idx),
                    "n_valid_batches": fold_valid["batch_id"].nunique(),
                }
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_id": train_df["sample_id"],
                    "batch_id": train_df["batch_id"],
                    "window_pct": train_df["window_pct"],
                    "target_stable_mean": train_df["target_stable_mean"],
                    "oof_pred": oof,
                    "alpha": alpha,
                }
            )
        )
        result_rows.append(
            {
                "alpha": alpha,
                "fold": 0,
                "fold_mae": mae(train_df["target_stable_mean"].to_numpy(dtype=float), oof),
                "n_train": len(train_df),
                "n_valid": len(train_df),
                "n_valid_batches": train_df["batch_id"].nunique(),
            }
        )

    return pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True)


def oof_global_median(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    train_df = features.merge(labels, on="sample_id", how="inner")
    folds = make_group_folds(train_df["batch_id"], n_splits=5)
    oof = np.full(len(train_df), np.nan)
    for train_idx, valid_idx in folds:
        oof[valid_idx] = float(train_df.iloc[train_idx]["target_stable_mean"].median())
    return pd.DataFrame(
        {
            "sample_id": train_df["sample_id"],
            "batch_id": train_df["batch_id"],
            "window_pct": train_df["window_pct"],
            "target_stable_mean": train_df["target_stable_mean"],
            "oof_pred": oof,
            "model": "global_median",
        }
    )


def ridge_submission_predictions(
    train_features: pd.DataFrame,
    labels: pd.DataFrame,
    test_features: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_cols: list[str],
    alpha: float,
) -> np.ndarray:
    train_df = train_features.merge(labels, on="sample_id", how="inner")
    requested_test = sample_submission[["sample_id"]].merge(test_features, on="sample_id", how="left")

    x_train, x_test, _ = prepare_matrix(train_df, requested_test, feature_cols)
    y_train = train_df["target_stable_mean"].to_numpy(dtype=float)
    coef, intercept = fit_ridge_dual(x_train, y_train, alpha)
    return predict_ridge(x_test, coef, intercept)


def train_and_predict(
    train_features: pd.DataFrame,
    labels: pd.DataFrame,
    test_features: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_cols: list[str],
    alpha: float,
    output_dir: Path,
) -> pd.DataFrame:
    train_df = train_features.merge(labels, on="sample_id", how="inner")
    requested_test = sample_submission[["sample_id"]].merge(test_features, on="sample_id", how="left")
    x_train, x_test, prep_info = prepare_matrix(train_df, requested_test, feature_cols)
    y_train = train_df["target_stable_mean"].to_numpy(dtype=float)
    coef, intercept = fit_ridge_dual(x_train, y_train, alpha)
    preds = predict_ridge(x_test, coef, intercept)

    submission = sample_submission.copy()
    submission["target_stable_mean"] = preds

    manifest = {
        "model": "dual_ridge_regression",
        "alpha": alpha,
        "n_train_samples": int(len(train_df)),
        "n_submission_samples": int(len(submission)),
        "n_input_features": int(len(feature_cols)),
        "n_kept_features_after_fold_preprocessing": int(len(prep_info["kept_cols"])),
        "n_all_nan_features_dropped": int(len(prep_info["all_nan_cols"])),
        "n_constant_features_detected": int(len(prep_info["constant_cols"])),
    }
    (output_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return submission


def build_feature_variants(feature_cols: list[str]) -> dict[str, list[str]]:
    compact_metadata = [
        "control_type_id",
        "window_pct",
        "n_steps",
        "t_rel_max",
        "time_h_max",
        "control_type__recipe",
        "control_type__operator",
        "control_type__apc",
    ]
    return {
        "full_stats": feature_cols,
        "spc_last_missing": [
            col
            for col in feature_cols
            if (col.startswith("spc_") and (col.endswith("__last") or col.endswith("__missing_frac")))
            or col in compact_metadata
        ],
        "last_delta_missing": [
            col
            for col in feature_cols
            if col.endswith("__last")
            or col.endswith("__delta")
            or col.endswith("__missing_frac")
            or col in compact_metadata
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_series = pd.read_csv(args.data_dir / "train_series.csv")
    train_labels = pd.read_csv(args.data_dir / "train_labels.csv")
    test_series = pd.read_csv(args.data_dir / "test_series.csv")
    sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")

    train_features = build_sample_features(train_series)
    test_features = build_sample_features(test_series)

    train_features.to_csv(output_dir / "train_sample_features.csv", index=False)
    test_features.to_csv(output_dir / "test_sample_features.csv", index=False)

    feature_cols = [
        col
        for col in train_features.columns
        if col not in {"sample_id", "batch_id"}
    ]

    alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    feature_variants = build_feature_variants(feature_cols)

    all_cv_results = []
    best_oof_by_variant = {}
    candidate_rows = []

    median_oof = oof_global_median(train_features, train_labels)
    median_mae = mae(
        median_oof["target_stable_mean"].to_numpy(dtype=float),
        median_oof["oof_pred"].to_numpy(dtype=float),
    )
    best_oof_by_variant["global_median"] = median_oof
    candidate_rows.append(
        {
            "candidate": "global_median",
            "feature_variant": "none",
            "alpha": np.nan,
            "ridge_weight": 0.0,
            "mae": median_mae,
            "n_features": 0,
        }
    )

    for variant_name, variant_cols in feature_variants.items():
        cv_results, oof_predictions = evaluate_alphas(train_features, train_labels, variant_cols, alphas)
        cv_results["feature_variant"] = variant_name
        all_cv_results.append(cv_results)

        overall = cv_results[cv_results["fold"] == 0].sort_values("fold_mae").reset_index(drop=True)
        best_alpha = float(overall.loc[0, "alpha"])
        best_mae = float(overall.loc[0, "fold_mae"])
        best_oof = oof_predictions[oof_predictions["alpha"] == best_alpha].copy()
        best_oof["model"] = f"ridge__{variant_name}"
        best_oof_by_variant[variant_name] = best_oof

        candidate_rows.append(
            {
                "candidate": f"ridge__{variant_name}",
                "feature_variant": variant_name,
                "alpha": best_alpha,
                "ridge_weight": 1.0,
                "mae": best_mae,
                "n_features": len(variant_cols),
            }
        )

        for ridge_weight in np.linspace(0.05, 0.50, 10):
            blended = best_oof.copy()
            blended["oof_pred"] = (
                ridge_weight * best_oof["oof_pred"].to_numpy(dtype=float)
                + (1.0 - ridge_weight) * median_oof["oof_pred"].to_numpy(dtype=float)
            )
            blended_mae = mae(
                blended["target_stable_mean"].to_numpy(dtype=float),
                blended["oof_pred"].to_numpy(dtype=float),
            )
            candidate_rows.append(
                {
                    "candidate": f"blend_median_ridge__{variant_name}",
                    "feature_variant": variant_name,
                    "alpha": best_alpha,
                    "ridge_weight": float(ridge_weight),
                    "mae": blended_mae,
                    "n_features": len(variant_cols),
                }
            )

    cv_table = pd.concat(all_cv_results, ignore_index=True)
    cv_table.to_csv(output_dir / "baseline_cv_results.csv", index=False)

    candidate_table = pd.DataFrame(candidate_rows).sort_values("mae").reset_index(drop=True)
    candidate_table.to_csv(output_dir / "baseline_candidate_results.csv", index=False)
    best_candidate = candidate_table.iloc[0]

    if best_candidate["candidate"] == "global_median":
        final_pred = np.full(
            len(sample_submission),
            float(train_labels["target_stable_mean"].median()),
        )
        final_oof = median_oof.copy()
    else:
        variant_name = str(best_candidate["feature_variant"])
        alpha = float(best_candidate["alpha"])
        ridge_weight = float(best_candidate["ridge_weight"])
        ridge_pred = ridge_submission_predictions(
            train_features,
            train_labels,
            test_features,
            sample_submission,
            feature_variants[variant_name],
            alpha,
        )
        median_pred = np.full(
            len(sample_submission),
            float(train_labels["target_stable_mean"].median()),
        )
        final_pred = ridge_weight * ridge_pred + (1.0 - ridge_weight) * median_pred

        final_oof = best_oof_by_variant[variant_name].copy()
        if ridge_weight < 1.0:
            final_oof["oof_pred"] = (
                ridge_weight * final_oof["oof_pred"].to_numpy(dtype=float)
                + (1.0 - ridge_weight) * median_oof["oof_pred"].to_numpy(dtype=float)
            )

    final_oof.to_csv(output_dir / "baseline_oof_predictions.csv", index=False)

    submission = sample_submission.copy()
    submission["target_stable_mean"] = final_pred
    submission.to_csv(output_dir / "submission_baseline.csv", index=False)

    window_mae = (
        final_oof.assign(abs_error=lambda df: (df["target_stable_mean"] - df["oof_pred"]).abs())
        .groupby("window_pct", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    window_mae.to_csv(output_dir / "baseline_window_mae.csv", index=False)

    manifest = {
        "selected_candidate": str(best_candidate["candidate"]),
        "selected_feature_variant": str(best_candidate["feature_variant"]),
        "selected_alpha": None if pd.isna(best_candidate["alpha"]) else float(best_candidate["alpha"]),
        "selected_ridge_weight": float(best_candidate["ridge_weight"]),
        "groupkfold_mae": float(best_candidate["mae"]),
        "n_train_samples": int(len(train_labels)),
        "n_submission_samples": int(len(submission)),
    }
    (output_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Selected candidate:", manifest["selected_candidate"])
    print("GroupKFold MAE:", manifest["groupkfold_mae"])
    print("\nCandidate search:")
    print(candidate_table.head(12).to_string(index=False))
    print("\nWindow MAE for selected candidate:")
    print(window_mae.to_string(index=False))
    print("\nWrote:", output_dir / "submission_baseline.csv")


if __name__ == "__main__":
    main()
