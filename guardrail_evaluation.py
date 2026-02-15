"""Per-category guardrail evaluation for the salary prediction model.

Runs cross-validation and computes R2 scores and predicted vs actual salary
comparisons broken down by each categorical feature value. Flags categories
that fall below configurable thresholds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.preprocessing import prepare_features, reduce_cardinality


CATEGORICAL_FEATURES = ["Country", "EdLevel", "DevType", "Industry", "Age", "ICorPM"]


def load_and_preprocess(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data and apply same preprocessing as train.py.

    Returns:
        (df, X, y) where df has original categorical columns (after cardinality
        reduction), X is one-hot encoded features, y is the target.
    """
    data_path = Path("data/survey_results_public.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(
        data_path,
        usecols=[
            "Country",
            "YearsCode",
            "WorkExp",
            "EdLevel",
            "DevType",
            "Industry",
            "Age",
            "ICorPM",
            "ConvertedCompYearly",
        ],
    )

    main_label = "ConvertedCompYearly"
    min_salary = config["data"]["min_salary"]
    df = df[df[main_label] > min_salary]

    # Per-country percentile outlier removal
    lower_pct = config["data"]["lower_percentile"] / 100
    upper_pct = config["data"]["upper_percentile"] / 100
    lower_bound = df.groupby("Country")[main_label].transform("quantile", lower_pct)
    upper_bound = df.groupby("Country")[main_label].transform("quantile", upper_pct)
    df = df[(df[main_label] > lower_bound) & (df[main_label] < upper_bound)]

    df = df.dropna(subset=[main_label])

    # Cardinality reduction (same as train.py)
    for col in CATEGORICAL_FEATURES:
        df[col] = reduce_cardinality(df[col])

    # Drop rows with "Other" in specified features (same as train.py)
    other_name = config["features"]["cardinality"].get("other_category", "Other")
    drop_other_from = config["features"]["cardinality"].get("drop_other_from", [])
    if drop_other_from:
        before_drop = len(df)
        for col in drop_other_from:
            df = df[df[col] != other_name]
        print(
            f"Dropped {before_drop - len(df):,} rows with '{other_name}' in {drop_other_from}"
        )

    X = prepare_features(df)
    y = df[main_label]

    return df, X, y


def run_cv_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> np.ndarray:
    """Run KFold CV and return out-of-fold predictions for every row.

    Each row gets exactly one prediction (from the fold where it was in the
    test set).
    """
    n_splits = config["data"].get("cv_splits", 5)
    random_state = config["data"]["random_state"]
    model_config = config["model"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.empty(len(y))
    oof_predictions[:] = np.nan

    print(f"Running {n_splits}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=model_config["n_estimators"],
            learning_rate=model_config["learning_rate"],
            max_depth=model_config["max_depth"],
            min_child_weight=model_config["min_child_weight"],
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
            early_stopping_rounds=model_config["early_stopping_rounds"],
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        test_r2 = model.score(X_test, y_test)
        print(
            f"  Fold {fold}: Test R2 = {test_r2:.4f} (best iter: {model.best_iteration + 1})"
        )

        oof_predictions[test_idx] = model.predict(X_test)

    overall_r2 = r2_score(y, oof_predictions)
    print(f"\nOverall OOF R2: {overall_r2:.4f}")

    return oof_predictions


def compute_category_metrics(
    df: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    feature: str,
) -> pd.DataFrame:
    """Compute per-category R2, mean actual/predicted, and abs % diff."""
    results = []
    categories = df[feature].values
    actuals = y.values

    for cat in sorted(df[feature].unique()):
        mask = categories == cat
        cat_actual = actuals[mask]
        cat_pred = predictions[mask]
        count = int(mask.sum())

        if count < 2:
            cat_r2 = float("nan")
        else:
            cat_r2 = r2_score(cat_actual, cat_pred)

        mean_actual = cat_actual.mean()
        mean_pred = cat_pred.mean()
        abs_pct_diff = abs(mean_pred - mean_actual) / mean_actual * 100

        results.append(
            {
                "Category": cat,
                "Count": count,
                "R2": cat_r2,
                "Mean Actual ($)": mean_actual,
                "Mean Predicted ($)": mean_pred,
                "Abs % Diff": abs_pct_diff,
            }
        )

    return pd.DataFrame(results)


def format_table(metrics_df: pd.DataFrame) -> str:
    """Format metrics DataFrame as a markdown table."""
    lines = []
    header = (
        "| Category | Count | R2 | Mean Actual ($) | Mean Predicted ($) | Abs % Diff |"
    )
    sep = (
        "|----------|------:|----:|----------------:|-------------------:|-----------:|"
    )
    lines.append(header)
    lines.append(sep)

    for _, row in metrics_df.iterrows():
        r2_str = f"{row['R2']:.2f}" if not np.isnan(row["R2"]) else "N/A"
        lines.append(
            f"| {row['Category'][:45]:45s} | {row['Count']:5,d} | {r2_str:>4s} "
            f"| {row['Mean Actual ($)']:>15,.0f} | {row['Mean Predicted ($)']:>18,.0f} "
            f"| {row['Abs % Diff']:>9.1f}% |"
        )

    return "\n".join(lines)


def main():
    """Run per-category guardrail evaluation."""
    config_path = Path("config/model_parameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    guardrails = config.get("guardrails", {})
    min_r2 = guardrails.get("min_r2_per_category", 0.30)
    max_pct_diff = guardrails.get("max_abs_pct_diff", 10)

    print("=" * 80)
    print("GUARDRAIL EVALUATION - Per-Category Model Quality")
    print(f"Thresholds: min R2 = {min_r2}, max abs % diff = {max_pct_diff}%")
    print("=" * 80)

    df, X, y = load_and_preprocess(config)
    print(f"Dataset: {len(df):,} rows, {X.shape[1]} features\n")

    predictions = run_cv_predictions(X, y, config)

    # Reset index alignment: df and y may have non-contiguous indices
    # predictions array is positional, so align everything by position
    df_eval = df.reset_index(drop=True)
    y_eval = y.reset_index(drop=True)

    warnings = []

    for feature in CATEGORICAL_FEATURES:
        print(f"\n## {feature}\n")
        metrics = compute_category_metrics(df_eval, y_eval, predictions, feature)
        print(format_table(metrics))

        # Check guardrails
        for _, row in metrics.iterrows():
            cat = row["Category"]
            if not np.isnan(row["R2"]) and row["R2"] < min_r2:
                warnings.append(
                    f'{feature} "{cat}": R2 = {row["R2"]:.2f} (threshold: {min_r2})'
                )
            if row["Abs % Diff"] > max_pct_diff:
                warnings.append(
                    f'{feature} "{cat}": Abs % Diff = {row["Abs % Diff"]:.1f}% '
                    f"(threshold: {max_pct_diff}%)"
                )

    # Summary
    print("\n" + "=" * 80)
    if warnings:
        print("### Guardrail Warnings\n")
        for w in warnings:
            print(f"  - {w}")
        print(f"\n{len(warnings)} guardrail violation(s) found.")
    else:
        print("All categories pass guardrail thresholds.")

    print("=" * 80)

    sys.exit(1 if warnings else 0)


if __name__ == "__main__":
    main()
