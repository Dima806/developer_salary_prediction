"""Per-category guardrail evaluation for the salary prediction model.

Runs cross-validation and computes MAPE scores and predicted vs actual salary
comparisons broken down by each categorical feature value. Flags categories
that exceed configurable thresholds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.preprocessing import prepare_features
from src.train import (
    CATEGORICAL_FEATURES,
    apply_cardinality_reduction,
    drop_other_rows,
    filter_salaries,
    load_survey_data,
)


def load_and_preprocess(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data and apply same preprocessing as train.py.

    Returns:
        (df, X, y) where df has original categorical columns (after cardinality
        reduction), X is one-hot encoded features, y is the target.
    """
    try:
        df = load_survey_data(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Same filtering and cardinality steps as train.py
    main_label = "ConvertedCompYearly"
    df = filter_salaries(df, config)
    df = apply_cardinality_reduction(df)
    before_drop = len(df)
    df = drop_other_rows(df, config)
    print(f"Dropped {before_drop - len(df):,} rows with 'Other' values")

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

        oof_predictions[test_idx] = model.predict(X_test)
        fold_preds = oof_predictions[test_idx]
        test_mape = np.mean(np.abs((y_test - fold_preds) / y_test)) * 100
        best_iter = model.best_iteration + 1
        print(f"  Fold {fold}: Test MAPE = {test_mape:.2f}% (best iter: {best_iter})")

    overall_mape = np.mean(np.abs((y.values - oof_predictions) / y.values)) * 100
    print(f"\nOverall OOF MAPE: {overall_mape:.2f}%")

    return oof_predictions


def compute_category_metrics(
    df: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    feature: str,
) -> pd.DataFrame:
    """Compute per-category MAPE, mean actual/predicted, and abs % diff."""
    results = []
    categories = df[feature].values
    actuals = y.values

    for cat in sorted(df[feature].unique()):
        mask = categories == cat
        cat_actual = actuals[mask]
        cat_pred = predictions[mask]
        count = int(mask.sum())

        cat_mape = np.mean(np.abs((cat_actual - cat_pred) / cat_actual)) * 100

        mean_actual = cat_actual.mean()
        mean_pred = cat_pred.mean()
        abs_pct_diff = abs(mean_pred - mean_actual) / mean_actual * 100

        results.append(
            {
                "Category": cat,
                "Count": count,
                "MAPE (%)": cat_mape,
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
        "| Category | Count | MAPE (%) "
        "| Mean Actual ($) | Mean Predicted ($) | Abs % Diff |"
    )
    sep = (
        "|----------|------:|---------:"
        "|----------------:|-------------------:|-----------:|"
    )
    lines.append(header)
    lines.append(sep)

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['Category'][:45]:45s}"
            f" | {row['Count']:5,d}"
            f" | {row['MAPE (%)']:>7.1f}%"
            f" | {row['Mean Actual ($)']:>15,.0f}"
            f" | {row['Mean Predicted ($)']:>18,.0f}"
            f" | {row['Abs % Diff']:>9.1f}% |"
        )

    return "\n".join(lines)


def check_guardrails(config: dict) -> bool:
    """Check all categories against guardrail thresholds.

    Runs cross-validation and checks per-category MAPE and abs % diff.
    Prints a summary and returns True if all categories pass.

    Args:
        config: Model configuration dict (from model_parameters.yaml).

    Returns:
        True if all categories pass, False if any violation found.
    """
    guardrails = config.get("guardrails", {})
    max_pct_diff = guardrails.get("max_abs_pct_diff", 20)

    df, X, y = load_and_preprocess(config)
    predictions = run_cv_predictions(X, y, config)

    df_eval = df.reset_index(drop=True)
    y_eval = y.reset_index(drop=True)

    violations = []
    for feature in CATEGORICAL_FEATURES:
        metrics = compute_category_metrics(df_eval, y_eval, predictions, feature)
        for _, row in metrics.iterrows():
            cat = row["Category"]
            if row["Abs % Diff"] > max_pct_diff:
                violations.append(
                    f'{feature} "{cat}": Abs % Diff = '
                    f"{row['Abs % Diff']:.1f}%"
                    f" (threshold: {max_pct_diff}%)"
                )

    if violations:
        print(f"Guardrail check FAILED: {len(violations)} violation(s)")
        for v in violations:
            print(f"  - {v}")
        return False

    print("Guardrail check passed.")
    return True


def main():
    """Run per-category guardrail evaluation."""
    config_path = Path("config/model_parameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    guardrails = config.get("guardrails", {})
    max_pct_diff = guardrails.get("max_abs_pct_diff", 20)

    print("=" * 80)
    print("GUARDRAIL EVALUATION - Per-Category Model Quality")
    print(f"Threshold: max abs % diff = {max_pct_diff}%")
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
            if row["Abs % Diff"] > max_pct_diff:
                warnings.append(
                    f'{feature} "{cat}": Abs % Diff = '
                    f"{row['Abs % Diff']:.1f}%"
                    f" (threshold: {max_pct_diff}%)"
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
