"""Pre-process survey data and generate config artifacts.

Validates the raw CSV, applies the same data-cleaning steps used by
src/train.py, then writes:

- config/valid_categories.yaml  — valid input values for runtime guardrails
- config/currency_rates.yaml    — per-country median currency conversion rates

Run before ``make train`` to validate data and pre-generate configs, or
standalone to inspect what categories the current dataset supports.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.train import (
    CATEGORICAL_FEATURES,
    apply_cardinality_reduction,
    compute_currency_rates,
    drop_other_rows,
    extract_valid_categories,
    filter_salaries,
)

REQUIRED_COLUMNS = [
    "Country",
    "YearsCode",
    "WorkExp",
    "EdLevel",
    "DevType",
    "Industry",
    "Age",
    "ICorPM",
    "OrgSize",
    "Employment",
    "Currency",
    "CompTotal",
    "ConvertedCompYearly",
]


def validate_columns(data_path: Path) -> None:
    """Exit 1 if any required column is absent from the CSV header."""
    header = pd.read_csv(data_path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        print(f"Error: missing required columns: {missing}")
        sys.exit(1)
    print(f"All {len(REQUIRED_COLUMNS)} required columns present.")


def print_category_summary(df: pd.DataFrame) -> None:
    """Print the number of unique categories per categorical feature."""
    for col in CATEGORICAL_FEATURES:
        n = df[col].dropna().nunique()
        print(f"  {col}: {n} categories")


def main() -> None:
    """Validate data, apply preprocessing, and write config artifacts."""
    config_path = Path("config/model_parameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = Path("data/survey_results_public.csv")

    # Step 1 — Validate data file ------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Validate data file")
    print("=" * 60)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Download from: https://insights.stackoverflow.com/survey")
        sys.exit(1)

    print(f"Checking columns in {data_path} ...")
    validate_columns(data_path)

    # Step 2 — Load and filter salaries ------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Load and filter salaries")
    print("=" * 60)

    df = pd.read_csv(data_path, usecols=REQUIRED_COLUMNS)
    print(f"Loaded {len(df):,} rows")

    df = filter_salaries(df, config)
    print(f"After salary filtering: {len(df):,} rows")

    # Step 3 — Cardinality reduction ---------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Cardinality reduction")
    print("=" * 60)

    df = apply_cardinality_reduction(df)
    before = len(df)
    df = drop_other_rows(df, config)
    drop_cols = config["features"]["cardinality"].get("drop_other_from", [])
    if drop_cols:
        print(f"Dropped {before - len(df):,} rows with 'Other' in {drop_cols}")
    print(f"Final dataset: {len(df):,} rows")

    # Step 4 — Category summary --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Category summary")
    print("=" * 60)

    print_category_summary(df)

    # Step 5 — Write config artifacts --------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Write config artifacts")
    print("=" * 60)

    valid_categories = extract_valid_categories(df)
    vc_path = Path("config/valid_categories.yaml")
    with open(vc_path, "w") as f:
        yaml.dump(valid_categories, f, default_flow_style=False, sort_keys=False)
    n_total = sum(len(v) for v in valid_categories.values())
    print(f"Saved {vc_path} ({n_total} total valid values)")

    currency_rates = compute_currency_rates(df, valid_categories["Country"])
    cr_path = Path("config/currency_rates.yaml")
    with open(cr_path, "w") as f:
        yaml.dump(
            currency_rates,
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )
    print(f"Saved {cr_path} ({len(currency_rates)} countries)")

    print("\nPre-processing complete. Ready for `make train`.")


if __name__ == "__main__":
    main()
