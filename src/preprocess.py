"""Pre-process survey data and generate config artifacts.

Validates the raw CSV, applies the same data-cleaning steps used by
src/train.py, then writes:

- config/valid_categories.yaml  — valid input values for runtime guardrails
- config/currency_rates.yaml    — 2025 per-country median currency rates
- config/currency_rates_2026.yaml — 2026 per-country median currency rates

Run before ``make train`` to validate data and pre-generate configs, or
standalone to inspect what categories the current dataset supports.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.train import (
    CATEGORICAL_FEATURES,
    REQUIRED_COLUMNS,
    apply_cardinality_reduction,
    drop_other_rows,
    extract_valid_categories,
    filter_salaries,
    load_survey_data,
    save_currency_rates,
)


def validate_columns(data_path: Path, renames: dict | None = None) -> None:
    """Exit 1 if any required column is absent from the CSV header.

    Args:
        data_path: Path to a survey CSV.
        renames: Optional {source_name: required_name} column renames.
    """
    header = pd.read_csv(data_path, nrows=0).rename(columns=renames or {})
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        print(f"Error: missing required columns in {data_path}: {missing}")
        sys.exit(1)
    print(f"All {len(REQUIRED_COLUMNS)} required columns present in {data_path}.")


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

    renames = config["data"].get("column_renames") or {}

    # Step 1 — Validate data files -----------------------------------------------
    print("=" * 60)
    print("STEP 1 — Validate data files")
    print("=" * 60)

    for year, path in config["data"]["sources"].items():
        data_path = Path(path)
        if not data_path.exists():
            print(f"Error: {data_path} not found.")
            print("Download from: https://survey.stackoverflow.co/")
            sys.exit(1)
        print(f"Checking columns in {data_path} ...")
        validate_columns(data_path, renames.get(str(year)))

    # Step 2 — Load and filter salaries ------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Load and filter salaries")
    print("=" * 60)

    df = load_survey_data(config)
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

    save_currency_rates(df, valid_categories["Country"], config)

    print("\nPre-processing complete. Ready for `make train`.")


if __name__ == "__main__":
    main()
