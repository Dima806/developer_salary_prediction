"""Data preprocessing utilities for consistent feature engineering."""

from pathlib import Path
import pandas as pd
import yaml

# Load configuration once at module level
_config_path = Path("config/model_parameters.yaml")
with open(_config_path, "r") as f:
    _config = yaml.safe_load(f)


def _get_other_category() -> str:
    """Get the standard 'Other' category name from config."""
    return _config["features"]["cardinality"].get("other_category", "Other")


def normalize_other_categories(series: pd.Series) -> pd.Series:
    """
    Normalize variants of 'Other' to the standard category name.

    Replaces values like 'Other (please specify):', 'Other:', etc.
    with the standard 'Other' category from config.
    """
    other_name = _get_other_category()
    return series.replace(
        to_replace=r"^Other\b.*$",
        value=other_name,
        regex=True,
    )


def reduce_cardinality(
    series: pd.Series, max_categories: int = None, min_frequency: int = None
) -> pd.Series:
    """
    Reduce cardinality by grouping rare categories into 'Other'.

    Args:
        series: Pandas Series with categorical values
        max_categories: Maximum number of categories to keep
                       (default: from config)
        min_frequency: Minimum occurrences for a category to be kept
                      (default: from config)

    Returns:
        Series with rare categories replaced by 'Other'
    """
    other_name = _get_other_category()

    # Use config defaults if not provided
    if max_categories is None:
        max_categories = _config["features"]["cardinality"]["max_categories"]
    if min_frequency is None:
        min_frequency = _config["features"]["cardinality"]["min_frequency"]

    # Normalize "Other" variants before counting frequencies
    series = normalize_other_categories(series)

    # Count value frequencies
    value_counts = series.value_counts()

    # Keep only categories that meet both criteria:
    # 1. In top max_categories by frequency
    # 2. Have at least min_frequency occurrences
    top_categories = value_counts.head(max_categories)
    kept_categories = top_categories[top_categories >= min_frequency].index.tolist()

    # Replace rare categories with the standard 'Other' name
    return series.apply(lambda x: x if x in kept_categories else other_name)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent feature transformations for both training and inference.

    This function ensures that the same preprocessing steps are applied
    during training and inference, preventing data leakage and inconsistencies.

    Args:
        df: DataFrame with columns: Country, YearsCode, WorkExp, EdLevel, DevType, Industry, Age, ICorPM
            NOTE: During training, cardinality reduction should be applied to df
            BEFORE calling this function. During inference, valid_categories.yaml
            ensures only valid (already-reduced) categories are used.

    Returns:
        DataFrame with one-hot encoded features ready for model input

    Note:
        - Fills missing values with defaults (0 for numeric, "Unknown" for categorical)
        - Normalizes Unicode apostrophes to regular apostrophes
        - Applies one-hot encoding with drop_first=True to avoid multicollinearity
        - Column names in output will be like: YearsCode, WorkExp, Country_X, EdLevel_Y, DevType_Z, Industry_W, Age_V, ICorPM_U
        - Does NOT apply cardinality reduction (must be done before calling this)
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Normalize Unicode apostrophes to regular apostrophes for consistency
    # This handles cases where data has \u2019 (') instead of '
    for col in ["Country", "EdLevel", "DevType", "Industry", "Age", "ICorPM"]:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.replace(
                "\u2019", "'", regex=False
            )

    # Normalize "Other" category variants (e.g. "Other (please specify):" -> "Other")
    for col in ["Country", "EdLevel", "DevType", "Industry", "Age", "ICorPM"]:
        if col in df_processed.columns:
            df_processed[col] = normalize_other_categories(df_processed[col])

    # Handle legacy column name (YearsCodePro -> YearsCode)
    if (
        "YearsCodePro" in df_processed.columns
        and "YearsCode" not in df_processed.columns
    ):
        df_processed.rename(columns={"YearsCodePro": "YearsCode"}, inplace=True)

    # Fill missing values with defaults
    df_processed["YearsCode"] = df_processed["YearsCode"].fillna(0)
    df_processed["WorkExp"] = df_processed["WorkExp"].fillna(0)
    df_processed["Country"] = df_processed["Country"].fillna("Unknown")
    df_processed["EdLevel"] = df_processed["EdLevel"].fillna("Unknown")
    df_processed["DevType"] = df_processed["DevType"].fillna("Unknown")
    df_processed["Industry"] = df_processed["Industry"].fillna("Unknown")
    df_processed["Age"] = df_processed["Age"].fillna("Unknown")
    df_processed["ICorPM"] = df_processed["ICorPM"].fillna("Unknown")

    # NOTE: Cardinality reduction is NOT applied here
    # It should be applied during training BEFORE calling this function
    # During inference, valid_categories.yaml ensures only valid values are used

    # Select only the features we need
    feature_cols = [
        "Country",
        "YearsCode",
        "WorkExp",
        "EdLevel",
        "DevType",
        "Industry",
        "Age",
        "ICorPM",
    ]
    df_features = df_processed[feature_cols]

    # Apply one-hot encoding for categorical variables
    # For inference (single rows), we need drop_first=False to create columns
    # The reindex in infer.py will align with training columns
    # For training (many rows), we use the config value
    is_inference = len(df_features) == 1
    drop_first = (
        False if is_inference else _config["features"]["encoding"]["drop_first"]
    )
    df_encoded = pd.get_dummies(df_features, drop_first=drop_first)

    return df_encoded
