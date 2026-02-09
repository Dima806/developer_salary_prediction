"""Data preprocessing utilities for consistent feature engineering."""

from pathlib import Path
import pandas as pd
import yaml

# Load configuration once at module level
_config_path = Path("config/model_parameters.yaml")
with open(_config_path, "r") as f:
    _config = yaml.safe_load(f)


def reduce_cardinality(
    series: pd.Series,
    max_categories: int = None,
    min_frequency: int = None
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
    # Use config defaults if not provided
    if max_categories is None:
        max_categories = _config['features']['cardinality']['max_categories']
    if min_frequency is None:
        min_frequency = _config['features']['cardinality']['min_frequency']

    # Count value frequencies
    value_counts = series.value_counts()

    # Keep only categories that meet both criteria:
    # 1. In top max_categories by frequency
    # 2. Have at least min_frequency occurrences
    top_categories = value_counts.head(max_categories)
    kept_categories = top_categories[top_categories >= min_frequency].index.tolist()

    # Replace rare categories with 'Other'
    return series.apply(lambda x: x if x in kept_categories else 'Other')


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent feature transformations for both training and inference.

    This function ensures that the same preprocessing steps are applied
    during training and inference, preventing data leakage and inconsistencies.

    Args:
        df: DataFrame with columns: Country, YearsCode (or YearsCodePro), EdLevel, DevType

    Returns:
        DataFrame with one-hot encoded features ready for model input

    Note:
        - Fills missing values with defaults (0 for numeric, "Unknown" for categorical)
        - Normalizes Unicode apostrophes to regular apostrophes
        - Applies one-hot encoding with drop_first=True to avoid multicollinearity
        - Column names in output will be like: YearsCode, Country_X, EdLevel_Y, DevType_Z
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Normalize Unicode apostrophes to regular apostrophes for consistency
    # This handles cases where data has \u2019 (') instead of '
    for col in ["Country", "EdLevel", "DevType"]:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].str.replace('\u2019', "'", regex=False)

    # Handle column name variations (YearsCode vs YearsCodePro)
    if "YearsCodePro" in df_processed.columns and "YearsCode" not in df_processed.columns:
        df_processed["YearsCode"] = df_processed["YearsCodePro"]

    # Fill missing values with defaults
    df_processed["YearsCode"] = df_processed["YearsCode"].fillna(0)
    df_processed["Country"] = df_processed["Country"].fillna("Unknown")
    df_processed["EdLevel"] = df_processed["EdLevel"].fillna("Unknown")
    df_processed["DevType"] = df_processed["DevType"].fillna("Unknown")

    # Reduce cardinality for categorical features
    # This groups rare categories into 'Other' to prevent overfitting
    # Uses config values from config/model_parameters.yaml
    df_processed["Country"] = reduce_cardinality(df_processed["Country"])
    df_processed["EdLevel"] = reduce_cardinality(df_processed["EdLevel"])
    df_processed["DevType"] = reduce_cardinality(df_processed["DevType"])

    # Select only the features we need
    feature_cols = ["Country", "YearsCode", "EdLevel", "DevType"]
    df_features = df_processed[feature_cols]

    # Apply one-hot encoding for categorical variables
    # drop_first removes the first category to avoid multicollinearity
    drop_first = _config['features']['encoding']['drop_first']
    df_encoded = pd.get_dummies(df_features, drop_first=drop_first)

    return df_encoded
