"""Data preprocessing utilities for consistent feature engineering."""

import pandas as pd


def reduce_cardinality(
    series: pd.Series,
    max_categories: int = 20,
    min_frequency: int = 50
) -> pd.Series:
    """
    Reduce cardinality by grouping rare categories into 'Other'.

    Args:
        series: Pandas Series with categorical values
        max_categories: Maximum number of categories to keep (default: 20)
        min_frequency: Minimum occurrences for a category to be kept (default: 50)

    Returns:
        Series with rare categories replaced by 'Other'
    """
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
        df: DataFrame with columns: Country, YearsCode (or YearsCodePro), EdLevel

    Returns:
        DataFrame with one-hot encoded features ready for model input

    Note:
        - Fills missing values with defaults (0 for numeric, "Unknown" for categorical)
        - Applies one-hot encoding with drop_first=True to avoid multicollinearity
        - Column names in output will be like: YearsCode, Country_X, EdLevel_Y
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Handle column name variations (YearsCode vs YearsCodePro)
    if "YearsCodePro" in df_processed.columns and "YearsCode" not in df_processed.columns:
        df_processed["YearsCode"] = df_processed["YearsCodePro"]

    # Fill missing values with defaults
    df_processed["YearsCode"] = df_processed["YearsCode"].fillna(0)
    df_processed["Country"] = df_processed["Country"].fillna("Unknown")
    df_processed["EdLevel"] = df_processed["EdLevel"].fillna("Unknown")

    # Reduce cardinality for categorical features
    # This groups rare categories into 'Other' to prevent overfitting
    df_processed["Country"] = reduce_cardinality(
        df_processed["Country"],
        max_categories=20,
        min_frequency=50
    )
    df_processed["EdLevel"] = reduce_cardinality(
        df_processed["EdLevel"],
        max_categories=20,
        min_frequency=50
    )

    # Select only the features we need
    feature_cols = ["Country", "YearsCode", "EdLevel"]
    df_features = df_processed[feature_cols]

    # Apply one-hot encoding for categorical variables
    # drop_first=True removes the first category to avoid multicollinearity
    df_encoded = pd.get_dummies(df_features, drop_first=True)

    return df_encoded
