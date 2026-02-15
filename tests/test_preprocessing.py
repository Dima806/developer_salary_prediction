"""Tests for src/preprocessing.py - Feature engineering utilities."""

import numpy as np
import pandas as pd

from src.preprocessing import (
    normalize_other_categories,
    prepare_features,
    reduce_cardinality,
)


class TestNormalizeOtherCategories:
    """Tests for normalize_other_categories()."""

    def test_replaces_other_please_specify(self):
        """'Other (please specify):' is replaced with 'Other'."""
        series = pd.Series(["Other (please specify):", "Developer, back-end"])
        result = normalize_other_categories(series)
        assert result.iloc[0] == "Other"
        assert result.iloc[1] == "Developer, back-end"

    def test_replaces_other_colon(self):
        """'Other:' is replaced with 'Other'."""
        series = pd.Series(["Other:", "Software Development"])
        result = normalize_other_categories(series)
        assert result.iloc[0] == "Other"

    def test_leaves_non_other_unchanged(self):
        """Non-Other values are not modified."""
        values = ["Developer, back-end", "Software Development", "India"]
        series = pd.Series(values)
        result = normalize_other_categories(series)
        assert list(result) == values

    def test_preserves_exact_other(self):
        """Exact 'Other' is kept as-is."""
        series = pd.Series(["Other"])
        result = normalize_other_categories(series)
        assert result.iloc[0] == "Other"


class TestReduceCardinality:
    """Tests for reduce_cardinality()."""

    def test_groups_rare_categories(self):
        """Rare categories are grouped into 'Other'."""
        # Create series with one dominant and many rare categories
        values = ["Common"] * 100 + ["Rare1", "Rare2", "Rare3"]
        series = pd.Series(values)
        result = reduce_cardinality(series, max_categories=5, min_frequency=10)
        assert "Common" in result.values
        assert "Rare1" not in result.values
        assert (result == "Other").sum() == 3

    def test_keeps_frequent_categories(self):
        """Frequent categories are kept intact."""
        values = ["A"] * 100 + ["B"] * 80 + ["C"] * 60
        series = pd.Series(values)
        result = reduce_cardinality(series, max_categories=5, min_frequency=50)
        assert set(result.unique()) == {"A", "B", "C"}


class TestPrepareFeatures:
    """Tests for prepare_features()."""

    def test_returns_dataframe_with_numeric_columns(self):
        """Output contains YearsCode and WorkExp as numeric columns."""
        df = pd.DataFrame(
            {
                "Country": ["India"],
                "YearsCode": [5.0],
                "WorkExp": [3.0],
                "EdLevel": ["Other"],
                "DevType": ["Developer, back-end"],
                "Industry": ["Software Development"],
                "Age": ["25-34 years old"],
                "ICorPM": ["Individual contributor"],
            }
        )
        result = prepare_features(df)
        assert "YearsCode" in result.columns
        assert "WorkExp" in result.columns

    def test_fills_missing_numeric_with_zero(self):
        """Missing numeric values are filled with 0."""
        df = pd.DataFrame(
            {
                "Country": ["India"],
                "YearsCode": [np.nan],
                "WorkExp": [np.nan],
                "EdLevel": ["Other"],
                "DevType": ["Developer, back-end"],
                "Industry": ["Software Development"],
                "Age": ["25-34 years old"],
                "ICorPM": ["Individual contributor"],
            }
        )
        result = prepare_features(df)
        assert result["YearsCode"].iloc[0] == 0.0
        assert result["WorkExp"].iloc[0] == 0.0

    def test_one_hot_encodes_categorical_columns(self):
        """Categorical columns are one-hot encoded."""
        df = pd.DataFrame(
            {
                "Country": ["India", "Germany"],
                "YearsCode": [5.0, 10.0],
                "WorkExp": [3.0, 8.0],
                "EdLevel": ["Other", "Other"],
                "DevType": ["Developer, back-end", "Developer, front-end"],
                "Industry": ["Software Development", "Healthcare"],
                "Age": ["25-34 years old", "35-44 years old"],
                "ICorPM": ["Individual contributor", "People manager"],
            }
        )
        result = prepare_features(df)
        # Should have one-hot columns for categorical features
        categorical_cols = [
            c for c in result.columns if "_" in c and c not in ("YearsCode", "WorkExp")
        ]
        assert len(categorical_cols) > 0

    def test_does_not_modify_original(self):
        """prepare_features does not modify the input DataFrame."""
        df = pd.DataFrame(
            {
                "Country": ["India"],
                "YearsCode": [5.0],
                "WorkExp": [3.0],
                "EdLevel": ["Other"],
                "DevType": ["Developer, back-end"],
                "Industry": ["Software Development"],
                "Age": ["25-34 years old"],
                "ICorPM": ["Individual contributor"],
            }
        )
        original_country = df["Country"].iloc[0]
        prepare_features(df)
        assert df["Country"].iloc[0] == original_country
