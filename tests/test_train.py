"""Tests for src/train.py - Training pipeline helper functions."""

import numpy as np
import pandas as pd

from src.train import (
    apply_cardinality_reduction,
    compute_currency_rates,
    drop_other_rows,
    extract_valid_categories,
    filter_salaries,
)


def _make_salary_df(countries=None, salaries=None, n=100) -> pd.DataFrame:
    """Create a minimal DataFrame resembling the survey data."""
    if salaries is not None:
        n = len(salaries)
    if countries is not None:
        n = len(countries)
    if countries is None:
        countries = ["United States of America"] * n
    if salaries is None:
        rng = np.random.default_rng(42)
        salaries = rng.integers(30000, 200000, size=n).astype(float)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Country": countries,
            "YearsCode": rng.integers(0, 30, size=n).astype(float),
            "WorkExp": rng.integers(0, 20, size=n).astype(float),
            "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"] * n,
            "DevType": ["Developer, full-stack"] * n,
            "Industry": ["Software Development"] * n,
            "Age": ["25-34 years old"] * n,
            "ICorPM": ["Individual contributor"] * n,
            "OrgSize": ["20 to 99 employees"] * n,
            "Currency": ["USD United States Dollar"] * n,
            "CompTotal": salaries,
            "ConvertedCompYearly": salaries,
        }
    )


class TestFilterSalaries:
    """Tests for filter_salaries()."""

    def test_removes_below_min_salary(self):
        """Rows with salary below min_salary are removed."""
        salaries = [500.0] * 5 + [2000.0] * 20 + [50000.0] * 20
        df = _make_salary_df(salaries=salaries)
        config = {
            "data": {
                "min_salary": 1000,
                "lower_percentile": 0,
                "upper_percentile": 100,
            }
        }
        result = filter_salaries(df, config)
        assert (result["ConvertedCompYearly"] > 1000).all()
        assert len(result) < len(df)

    def test_removes_outliers_by_percentile(self):
        """Per-country percentile outlier removal works."""
        salaries = [10000.0] * 50 + [500000.0] + [10000.0] * 49
        df = _make_salary_df(salaries=salaries)
        config = {
            "data": {
                "min_salary": 1000,
                "lower_percentile": 2,
                "upper_percentile": 98,
            }
        }
        result = filter_salaries(df, config)
        assert len(result) < len(df)

    def test_drops_missing_target(self):
        """Rows with NaN target are dropped."""
        df = _make_salary_df(salaries=[50000.0, np.nan, 60000.0])
        config = {
            "data": {
                "min_salary": 1000,
                "lower_percentile": 0,
                "upper_percentile": 100,
            }
        }
        result = filter_salaries(df, config)
        assert not result["ConvertedCompYearly"].isna().any()

    def test_returns_dataframe(self):
        """Returns a pandas DataFrame."""
        df = _make_salary_df()
        config = {
            "data": {
                "min_salary": 1000,
                "lower_percentile": 2,
                "upper_percentile": 98,
            }
        }
        result = filter_salaries(df, config)
        assert isinstance(result, pd.DataFrame)


class TestApplyCardinalityReduction:
    """Tests for apply_cardinality_reduction()."""

    def test_normalizes_unicode_apostrophes(self):
        """Unicode right single quotation marks are replaced."""
        df = _make_salary_df(n=100)
        df["EdLevel"] = "Master\u2019s degree"
        result = apply_cardinality_reduction(df)
        # Unicode apostrophe should be normalized to ASCII
        assert "\u2019" not in result["EdLevel"].iloc[0]

    def test_does_not_modify_original(self):
        """The input DataFrame is not modified."""
        df = _make_salary_df(n=5)
        original_country = df["Country"].iloc[0]
        apply_cardinality_reduction(df)
        assert df["Country"].iloc[0] == original_country

    def test_rare_categories_become_other(self):
        """Categories below min_frequency are grouped into 'Other'."""
        countries = ["United States of America"] * 100 + ["Narnia"] * 2
        df = _make_salary_df(countries=countries, n=102)
        result = apply_cardinality_reduction(df)
        assert "Narnia" not in result["Country"].values
        assert "Other" in result["Country"].values


class TestDropOtherRows:
    """Tests for drop_other_rows()."""

    def test_drops_other_from_specified_columns(self):
        """Rows with 'Other' in specified columns are dropped."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "Other", "Germany"],
                "DevType": ["Dev", "Dev", "Other"],
                "EdLevel": ["BS", "BS", "BS"],
                "Industry": ["SW", "SW", "SW"],
                "Age": ["25-34", "25-34", "25-34"],
                "ICorPM": ["IC", "IC", "IC"],
                "OrgSize": ["Small", "Small", "Small"],
            }
        )
        config = {
            "features": {
                "cardinality": {
                    "other_category": "Other",
                    "drop_other_from": ["Country", "DevType"],
                }
            }
        }
        result = drop_other_rows(df, config)
        assert len(result) == 1
        assert result.iloc[0]["Country"] == "USA"

    def test_no_drop_when_list_empty(self):
        """No rows dropped when drop_other_from is empty."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "Other"],
                "DevType": ["Dev", "Other"],
                "EdLevel": ["BS", "BS"],
                "Industry": ["SW", "SW"],
                "Age": ["25-34", "25-34"],
                "ICorPM": ["IC", "IC"],
                "OrgSize": ["Small", "Small"],
            }
        )
        config = {
            "features": {
                "cardinality": {
                    "other_category": "Other",
                    "drop_other_from": [],
                }
            }
        }
        result = drop_other_rows(df, config)
        assert len(result) == 2

    def test_uses_configured_other_name(self):
        """Uses the configured other_category name for matching."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "Misc"],
                "DevType": ["Dev", "Dev"],
                "EdLevel": ["BS", "BS"],
                "Industry": ["SW", "SW"],
                "Age": ["25-34", "25-34"],
                "ICorPM": ["IC", "IC"],
                "OrgSize": ["Small", "Small"],
            }
        )
        config = {
            "features": {
                "cardinality": {
                    "other_category": "Misc",
                    "drop_other_from": ["Country"],
                }
            }
        }
        result = drop_other_rows(df, config)
        assert len(result) == 1


class TestExtractValidCategories:
    """Tests for extract_valid_categories()."""

    def test_returns_sorted_unique_values(self):
        """Returns sorted unique values for each categorical feature."""
        df = pd.DataFrame(
            {
                "Country": ["Germany", "USA", "Germany"],
                "EdLevel": ["BS", "MS", "BS"],
                "DevType": ["Front", "Back", "Front"],
                "Industry": ["SW", "Fin", "SW"],
                "Age": ["25-34", "35-44", "25-34"],
                "ICorPM": ["IC", "PM", "IC"],
                "OrgSize": ["Small", "Large", "Small"],
            }
        )
        result = extract_valid_categories(df)
        assert result["Country"] == ["Germany", "USA"]
        assert result["EdLevel"] == ["BS", "MS"]
        assert result["ICorPM"] == ["IC", "PM"]
        assert result["OrgSize"] == ["Large", "Small"]

    def test_all_categorical_features_present(self):
        """All 7 categorical features are present as keys."""
        df = pd.DataFrame(
            {
                "Country": ["USA"],
                "EdLevel": ["BS"],
                "DevType": ["Dev"],
                "Industry": ["SW"],
                "Age": ["25-34"],
                "ICorPM": ["IC"],
                "OrgSize": ["Small"],
            }
        )
        result = extract_valid_categories(df)
        assert set(result.keys()) == {
            "Country",
            "EdLevel",
            "DevType",
            "Industry",
            "Age",
            "ICorPM",
            "OrgSize",
        }

    def test_excludes_nan_values(self):
        """NaN values are not included in valid categories."""
        df = pd.DataFrame(
            {
                "Country": ["USA", np.nan],
                "EdLevel": ["BS", "MS"],
                "DevType": ["Dev", "Dev"],
                "Industry": ["SW", "SW"],
                "Age": ["25-34", "25-34"],
                "ICorPM": ["IC", "IC"],
                "OrgSize": ["Small", "Small"],
            }
        )
        result = extract_valid_categories(df)
        assert result["Country"] == ["USA"]


class TestComputeCurrencyRates:
    """Tests for compute_currency_rates()."""

    def test_computes_rates_for_valid_countries(self):
        """Returns currency rates for countries present in the data."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "USA", "Germany", "Germany"],
                "Currency": [
                    "USD United States Dollar",
                    "USD United States Dollar",
                    "EUR European Euro",
                    "EUR European Euro",
                ],
                "CompTotal": [100000.0, 120000.0, 80000.0, 90000.0],
                "ConvertedCompYearly": [100000.0, 120000.0, 80000.0, 90000.0],
            }
        )
        result = compute_currency_rates(df, ["USA", "Germany"])
        assert "USA" in result
        assert "Germany" in result
        assert result["USA"]["code"] == "USD"
        assert result["Germany"]["code"] == "EUR"
        assert isinstance(result["USA"]["rate"], float)

    def test_skips_countries_not_in_data(self):
        """Countries not in the data are not included."""
        df = pd.DataFrame(
            {
                "Country": ["USA"],
                "Currency": ["USD United States Dollar"],
                "CompTotal": [100000.0],
                "ConvertedCompYearly": [100000.0],
            }
        )
        result = compute_currency_rates(df, ["USA", "Narnia"])
        assert "USA" in result
        assert "Narnia" not in result

    def test_returns_dict_with_expected_keys(self):
        """Each country entry has code, name, and rate keys."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "USA"],
                "Currency": [
                    "USD United States Dollar",
                    "USD United States Dollar",
                ],
                "CompTotal": [100000.0, 100000.0],
                "ConvertedCompYearly": [100000.0, 100000.0],
            }
        )
        result = compute_currency_rates(df, ["USA"])
        assert set(result["USA"].keys()) == {"code", "name", "rate"}

    def test_filters_extreme_rates(self):
        """Extreme conversion rates are filtered out."""
        df = pd.DataFrame(
            {
                "Country": ["USA", "USA"],
                "Currency": [
                    "USD United States Dollar",
                    "USD United States Dollar",
                ],
                "CompTotal": [100000.0, 0.0001],
                "ConvertedCompYearly": [100000.0, 100000.0],
            }
        )
        result = compute_currency_rates(df, ["USA"])
        # The rate=1.0 row should be kept, the extreme one filtered
        assert result["USA"]["rate"] == 1.0

    def test_empty_dataframe(self):
        """Returns empty dict for empty DataFrame."""
        df = pd.DataFrame(
            columns=["Country", "Currency", "CompTotal", "ConvertedCompYearly"]
        )
        result = compute_currency_rates(df, ["USA"])
        assert result == {}
