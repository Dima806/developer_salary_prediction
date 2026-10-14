"""Tests for src/train.py - Training pipeline helper functions."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.train import (
    REQUIRED_COLUMNS,
    apply_cardinality_reduction,
    compute_currency_rates,
    drop_other_rows,
    extract_valid_categories,
    filter_salaries,
    load_survey_data,
    save_currency_rates,
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
            "Employment": ["Employed"] * n,
            "SurveyYear": ["2025"] * n,
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

    def test_percentiles_computed_per_year(self):
        """Each survey year's outliers are removed using its own bounds."""
        low = [10000.0 + i for i in range(50)]
        high = [100000.0 + i for i in range(50)]
        df = _make_salary_df(salaries=low + high)
        df["SurveyYear"] = ["2025"] * 50 + ["2026"] * 50
        config = {
            "data": {
                "min_salary": 1000,
                "lower_percentile": 2,
                "upper_percentile": 98,
            }
        }
        result = filter_salaries(df, config)
        kept = set(result["ConvertedCompYearly"])
        # Pooled bounds would keep 10049 and 100000; per-year bounds drop them
        for value in (10000.0, 10049.0, 100000.0, 100049.0):
            assert value not in kept


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
                "Employment": ["FT", "FT", "FT"],
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
                "Employment": ["FT", "FT"],
                "SurveyYear": ["2025", "2025"],
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
                "Employment": ["FT", "FT"],
                "SurveyYear": ["2025", "2025"],
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
                "Employment": ["FT", "PT", "FT"],
                "SurveyYear": ["2025", "2026", "2025"],
            }
        )
        result = extract_valid_categories(df)
        assert result["Country"] == ["Germany", "USA"]
        assert result["EdLevel"] == ["BS", "MS"]
        assert result["ICorPM"] == ["IC", "PM"]
        assert result["OrgSize"] == ["Large", "Small"]
        assert result["Employment"] == ["FT", "PT"]

    def test_all_categorical_features_present(self):
        """All 9 categorical features are present as keys."""
        df = pd.DataFrame(
            {
                "Country": ["USA"],
                "EdLevel": ["BS"],
                "DevType": ["Dev"],
                "Industry": ["SW"],
                "Age": ["25-34"],
                "ICorPM": ["IC"],
                "OrgSize": ["Small"],
                "Employment": ["FT"],
                "SurveyYear": ["2025"],
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
            "Employment",
            "SurveyYear",
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
                "Employment": ["FT", "FT"],
                "SurveyYear": ["2025", "2025"],
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


class TestLoadSurveyData:
    """Tests for load_survey_data()."""

    def _write_year(self, path, renames=None, **overrides):
        row = {col: "x" for col in REQUIRED_COLUMNS}
        row.update(
            {"YearsCode": 5, "WorkExp": 3, "CompTotal": 90, "ConvertedCompYearly": 100}
        )
        row.update(overrides)
        row["Extra"] = "ignored"
        df = pd.DataFrame([row]).rename(columns=renames or {})
        df.to_csv(path, index=False)

    def test_stacks_years_with_renames_and_mapping(self, tmp_path):
        """Years are concatenated with SurveyYear, renamed and mapped."""
        path_2025 = tmp_path / "2025.csv"
        path_2026 = tmp_path / "2026.csv"
        mapping_path = tmp_path / "mapping.yaml"
        self._write_year(path_2025, ICorPM="Individual contributor")
        self._write_year(
            path_2026,
            renames={"EdLevel": "Education"},
            ICorPM="Individual contributor in a senior or staff-level role",
        )
        mapping_path.write_text(
            yaml.dump(
                {
                    "ICorPM": {
                        "Individual contributor in a senior or staff-level role": (
                            "Individual contributor"
                        )
                    }
                }
            )
        )
        config = {
            "data": {
                "sources": {"2025": str(path_2025), "2026": str(path_2026)},
                "column_renames": {"2026": {"Education": "EdLevel"}},
                "category_mappings": {"2026": str(mapping_path)},
            }
        }
        result = load_survey_data(config)
        assert list(result["SurveyYear"]) == ["2025", "2026"]
        assert set(result.columns) == set(REQUIRED_COLUMNS) | {"SurveyYear"}
        assert (result["ICorPM"] == "Individual contributor").all()

    def test_missing_file_raises(self, tmp_path):
        """A configured but missing CSV raises FileNotFoundError."""
        config = {"data": {"sources": {"2026": str(tmp_path / "nope.csv")}}}
        with pytest.raises(FileNotFoundError, match="2026"):
            load_survey_data(config)


class TestPreferredCurrencyCode:
    """Tests for compute_currency_rates() with preferred_codes."""

    def _df(self, currencies):
        return pd.DataFrame(
            {
                "Country": ["Argentina"] * len(currencies),
                "Currency": currencies,
                "CompTotal": [
                    1000.0 if c.startswith("ARS") else 1.0 for c in currencies
                ],
                "ConvertedCompYearly": [1.0] * len(currencies),
            }
        )

    def test_keeps_preferred_code_when_present(self):
        """Preferred code wins even if another code is more common."""
        df = self._df(["USD US dollar"] * 3 + ["ARS Argentine peso"])
        result = compute_currency_rates(df, ["Argentina"], {"Argentina": "ARS"})
        assert result["Argentina"]["code"] == "ARS"
        assert result["Argentina"]["rate"] == 1000.0

    def test_falls_back_to_mode_when_preferred_absent(self):
        """If the preferred code is gone (e.g. BGN -> EUR), use the mode."""
        df = self._df(["USD US dollar"] * 3)
        result = compute_currency_rates(df, ["Argentina"], {"Argentina": "ARS"})
        assert result["Argentina"]["code"] == "USD"


class TestSaveCurrencyRates:
    """Tests for save_currency_rates()."""

    def test_rates_computed_per_year(self, tmp_path):
        """Each year's file only uses that year's rows."""
        df = pd.DataFrame(
            {
                "Country": ["Germany", "Germany"],
                "Currency": ["EUR European Euro", "EUR European Euro"],
                "CompTotal": [80.0, 90.0],
                "ConvertedCompYearly": [100.0, 100.0],
                "SurveyYear": ["2025", "2026"],
            }
        )
        paths = {"2025": tmp_path / "r25.yaml", "2026": tmp_path / "r26.yaml"}
        config = {"data": {"currency_rates": {k: str(v) for k, v in paths.items()}}}
        result = save_currency_rates(df, ["Germany"], config)
        assert result["2025"]["Germany"]["rate"] == 0.8
        assert result["2026"]["Germany"]["rate"] == 0.9
        on_disk = yaml.safe_load(paths["2026"].read_text())
        assert on_disk["Germany"]["rate"] == 0.9
