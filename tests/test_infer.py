"""Tests for src/infer.py - Inference and validation."""

import pytest

import src.infer as infer
from src.infer import (
    get_local_currency,
    predict_salaries,
    predict_salaries_with_intervals,
    predict_salary,
    predict_salary_interval,
)
from src.schema import SalaryInput


def test_predict_salary_returns_positive_float(sample_salary_input):
    """predict_salary returns a positive float."""
    result = predict_salary(SalaryInput(**sample_salary_input))
    assert isinstance(result, float)
    assert result > 0


def test_invalid_country(sample_salary_input):
    """Invalid country raises ValueError."""
    sample_salary_input["country"] = "Narnia"
    with pytest.raises(ValueError, match="Invalid country"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_education_level(sample_salary_input):
    """Invalid education level raises ValueError."""
    sample_salary_input["education_level"] = "Fake Degree"
    with pytest.raises(ValueError, match="Invalid education level"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_dev_type(sample_salary_input):
    """Invalid developer type raises ValueError."""
    sample_salary_input["dev_type"] = "Wizard"
    with pytest.raises(ValueError, match="Invalid developer type"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_industry(sample_salary_input):
    """Invalid industry raises ValueError."""
    sample_salary_input["industry"] = "Space Tourism"
    with pytest.raises(ValueError, match="Invalid industry"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_age(sample_salary_input):
    """Invalid age raises ValueError."""
    sample_salary_input["age"] = "100+ years old"
    with pytest.raises(ValueError, match="Invalid age"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_ic_or_pm(sample_salary_input):
    """Invalid IC/PM value raises ValueError."""
    sample_salary_input["ic_or_pm"] = "CEO"
    with pytest.raises(ValueError, match="Invalid IC or PM"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_invalid_org_size(sample_salary_input):
    """Invalid organization size raises ValueError."""
    sample_salary_input["org_size"] = "Megacorp 10M+"
    with pytest.raises(ValueError, match="Invalid organization size"):
        predict_salary(SalaryInput(**sample_salary_input))


def test_get_local_currency_unknown_country():
    """get_local_currency returns None for unknown country."""
    result = get_local_currency("Narnia", 100000)
    assert result is None


def test_get_local_currency_known_country():
    """get_local_currency returns dict with expected keys for a known country."""
    # Use a country that is likely in currency_rates
    result = get_local_currency("United States of America", 100000)
    if result is not None:
        assert "code" in result
        assert "name" in result
        assert "rate" in result
        assert "salary_local" in result
        assert isinstance(result["salary_local"], float)


def test_invalid_survey_year(sample_salary_input):
    """Unknown survey year raises ValueError."""
    with pytest.raises(ValueError, match="Invalid survey year"):
        predict_salary(SalaryInput(**sample_salary_input), survey_year="1999")


def test_predict_salaries_returns_both_years(sample_salary_input):
    """predict_salaries returns one positive USD value per survey year."""
    data = SalaryInput(**sample_salary_input)
    result = predict_salaries(data)
    assert set(result) == {"2025", "2026"}
    assert all(v > 0 for v in result.values())
    assert result["2026"] == predict_salary(data, survey_year="2026")


def test_predict_salary_defaults_to_2025(sample_salary_input):
    """Calling without survey_year predicts for 2025 (backward compatible)."""
    data = SalaryInput(**sample_salary_input)
    assert predict_salary(data) == predict_salary(data, survey_year="2025")


def test_get_local_currency_uses_year_rate(monkeypatch):
    """Each survey year uses its own rate."""
    rates = {
        "2025": {"Germany": {"code": "EUR", "name": "Euro", "rate": 0.8}},
        "2026": {"Germany": {"code": "EUR", "name": "Euro", "rate": 0.9}},
    }
    monkeypatch.setattr(infer, "currency_rates_by_year", rates)
    assert get_local_currency("Germany", 100, "2025")["salary_local"] == 80.0
    result = get_local_currency("Germany", 100, "2026")
    assert result["salary_local"] == 90.0
    assert result["rate_year"] == "2026"


def test_get_local_currency_falls_back_to_other_year(monkeypatch):
    """A country missing from one year's rates uses the other year's rate."""
    rates = {
        "2025": {"Germany": {"code": "EUR", "name": "Euro", "rate": 0.8}},
        "2026": {},
    }
    monkeypatch.setattr(infer, "currency_rates_by_year", rates)
    result = get_local_currency("Germany", 100, "2026")
    assert result["rate"] == 0.8
    assert result["rate_year"] == "2025"


def test_predict_salary_interval_bounds_bracket_point(sample_salary_input):
    """predict_salary_interval returns lower <= prediction <= upper."""
    data = SalaryInput(**sample_salary_input)
    result = predict_salary_interval(data, survey_year="2025")
    assert set(result) == {"prediction", "lower", "upper", "coverage"}
    assert result["prediction"] == predict_salary(data, survey_year="2025")
    if result["lower"] is not None:
        assert result["lower"] >= 0
        assert result["lower"] <= result["prediction"] <= result["upper"]
        assert 0 < result["coverage"] < 1


def test_predict_salaries_with_intervals_covers_all_years(sample_salary_input):
    """One interval dict per survey year, each bracketing its point."""
    data = SalaryInput(**sample_salary_input)
    result = predict_salaries_with_intervals(data)
    assert set(result) == {"2025", "2026"}
    for info in result.values():
        assert info["lower"] <= info["prediction"] <= info["upper"]


def test_interval_width_scales_with_salary(sample_salary_input):
    """The multiplicative interval is wider (in USD) for higher salaries."""
    low = SalaryInput(**{**sample_salary_input, "country": "India"})
    high = SalaryInput(**{**sample_salary_input, "country": "United States of America"})
    low_i = predict_salary_interval(low, "2025")
    high_i = predict_salary_interval(high, "2025")
    if low_i["lower"] is not None and high_i["lower"] is not None:
        low_width = low_i["upper"] - low_i["lower"]
        high_width = high_i["upper"] - high_i["lower"]
        assert high_i["prediction"] > low_i["prediction"]
        assert high_width > low_width


def test_interval_missing_calibration_returns_none(sample_salary_input, monkeypatch):
    """With no conformal calibration, bounds are None but point survives."""
    monkeypatch.setattr(infer, "conformal", None)
    data = SalaryInput(**sample_salary_input)
    result = predict_salary_interval(data, survey_year="2025")
    assert result["lower"] is None
    assert result["upper"] is None
    assert result["coverage"] is None
    assert result["prediction"] > 0
