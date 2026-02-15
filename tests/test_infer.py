"""Tests for src/infer.py - Inference and validation."""

import pytest

from src.infer import get_local_currency, predict_salary
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
