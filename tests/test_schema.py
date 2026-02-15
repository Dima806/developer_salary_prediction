"""Tests for src/schema.py - Pydantic input validation."""

import pytest
from pydantic import ValidationError

from src.schema import SalaryInput


def test_valid_input(sample_salary_input):
    """Valid input creates SalaryInput successfully."""
    result = SalaryInput(**sample_salary_input)
    assert result.country == sample_salary_input["country"]
    assert result.years_code == sample_salary_input["years_code"]
    assert result.work_exp == sample_salary_input["work_exp"]
    assert result.education_level == sample_salary_input["education_level"]
    assert result.dev_type == sample_salary_input["dev_type"]
    assert result.industry == sample_salary_input["industry"]
    assert result.age == sample_salary_input["age"]
    assert result.ic_or_pm == sample_salary_input["ic_or_pm"]


def test_negative_years_code(sample_salary_input):
    """Negative years_code raises ValidationError."""
    sample_salary_input["years_code"] = -1.0
    with pytest.raises(ValidationError):
        SalaryInput(**sample_salary_input)


def test_negative_work_exp(sample_salary_input):
    """Negative work_exp raises ValidationError."""
    sample_salary_input["work_exp"] = -5.0
    with pytest.raises(ValidationError):
        SalaryInput(**sample_salary_input)


def test_missing_country():
    """Missing required field raises ValidationError."""
    with pytest.raises(ValidationError):
        SalaryInput(
            years_code=5.0,
            work_exp=3.0,
            education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
            dev_type="Developer, full-stack",
            industry="Software Development",
            age="25-34 years old",
            ic_or_pm="Individual contributor",
        )


def test_missing_education_level():
    """Missing education_level raises ValidationError."""
    with pytest.raises(ValidationError):
        SalaryInput(
            country="United States of America",
            years_code=5.0,
            work_exp=3.0,
            dev_type="Developer, full-stack",
            industry="Software Development",
            age="25-34 years old",
            ic_or_pm="Individual contributor",
        )


def test_zero_years_code(sample_salary_input):
    """Zero years_code is valid (ge=0)."""
    sample_salary_input["years_code"] = 0.0
    result = SalaryInput(**sample_salary_input)
    assert result.years_code == 0.0


def test_zero_work_exp(sample_salary_input):
    """Zero work_exp is valid (ge=0)."""
    sample_salary_input["work_exp"] = 0.0
    result = SalaryInput(**sample_salary_input)
    assert result.work_exp == 0.0
