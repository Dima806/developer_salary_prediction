"""Shared fixtures for pytest tests."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_salary_input():
    """Return a dict with valid SalaryInput fields."""
    return {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }


@pytest.fixture
def valid_categories_data():
    """Load and return valid categories from config."""
    path = Path("config/valid_categories.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def model_config():
    """Load and return model parameters config."""
    path = Path("config/model_parameters.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
