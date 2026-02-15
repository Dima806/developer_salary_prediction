"""Inference utilities for salary prediction."""

import pickle
from pathlib import Path

import pandas as pd
import yaml

from src.schema import SalaryInput
from src.preprocessing import prepare_features

# Load model and artifacts at module level
model_path = Path("models/model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found at {model_path}. Please run 'python -m src.train' first."
    )

with open(model_path, "rb") as f:
    artifacts = pickle.load(f)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]

# Load valid categories for input validation
valid_categories_path = Path("config/valid_categories.yaml")

if not valid_categories_path.exists():
    raise FileNotFoundError(
        f"Valid categories file not found at {valid_categories_path}. Please run 'python -m src.train' first."
    )

with open(valid_categories_path, "r") as f:
    valid_categories = yaml.safe_load(f)

# Load currency conversion rates
currency_rates_path = Path("config/currency_rates.yaml")
currency_rates = {}
if currency_rates_path.exists():
    with open(currency_rates_path, "r") as f:
        currency_rates = yaml.safe_load(f) or {}


def get_local_currency(country: str, salary_usd: float) -> dict | None:
    """Convert USD salary to local currency for a given country.

    Returns:
        Dict with code, name, rate, and salary_local, or None if unavailable.
    """
    if country not in currency_rates:
        return None
    info = currency_rates[country]
    return {
        "code": info["code"],
        "name": info["name"],
        "rate": info["rate"],
        "salary_local": round(salary_usd * info["rate"], 2),
    }


def predict_salary(data: SalaryInput) -> float:
    """Predict salary based on input features.

    Args:
        data: SalaryInput model with developer information

    Returns:
        Predicted annual salary in USD

    Raises:
        ValueError: If country or education_level is not in valid categories
    """
    # Validate input against valid categories from training
    if data.country not in valid_categories["Country"]:
        raise ValueError(
            f"Invalid country: '{data.country}'. "
            f"Must be one of {len(valid_categories['Country'])} valid countries. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.education_level not in valid_categories["EdLevel"]:
        raise ValueError(
            f"Invalid education level: '{data.education_level}'. "
            f"Must be one of {len(valid_categories['EdLevel'])} valid education levels. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.dev_type not in valid_categories["DevType"]:
        raise ValueError(
            f"Invalid developer type: '{data.dev_type}'. "
            f"Must be one of {len(valid_categories['DevType'])} valid developer types. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.industry not in valid_categories["Industry"]:
        raise ValueError(
            f"Invalid industry: '{data.industry}'. "
            f"Must be one of {len(valid_categories['Industry'])} valid industries. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.age not in valid_categories["Age"]:
        raise ValueError(
            f"Invalid age: '{data.age}'. "
            f"Must be one of {len(valid_categories['Age'])} valid age ranges. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.ic_or_pm not in valid_categories["ICorPM"]:
        raise ValueError(
            f"Invalid IC or PM value: '{data.ic_or_pm}'. "
            f"Must be one of {len(valid_categories['ICorPM'])} valid values. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    # Create a DataFrame with the input data
    input_df = pd.DataFrame(
        {
            "Country": [data.country],
            "YearsCode": [data.years_code],
            "WorkExp": [data.work_exp],
            "EdLevel": [data.education_level],
            "DevType": [data.dev_type],
            "Industry": [data.industry],
            "Age": [data.age],
            "ICorPM": [data.ic_or_pm],
        }
    )

    # Apply the same preprocessing as training
    input_encoded = prepare_features(input_df)

    # Ensure all feature columns from training are present and in correct order
    # Use reindex to add missing columns with 0s and reorder in one operation
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Ensure non-negative salary
    return max(0.0, float(prediction))
