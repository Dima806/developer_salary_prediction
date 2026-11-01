"""Inference utilities for salary prediction."""

import math
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
    artifacts = pickle.load(f)  # nosec B301 - loading trusted local model artifact
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]

# Conformal prediction-interval calibration (per survey year). Prefer the
# copy saved inside the model artifact; fall back to the standalone YAML so
# an older model.pkl can still gain intervals after a re-calibration.
conformal = artifacts.get("conformal")
if conformal is None:
    _conformal_path = Path("config/conformal_intervals.yaml")
    if _conformal_path.exists():
        with open(_conformal_path, "r") as f:
            conformal = yaml.safe_load(f)

# Load valid categories for input validation
valid_categories_path = Path("config/valid_categories.yaml")

if not valid_categories_path.exists():
    raise FileNotFoundError(
        f"Valid categories file not found at {valid_categories_path}. Please run 'python -m src.train' first."
    )

with open(valid_categories_path, "r") as f:
    valid_categories = yaml.safe_load(f)

# Load currency conversion rates, one file per survey year
with open(Path("config/model_parameters.yaml"), "r") as f:
    _currency_rate_paths = yaml.safe_load(f)["data"]["currency_rates"]

currency_rates_by_year = {}
for _year, _path in _currency_rate_paths.items():
    if Path(_path).exists():
        with open(_path, "r") as f:
            currency_rates_by_year[str(_year)] = yaml.safe_load(f) or {}

# 2025 rates, kept under the old name for backward compatibility
currency_rates = currency_rates_by_year.get("2025", {})


def get_local_currency(
    country: str, salary_usd: float, survey_year: str = "2025"
) -> dict | None:
    """Convert USD salary to local currency for a given country and year.

    Uses that survey year's rate; if the country has no rate for that year,
    falls back to another year's rate.

    Returns:
        Dict with code, name, rate, rate_year, and salary_local, or None if
        no rate is available for the country in any year.
    """
    years = [survey_year] + [y for y in currency_rates_by_year if y != survey_year]
    for year in years:
        info = currency_rates_by_year.get(year, {}).get(country)
        if info is not None:
            return {
                "code": info["code"],
                "name": info["name"],
                "rate": info["rate"],
                "rate_year": year,
                "salary_local": round(salary_usd * info["rate"], 2),
            }
    return None


def predict_salaries(data: SalaryInput) -> dict[str, float]:
    """Predict salary for every survey year the model was trained on.

    Returns:
        Dict mapping survey year (e.g. "2025", "2026") to USD salary.
    """
    return {
        year: predict_salary(data, survey_year=year)
        for year in valid_categories["SurveyYear"]
    }


def _interval_bounds(prediction: float, survey_year: str) -> dict:
    """Apply the conformal log-space quantile to a point prediction.

    Returns a dict with prediction plus lower/upper/coverage. When no
    calibration is available, lower/upper/coverage are None (prediction is
    still returned).
    """
    result = {
        "prediction": prediction,
        "lower": None,
        "upper": None,
        "coverage": None,
    }
    if not conformal:
        return result

    quantiles = conformal.get("log_residual_quantile", {})
    q = quantiles.get(survey_year, quantiles.get("_overall"))
    if q is None:
        return result

    result["lower"] = round(max(0.0, prediction * math.exp(-q)), 2)
    result["upper"] = round(prediction * math.exp(q), 2)
    result["coverage"] = conformal.get("coverage")
    return result


def predict_salary_interval(data: SalaryInput, survey_year: str = "2025") -> dict:
    """Predict salary with a conformal prediction interval.

    Args:
        data: SalaryInput model with developer information.
        survey_year: Survey year to predict for (e.g. "2025", "2026").

    Returns:
        Dict with keys: prediction (USD point estimate), lower, upper
        (interval bounds in USD) and coverage (e.g. 0.80). lower/upper/
        coverage are None if the model has no conformal calibration.
    """
    prediction = predict_salary(data, survey_year=survey_year)
    return _interval_bounds(prediction, survey_year)


def predict_salaries_with_intervals(data: SalaryInput) -> dict[str, dict]:
    """Predict salary and interval for every survey year.

    Returns:
        Dict mapping survey year to the dict returned by
        predict_salary_interval.
    """
    return {
        year: predict_salary_interval(data, survey_year=year)
        for year in valid_categories["SurveyYear"]
    }


def predict_salary(data: SalaryInput, survey_year: str = "2025") -> float:
    """Predict salary based on input features.

    Args:
        data: SalaryInput model with developer information
        survey_year: Survey year to predict for (e.g. "2025", "2026")

    Returns:
        Predicted annual salary in USD

    Raises:
        ValueError: If any categorical input or survey_year is not in
            valid categories
    """
    if survey_year not in valid_categories["SurveyYear"]:
        raise ValueError(
            f"Invalid survey year: '{survey_year}'. "
            f"Must be one of {valid_categories['SurveyYear']}."
        )

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

    if data.org_size not in valid_categories["OrgSize"]:
        raise ValueError(
            f"Invalid organization size: '{data.org_size}'. "
            f"Must be one of {len(valid_categories['OrgSize'])} valid sizes. "
            f"Check config/valid_categories.yaml for all valid values."
        )

    if data.employment not in valid_categories["Employment"]:
        raise ValueError(
            f"Invalid employment status: '{data.employment}'. "
            f"Must be one of {valid_categories['Employment']}. "
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
            "OrgSize": [data.org_size],
            "Employment": [data.employment],
            "SurveyYear": [survey_year],
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
