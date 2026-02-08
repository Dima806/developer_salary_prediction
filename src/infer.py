"""Inference utilities for salary prediction."""

import pickle
from pathlib import Path

import pandas as pd

from src.schema import SalaryInput

# Load model and artifacts at module level
model_path = Path("src/model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found at {model_path}. Please run 'python src/train.py' first."
    )

with open(model_path, "rb") as f:
    artifacts = pickle.load(f)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]


def predict_salary(data: SalaryInput) -> float:
    """Predict salary based on input features.

    Args:
        data: SalaryInput model with developer information

    Returns:
        Predicted annual salary in USD
    """
    # Create a DataFrame with the input data
    input_df = pd.DataFrame(
        {
            "Country": [data.country],
            "YearsCodePro": [data.years_code_pro],
            "EdLevel": [data.education_level],
        }
    )

    # Apply one-hot encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure all feature columns from training are present
    # Add missing columns with 0s
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training
    input_encoded = input_encoded[feature_columns]

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Ensure non-negative salary
    return max(0.0, float(prediction))
