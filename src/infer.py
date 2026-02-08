"""Inference utilities for salary prediction."""

import pickle
from pathlib import Path

import pandas as pd

from src.schema import SalaryInput
from src.preprocessing import prepare_features

# Load model and artifacts at module level
model_path = Path("models/model.pkl")

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

    # Apply the same preprocessing as training
    input_encoded = prepare_features(input_df)

    # Ensure all feature columns from training are present and in correct order
    # Use reindex to add missing columns with 0s and reorder in one operation
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Ensure non-negative salary
    return max(0.0, float(prediction))
