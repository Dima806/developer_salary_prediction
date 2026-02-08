"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


def main():
    """Train and save the salary prediction model."""
    print("Loading data...")
    data_path = Path("data/survey_results_public.csv")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please download the Stack Overflow Developer Survey CSV and place it in the data/ directory.")
        print("Download from: https://insights.stackoverflow.com/survey")
        return

    # Load only required columns to save memory
    df = pd.read_csv(
        data_path,
        usecols=["Country", "YearsCode", "EdLevel", "ConvertedCompYearly"],
    )

    print(f"Loaded {len(df):,} rows")

    print("Removing null, extremely small and large reported salaries")
    # select main label
    main_label = "ConvertedCompYearly"
    # Convert compensations into kUSD/year
    df[main_label] = df[main_label]*1e-3
    # select records with main label more than 1kUSD/year
    df = df[df[main_label]>1.0]
    # further exclude 2% of smallest and 2% of highest salaries
    P = np.percentile(df[main_label], [2, 98])
    df = df[(df[main_label] > P[0]) & (df[main_label] < P[1])]

    print(df.shape)

    # Drop rows with missing target
    df = df.dropna(subset=[main_label])
    print(f"After removing missing targets: {len(df):,} rows")

    # Fill missing values in features
    df["YearsCode"] = df["YearsCode"].fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")
    df["EdLevel"] = df["EdLevel"].fillna("Unknown")

    # Create feature matrix with one-hot encoding for categoricals
    X = pd.get_dummies(df[["Country", "YearsCode", "EdLevel"]], drop_first=True)
    y = df[main_label]

    print(f"Feature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration + 1} (early stopping at {model.n_estimators} max)")

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R2 score: {train_score:.4f}")
    print(f"Test R2 score: {test_score:.4f}")

    # Save model and feature columns for inference
    model_path = Path("src/model.pkl")
    artifacts = {
        "model": model,
        "feature_columns": list(X.columns),
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
