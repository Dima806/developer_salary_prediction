"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
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
        usecols=["Country", "YearsCodePro", "EdLevel", "ConvertedCompYearly"],
    )

    print(f"Loaded {len(df):,} rows")

    # Drop rows with missing target
    df = df.dropna(subset=["ConvertedCompYearly"])
    print(f"After removing missing targets: {len(df):,} rows")

    # Fill missing values in features
    df["YearsCodePro"] = df["YearsCodePro"].fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")
    df["EdLevel"] = df["EdLevel"].fillna("Unknown")

    # Create feature matrix with one-hot encoding for categoricals
    X = pd.get_dummies(df[["Country", "YearsCodePro", "EdLevel"]], drop_first=True)
    y = df["ConvertedCompYearly"]

    print(f"Feature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")

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
