"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from preprocessing import prepare_features


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
    # select records with main label more than 1000 USD/year
    df = df[df[main_label] > 1000]
    # further exclude 2% of smallest and 2% of highest salaries
    P = np.percentile(df[main_label], [2, 98])
    df = df[(df[main_label] > P[0]) & (df[main_label] < P[1])]

    print(df.shape)

    # Drop rows with missing target
    df = df.dropna(subset=[main_label])
    print(f"After removing missing targets: {len(df):,} rows")

    # Apply consistent feature transformations (same as used in inference)
    X = prepare_features(df)
    y = df[main_label]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Total features: {X.shape[1]}")

    # Display feature information for debugging and inference comparison
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS (for comparing with inference)")
    print("=" * 60)

    # Show top countries in the dataset
    print("\n📍 Top 10 Countries:")
    top_countries = df["Country"].value_counts().head(10)
    for country, count in top_countries.items():
        print(f"  - {country}: {count:,} ({count/len(df)*100:.1f}%)")

    # Show top education levels
    print("\n🎓 Top Education Levels:")
    top_edu = df["EdLevel"].value_counts().head(10)
    for edu, count in top_edu.items():
        print(f"  - {edu}: {count:,} ({count/len(df)*100:.1f}%)")

    # Show YearsCode statistics
    print("\n💼 Years of Coding Experience:")
    print(f"  - Min: {df['YearsCode'].min():.1f}")
    print(f"  - Max: {df['YearsCode'].max():.1f}")
    print(f"  - Mean: {df['YearsCode'].mean():.1f}")
    print(f"  - Median: {df['YearsCode'].median():.1f}")
    print(f"  - 25th percentile: {df['YearsCode'].quantile(0.25):.1f}")
    print(f"  - 75th percentile: {df['YearsCode'].quantile(0.75):.1f}")

    # Show most common one-hot encoded features (by frequency)
    # Separate analysis for each categorical feature

    # Calculate feature frequencies (sum of each column for one-hot encoded)
    feature_counts = X.sum().sort_values(ascending=False)

    # Exclude numeric features (YearsCode)
    categorical_features = feature_counts[~feature_counts.index.str.startswith('YearsCode')]

    # Country features
    print("\n🌍 Top 15 Country Features (most common):")
    country_features = categorical_features[categorical_features.index.str.startswith('Country_')]
    for i, (feature, count) in enumerate(country_features.head(15).items(), 1):
        percentage = (count / len(X)) * 100
        country_name = feature.replace('Country_', '')
        print(f"  {i:2d}. {country_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)")

    # Education level features
    print("\n🎓 Top 10 Education Level Features (most common):")
    edlevel_features = categorical_features[categorical_features.index.str.startswith('EdLevel_')]
    for i, (feature, count) in enumerate(edlevel_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        edu_name = feature.replace('EdLevel_', '')
        print(f"  {i:2d}. {edu_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)")

    print(f"\n📊 Total one-hot encoded features: {len(X.columns)}")
    print("   - Numeric: 1 (YearsCode)")
    print(f"   - Country: {len(country_features)}")
    print(f"   - Education: {len(edlevel_features)}")

    print("=" * 60 + "\n")

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
    model_path = Path("models/model.pkl")
    artifacts = {
        "model": model,
        "feature_columns": list(X.columns),
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
