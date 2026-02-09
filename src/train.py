"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_features, reduce_cardinality


def main():
    """Train and save the salary prediction model."""
    # Load configuration
    print("Loading configuration...")
    config_path = Path("config/model_parameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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
        usecols=["Country", "YearsCode", "EdLevel", "DevType", "ConvertedCompYearly"],
    )

    print(f"Loaded {len(df):,} rows")

    print("Removing null, extremely small and large reported salaries")
    # select main label
    main_label = "ConvertedCompYearly"
    # select records with main label more than min_salary threshold
    min_salary = config['data']['min_salary']
    df = df[df[main_label] > min_salary]
    # further exclude outliers based on percentile bounds
    lower_pct = config['data']['lower_percentile']
    upper_pct = config['data']['upper_percentile']
    P = np.percentile(df[main_label], [lower_pct, upper_pct])
    df = df[(df[main_label] > P[0]) & (df[main_label] < P[1])]

    print(df.shape)

    # Drop rows with missing target
    df = df.dropna(subset=[main_label])
    print(f"After removing missing targets: {len(df):,} rows")

    # Apply preprocessing first to get cardinality-reduced categories
    df_copy = df.copy()

    # Normalize Unicode apostrophes to regular apostrophes for consistency
    df_copy["Country"] = df_copy["Country"].str.replace('\u2019', "'", regex=False)
    df_copy["EdLevel"] = df_copy["EdLevel"].str.replace('\u2019', "'", regex=False)
    df_copy["DevType"] = df_copy["DevType"].str.replace('\u2019', "'", regex=False)

    # Apply cardinality reduction
    df_copy["Country"] = reduce_cardinality(df_copy["Country"])
    df_copy["EdLevel"] = reduce_cardinality(df_copy["EdLevel"])
    df_copy["DevType"] = reduce_cardinality(df_copy["DevType"])

    # Apply cardinality reduction to the actual training data as well
    # (prepare_features no longer does this internally)
    df["Country"] = reduce_cardinality(df["Country"])
    df["EdLevel"] = reduce_cardinality(df["EdLevel"])
    df["DevType"] = reduce_cardinality(df["DevType"])

    # Now apply full feature transformations for model training
    X = prepare_features(df)
    y = df[main_label]

    # Save valid categories after cardinality reduction for validation during inference
    # Extract unique values from the reduced dataframe
    country_values = df_copy["Country"].dropna().unique().tolist()
    edlevel_values = df_copy["EdLevel"].dropna().unique().tolist()
    devtype_values = df_copy["DevType"].dropna().unique().tolist()

    valid_categories = {
        "Country": sorted(country_values),
        "EdLevel": sorted(edlevel_values),
        "DevType": sorted(devtype_values),
    }

    valid_categories_path = Path("config/valid_categories.yaml")
    with open(valid_categories_path, "w") as f:
        yaml.dump(valid_categories, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved {len(valid_categories['Country'])} valid countries, {len(valid_categories['EdLevel'])} valid education levels, and {len(valid_categories['DevType'])} valid developer types to {valid_categories_path}")

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

    # Show top developer types
    print("\n👨‍💻 Top Developer Types:")
    top_devtype = df["DevType"].value_counts().head(10)
    for devtype, count in top_devtype.items():
        print(f"  - {devtype}: {count:,} ({count/len(df)*100:.1f}%)")

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

    # Developer type features
    print("\n👨‍💻 Top 10 Developer Type Features (most common):")
    devtype_features = categorical_features[categorical_features.index.str.startswith('DevType_')]
    for i, (feature, count) in enumerate(devtype_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        devtype_name = feature.replace('DevType_', '')
        print(f"  {i:2d}. {devtype_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)")

    print(f"\n📊 Total one-hot encoded features: {len(X.columns)}")
    print("   - Numeric: 1 (YearsCode)")
    print(f"   - Country: {len(country_features)}")
    print(f"   - Education: {len(edlevel_features)}")
    print(f"   - DevType: {len(devtype_features)}")

    print("=" * 60 + "\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Train model
    print("Training XGBoost model...")
    model_config = config['model']
    model = XGBRegressor(
        n_estimators=model_config['n_estimators'],
        learning_rate=model_config['learning_rate'],
        max_depth=model_config['max_depth'],
        min_child_weight=model_config['min_child_weight'],
        random_state=model_config['random_state'],
        n_jobs=model_config['n_jobs'],
        early_stopping_rounds=model_config['early_stopping_rounds'],
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=config['training']['verbose'],
    )

    print(f"Best iteration: {model.best_iteration + 1} (early stopping at {model.n_estimators} max)")

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R2 score: {train_score:.4f}")
    print(f"Test R2 score: {test_score:.4f}")

    # Save model and feature columns for inference
    model_path = Path(config['training']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    artifacts = {
        "model": model,
        "feature_columns": list(X.columns),
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
