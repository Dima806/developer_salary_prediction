"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split

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
        print(
            "Please download the Stack Overflow Developer Survey CSV and place it in the data/ directory."
        )
        print("Download from: https://insights.stackoverflow.com/survey")
        return

    # Load only required columns to save memory
    df = pd.read_csv(
        data_path,
        usecols=[
            "Country",
            "YearsCode",
            "WorkExp",
            "EdLevel",
            "DevType",
            "Industry",
            "Age",
            "ICorPM",
            "Currency",
            "CompTotal",
            "ConvertedCompYearly",
        ],
    )

    print(f"Loaded {len(df):,} rows")

    print("Removing null, extremely small and large reported salaries")
    # select main label
    main_label = "ConvertedCompYearly"
    # select records with main label more than min_salary threshold
    min_salary = config["data"]["min_salary"]
    df = df[df[main_label] > min_salary]
    # Exclude outliers based on percentile bounds PER COUNTRY
    # This preserves records from lower-paid and higher-paid countries
    # that would otherwise be removed by global percentile filtering
    lower_pct = config["data"]["lower_percentile"] / 100
    upper_pct = config["data"]["upper_percentile"] / 100
    lower_bound = df.groupby("Country")[main_label].transform("quantile", lower_pct)
    upper_bound = df.groupby("Country")[main_label].transform("quantile", upper_pct)
    df = df[(df[main_label] > lower_bound) & (df[main_label] < upper_bound)]

    print(df.shape)

    # Drop rows with missing target
    df = df.dropna(subset=[main_label])
    print(f"After removing missing targets: {len(df):,} rows")

    # Apply preprocessing first to get cardinality-reduced categories
    df_copy = df.copy()

    # Normalize Unicode apostrophes to regular apostrophes for consistency
    df_copy["Country"] = df_copy["Country"].str.replace("\u2019", "'", regex=False)
    df_copy["EdLevel"] = df_copy["EdLevel"].str.replace("\u2019", "'", regex=False)
    df_copy["DevType"] = df_copy["DevType"].str.replace("\u2019", "'", regex=False)
    df_copy["Industry"] = df_copy["Industry"].str.replace("\u2019", "'", regex=False)
    df_copy["Age"] = df_copy["Age"].str.replace("\u2019", "'", regex=False)
    df_copy["ICorPM"] = df_copy["ICorPM"].str.replace("\u2019", "'", regex=False)

    # Apply cardinality reduction
    df_copy["Country"] = reduce_cardinality(df_copy["Country"])
    df_copy["EdLevel"] = reduce_cardinality(df_copy["EdLevel"])
    df_copy["DevType"] = reduce_cardinality(df_copy["DevType"])
    df_copy["Industry"] = reduce_cardinality(df_copy["Industry"])
    df_copy["Age"] = reduce_cardinality(df_copy["Age"])
    df_copy["ICorPM"] = reduce_cardinality(df_copy["ICorPM"])

    # Apply cardinality reduction to the actual training data as well
    # (prepare_features no longer does this internally)
    df["Country"] = reduce_cardinality(df["Country"])
    df["EdLevel"] = reduce_cardinality(df["EdLevel"])
    df["DevType"] = reduce_cardinality(df["DevType"])
    df["Industry"] = reduce_cardinality(df["Industry"])
    df["Age"] = reduce_cardinality(df["Age"])
    df["ICorPM"] = reduce_cardinality(df["ICorPM"])

    # Drop rows with "Other" in specified features (low-quality catch-all categories)
    other_name = config["features"]["cardinality"].get("other_category", "Other")
    drop_other_from = config["features"]["cardinality"].get("drop_other_from", [])
    if drop_other_from:
        before_drop = len(df)
        for col in drop_other_from:
            df = df[df[col] != other_name]
            df_copy = df_copy[df_copy[col] != other_name]
        print(
            f"Dropped {before_drop - len(df):,} rows with '{other_name}' in {drop_other_from}"
        )
        print(f"After dropping 'Other': {len(df):,} rows")

    # Now apply full feature transformations for model training
    X = prepare_features(df)
    y = df[main_label]

    # Save valid categories after cardinality reduction for validation during inference
    # Extract unique values from the reduced dataframe
    country_values = df_copy["Country"].dropna().unique().tolist()
    edlevel_values = df_copy["EdLevel"].dropna().unique().tolist()
    devtype_values = df_copy["DevType"].dropna().unique().tolist()
    industry_values = df_copy["Industry"].dropna().unique().tolist()
    age_values = df_copy["Age"].dropna().unique().tolist()
    icorpm_values = df_copy["ICorPM"].dropna().unique().tolist()

    valid_categories = {
        "Country": sorted(country_values),
        "EdLevel": sorted(edlevel_values),
        "DevType": sorted(devtype_values),
        "Industry": sorted(industry_values),
        "Age": sorted(age_values),
        "ICorPM": sorted(icorpm_values),
    }

    valid_categories_path = Path("config/valid_categories.yaml")
    with open(valid_categories_path, "w") as f:
        yaml.dump(valid_categories, f, default_flow_style=False, sort_keys=False)

    print(
        f"\nSaved {len(valid_categories['Country'])} valid countries, {len(valid_categories['EdLevel'])} valid education levels, {len(valid_categories['DevType'])} valid developer types, {len(valid_categories['Industry'])} valid industries, {len(valid_categories['Age'])} valid age ranges, and {len(valid_categories['ICorPM'])} valid IC/PM values to {valid_categories_path}"
    )

    # Compute currency conversion rates per country
    # Use the original data with Currency and CompTotal columns
    print("\nComputing currency conversion rates per country...")
    currency_df = df[["Country", "Currency", "CompTotal", main_label]].dropna()
    # Extract 3-letter currency code from values like "EUR European Euro"
    currency_df = currency_df.copy()
    currency_df["CurrencyCode"] = currency_df["Currency"].str.split(r"\s+", n=1).str[0]
    currency_df["CurrencyName"] = currency_df["Currency"].str.split(r"\s+", n=1).str[1]
    # Compute conversion rate: local currency / USD
    currency_df["rate"] = currency_df["CompTotal"] / currency_df[main_label]
    # Filter out unreasonable rates (negative, zero, or extreme)
    currency_df = currency_df[
        (currency_df["rate"] > 0.001) & (currency_df["rate"] < 100000)
    ]

    currency_rates = {}
    for country in valid_categories["Country"]:
        country_data = currency_df[currency_df["Country"] == country]
        if country_data.empty:
            continue
        # Find the most common currency for this country
        most_common = country_data["CurrencyCode"].mode()
        if most_common.empty:
            continue
        code = most_common.iloc[0]
        # Get the full name from the first matching record
        name_row = country_data[country_data["CurrencyCode"] == code].iloc[0]
        full_name = name_row["CurrencyName"]
        # Compute median conversion rate for this country+currency pair
        rates = country_data[country_data["CurrencyCode"] == code]["rate"]
        median_rate = round(float(rates.median()), 2)
        currency_rates[country] = {
            "code": code,
            "name": full_name,
            "rate": median_rate,
        }

    currency_rates_path = Path("config/currency_rates.yaml")
    with open(currency_rates_path, "w") as f:
        yaml.dump(
            currency_rates,
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )

    print(
        f"Saved currency rates for {len(currency_rates)} countries to {currency_rates_path}"
    )
    for country, info in sorted(currency_rates.items()):
        print(
            f"  {country:45s} -> {info['code']} ({info['name']}, rate: {info['rate']})"
        )

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
        print(f"  - {country}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show top education levels
    print("\n🎓 Top Education Levels:")
    top_edu = df["EdLevel"].value_counts().head(10)
    for edu, count in top_edu.items():
        print(f"  - {edu}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show top developer types
    print("\n👨‍💻 Top Developer Types:")
    top_devtype = df["DevType"].value_counts().head(10)
    for devtype, count in top_devtype.items():
        print(f"  - {devtype}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show top industries
    print("\n🏢 Top Industries:")
    top_industry = df["Industry"].value_counts().head(10)
    for industry, count in top_industry.items():
        print(f"  - {industry}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show age distribution
    print("\n🎂 Age Distribution:")
    top_age = df["Age"].value_counts().head(10)
    for age, count in top_age.items():
        print(f"  - {age}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show IC or PM distribution
    print("\n👥 IC or PM Distribution:")
    top_icorpm = df["ICorPM"].value_counts().head(10)
    for icorpm, count in top_icorpm.items():
        print(f"  - {icorpm}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show YearsCode statistics
    print("\n💼 Years of Coding Experience:")
    print(f"  - Min: {df['YearsCode'].min():.1f}")
    print(f"  - Max: {df['YearsCode'].max():.1f}")
    print(f"  - Mean: {df['YearsCode'].mean():.1f}")
    print(f"  - Median: {df['YearsCode'].median():.1f}")
    print(f"  - 25th percentile: {df['YearsCode'].quantile(0.25):.1f}")
    print(f"  - 75th percentile: {df['YearsCode'].quantile(0.75):.1f}")

    # Show WorkExp statistics
    print("\n💼 Years of Professional Work Experience:")
    print(f"  - Min: {df['WorkExp'].min():.1f}")
    print(f"  - Max: {df['WorkExp'].max():.1f}")
    print(f"  - Mean: {df['WorkExp'].mean():.1f}")
    print(f"  - Median: {df['WorkExp'].median():.1f}")
    print(f"  - 25th percentile: {df['WorkExp'].quantile(0.25):.1f}")
    print(f"  - 75th percentile: {df['WorkExp'].quantile(0.75):.1f}")

    # Show most common one-hot encoded features (by frequency)
    # Separate analysis for each categorical feature

    # Calculate feature frequencies (sum of each column for one-hot encoded)
    feature_counts = X.sum().sort_values(ascending=False)

    # Exclude numeric features (YearsCode)
    categorical_features = feature_counts[
        ~feature_counts.index.str.startswith("YearsCode")
    ]

    # Country features
    print("\n🌍 Top 15 Country Features (most common):")
    country_features = categorical_features[
        categorical_features.index.str.startswith("Country_")
    ]
    for i, (feature, count) in enumerate(country_features.head(15).items(), 1):
        percentage = (count / len(X)) * 100
        country_name = feature.replace("Country_", "")
        print(
            f"  {i:2d}. {country_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    # Education level features
    print("\n🎓 Top 10 Education Level Features (most common):")
    edlevel_features = categorical_features[
        categorical_features.index.str.startswith("EdLevel_")
    ]
    for i, (feature, count) in enumerate(edlevel_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        edu_name = feature.replace("EdLevel_", "")
        print(
            f"  {i:2d}. {edu_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    # Developer type features
    print("\n👨‍💻 Top 10 Developer Type Features (most common):")
    devtype_features = categorical_features[
        categorical_features.index.str.startswith("DevType_")
    ]
    for i, (feature, count) in enumerate(devtype_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        devtype_name = feature.replace("DevType_", "")
        print(
            f"  {i:2d}. {devtype_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    # Industry features
    print("\n🏢 Top 10 Industry Features (most common):")
    industry_features = categorical_features[
        categorical_features.index.str.startswith("Industry_")
    ]
    for i, (feature, count) in enumerate(industry_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        industry_name = feature.replace("Industry_", "")
        print(
            f"  {i:2d}. {industry_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    # Age features
    print("\n🎂 Top 10 Age Features (most common):")
    age_features = categorical_features[
        categorical_features.index.str.startswith("Age_")
    ]
    for i, (feature, count) in enumerate(age_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        age_name = feature.replace("Age_", "")
        print(
            f"  {i:2d}. {age_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    # ICorPM features
    print("\n👥 Top 10 IC/PM Features (most common):")
    icorpm_features = categorical_features[
        categorical_features.index.str.startswith("ICorPM_")
    ]
    for i, (feature, count) in enumerate(icorpm_features.head(10).items(), 1):
        percentage = (count / len(X)) * 100
        icorpm_name = feature.replace("ICorPM_", "")
        print(
            f"  {i:2d}. {icorpm_name:45s} - {count:6.0f} occurrences ({percentage:5.1f}%)"
        )

    print(f"\n📊 Total one-hot encoded features: {len(X.columns)}")
    print("   - Numeric: 2 (YearsCode, WorkExp)")
    print(f"   - Country: {len(country_features)}")
    print(f"   - Education: {len(edlevel_features)}")
    print(f"   - DevType: {len(devtype_features)}")
    print(f"   - Industry: {len(industry_features)}")
    print(f"   - Age: {len(age_features)}")
    print(f"   - ICorPM: {len(icorpm_features)}")

    print("=" * 60 + "\n")

    # Cross-validation for robust evaluation
    n_splits = config["data"].get("cv_splits", 5)
    random_state = config["data"]["random_state"]
    model_config = config["model"]

    print(f"Running {n_splits}-fold cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []
    best_iterations = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=model_config["n_estimators"],
            learning_rate=model_config["learning_rate"],
            max_depth=model_config["max_depth"],
            min_child_weight=model_config["min_child_weight"],
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
            early_stopping_rounds=model_config["early_stopping_rounds"],
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        train_scores.append(train_r2)
        test_scores.append(test_r2)
        best_iterations.append(model.best_iteration + 1)
        print(
            f"  Fold {fold}: Train R2 = {train_r2:.4f}, Test R2 = {test_r2:.4f} (best iter: {model.best_iteration + 1})"
        )

    avg_train = np.mean(train_scores)
    avg_test = np.mean(test_scores)
    std_test = np.std(test_scores)
    avg_best_iter = int(np.mean(best_iterations))
    print(f"\nCV Average Train R2: {avg_train:.4f}")
    print(f"CV Average Test R2:  {avg_test:.4f} (+/- {std_test:.4f})")
    print(f"CV Average best iteration: {avg_best_iter}")

    # Train final model on all data for deployment
    # Use a small held-out split for early stopping only
    print("\nTraining final model on full dataset...")
    X_train_final, X_es, y_train_final, y_es = train_test_split(
        X, y, test_size=0.1, random_state=random_state
    )

    final_model = XGBRegressor(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        min_child_weight=model_config["min_child_weight"],
        random_state=model_config["random_state"],
        n_jobs=model_config["n_jobs"],
        early_stopping_rounds=model_config["early_stopping_rounds"],
    )
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=[(X_es, y_es)],
        verbose=config["training"]["verbose"],
    )
    print(f"Final model best iteration: {final_model.best_iteration + 1}")

    # Save model and feature columns for inference
    model_path = Path(config["training"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model": final_model,
        "feature_columns": list(X.columns),
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
