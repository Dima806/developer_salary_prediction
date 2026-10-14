"""Training script for salary prediction model."""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split

from src.preprocessing import prepare_features, reduce_cardinality

CATEGORICAL_FEATURES = [
    "Country",
    "EdLevel",
    "DevType",
    "Industry",
    "Age",
    "ICorPM",
    "OrgSize",
    "Employment",
    "SurveyYear",
]

# Columns read from each survey CSV (after per-year column renames)
REQUIRED_COLUMNS = [
    "Country",
    "YearsCode",
    "WorkExp",
    "EdLevel",
    "DevType",
    "Industry",
    "Age",
    "ICorPM",
    "OrgSize",
    "Employment",
    "Currency",
    "CompTotal",
    "ConvertedCompYearly",
]


def load_survey_data(config: dict) -> pd.DataFrame:
    """Load every configured survey year and stack them with SurveyYear.

    For each year in config data.sources, reads REQUIRED_COLUMNS (applying
    data.column_renames), maps answer wording via data.category_mappings,
    and adds SurveyYear as a string.

    Args:
        config: Config dict with data.sources and optional
                data.column_renames / data.category_mappings.

    Returns:
        Concatenated DataFrame of all years.

    Raises:
        FileNotFoundError: If a configured survey CSV does not exist.
    """
    data_config = config["data"]
    renames = data_config.get("column_renames") or {}
    mappings = data_config.get("category_mappings") or {}

    frames = []
    for year, path in data_config["sources"].items():
        year = str(year)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file for {year} not found at {path}. Download from "
                "https://survey.stackoverflow.co/"
            )

        year_renames = renames.get(year) or {}
        source_names = {new: old for old, new in year_renames.items()}
        usecols = [source_names.get(col, col) for col in REQUIRED_COLUMNS]
        df = pd.read_csv(path, usecols=usecols).rename(columns=year_renames)

        if year in mappings:
            with open(mappings[year], "r") as f:
                category_mapping = yaml.safe_load(f) or {}
            for col, mapping in category_mapping.items():
                df[col] = df[col].replace(mapping)

        df["SurveyYear"] = year
        print(f"Loaded {len(df):,} rows for {year} from {path}")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def filter_salaries(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter rows by minimum salary and per-year, per-country outlier removal.

    Args:
        df: DataFrame with ConvertedCompYearly, SurveyYear and Country columns.
        config: Config dict with data.min_salary, data.lower_percentile,
                data.upper_percentile.

    Returns:
        Filtered DataFrame with outliers removed.
    """
    main_label = "ConvertedCompYearly"
    min_salary = config["data"]["min_salary"]
    df = df[df[main_label] > min_salary]

    lower_pct = config["data"]["lower_percentile"] / 100
    upper_pct = config["data"]["upper_percentile"] / 100
    groups = df.groupby(["SurveyYear", "Country"])[main_label]
    lower_bound = groups.transform("quantile", lower_pct)
    upper_bound = groups.transform("quantile", upper_pct)
    df = df[(df[main_label] > lower_bound) & (df[main_label] < upper_bound)]

    df = df.dropna(subset=[main_label])
    return df


def apply_cardinality_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cardinality reduction and unicode normalization to categorical columns.

    Args:
        df: DataFrame with categorical feature columns.

    Returns:
        DataFrame with reduced cardinality categories.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].str.replace("\u2019", "'", regex=False)
        df[col] = reduce_cardinality(df[col])
    return df


def drop_other_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop rows where specified features have the 'Other' catch-all category.

    Args:
        df: DataFrame with categorical feature columns (after cardinality reduction).
        config: Config dict with features.cardinality.other_category and
                features.cardinality.drop_other_from.

    Returns:
        DataFrame with 'Other' rows removed from specified features.
    """
    other_name = config["features"]["cardinality"].get("other_category", "Other")
    drop_other_from = config["features"]["cardinality"].get("drop_other_from", [])
    for col in drop_other_from:
        df = df[df[col] != other_name]
    return df


def extract_valid_categories(df: pd.DataFrame) -> dict:
    """Extract sorted unique values per categorical feature for inference validation.

    Args:
        df: DataFrame with categorical feature columns (after cardinality reduction).

    Returns:
        Dict mapping feature name to sorted list of valid category values.
    """
    return {
        col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES
    }


def compute_currency_rates(
    df: pd.DataFrame,
    valid_countries: list[str],
    preferred_codes: dict[str, str] | None = None,
) -> dict:
    """Compute median currency conversion rates per country.

    Args:
        df: DataFrame with Country, Currency, CompTotal, ConvertedCompYearly columns.
        valid_countries: List of valid country names to compute rates for.
        preferred_codes: Optional {country: currency code} to keep (e.g. the
            previous survey year's code) when it still appears in the data;
            otherwise the most common code is used.

    Returns:
        Dict mapping country to {code, name, rate}.
    """
    main_label = "ConvertedCompYearly"
    currency_df = df[["Country", "Currency", "CompTotal", main_label]].dropna()
    currency_df = currency_df.copy()
    currency_df["CurrencyCode"] = currency_df["Currency"].str.split(r"\s+", n=1).str[0]
    currency_df["CurrencyName"] = currency_df["Currency"].str.split(r"\s+", n=1).str[1]
    currency_df["rate"] = currency_df["CompTotal"] / currency_df[main_label]
    currency_df = currency_df[
        (currency_df["rate"] > 0.001) & (currency_df["rate"] < 100000)
    ]

    currency_rates = {}
    for country in valid_countries:
        country_data = currency_df[currency_df["Country"] == country]
        if country_data.empty:
            continue
        preferred = (preferred_codes or {}).get(country)
        if preferred in set(country_data["CurrencyCode"]):
            code = preferred
        else:
            code = country_data["CurrencyCode"].mode().iloc[0]
        name_row = country_data[country_data["CurrencyCode"] == code].iloc[0]
        full_name = name_row["CurrencyName"]
        rates = country_data[country_data["CurrencyCode"] == code]["rate"]
        median_rate = round(float(rates.median()), 2)
        currency_rates[country] = {
            "code": code,
            "name": full_name,
            "rate": median_rate,
        }

    return currency_rates


def save_currency_rates(
    df: pd.DataFrame, valid_countries: list[str], config: dict
) -> dict:
    """Compute currency rates per survey year and write one YAML per year.

    Each year's rates use only that year's rows; output paths come from
    config data.currency_rates. A country keeps the previous year's currency
    while that currency still appears in the data (e.g. Argentina stays on
    ARS even if as many respondents report USD), so years stay comparable.

    Returns:
        Dict mapping survey year to its {country: {code, name, rate}} dict.
    """
    rates_by_year = {}
    previous_codes = {}
    for year, path in sorted(config["data"]["currency_rates"].items()):
        year = str(year)
        year_df = df[df["SurveyYear"] == year]
        rates = compute_currency_rates(year_df, valid_countries, previous_codes)
        previous_codes = {country: info["code"] for country, info in rates.items()}
        with open(path, "w") as f:
            yaml.dump(
                rates,
                f,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=True,
            )
        print(f"Saved currency rates for {len(rates)} countries to {path}")
        for country, info in sorted(rates.items()):
            print(
                f"  {country:45s} -> {info['code']} ({info['name']}, rate: {info['rate']})"
            )
        rates_by_year[year] = rates
    return rates_by_year


def main():
    """Train and save the salary prediction model."""
    # Load configuration
    print("Loading configuration...")
    config_path = Path("config/model_parameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loading data...")
    try:
        df = load_survey_data(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loaded {len(df):,} rows")

    print("Removing null, extremely small and large reported salaries")
    df = filter_salaries(df, config)
    print(f"After filtering: {len(df):,} rows")

    # Apply cardinality reduction to a copy (for valid_categories extraction)
    # and to the main df (for training)
    df_copy = apply_cardinality_reduction(df)
    df = apply_cardinality_reduction(df)

    # Drop rows with "Other" in specified features (low-quality catch-all categories)
    before_drop = len(df)
    df = drop_other_rows(df, config)
    df_copy = drop_other_rows(df_copy, config)
    drop_other_from = config["features"]["cardinality"].get("drop_other_from", [])
    if drop_other_from:
        print(
            f"Dropped {before_drop - len(df):,} rows with 'Other' in {drop_other_from}"
        )
        print(f"After dropping 'Other': {len(df):,} rows")

    # Now apply full feature transformations for model training
    main_label = "ConvertedCompYearly"
    X = prepare_features(df)
    y = df[main_label]

    # Save valid categories after cardinality reduction for validation during inference
    valid_categories = extract_valid_categories(df_copy)

    valid_categories_path = Path("config/valid_categories.yaml")
    with open(valid_categories_path, "w") as f:
        yaml.dump(valid_categories, f, default_flow_style=False, sort_keys=False)

    print(
        f"\nSaved {len(valid_categories['Country'])} valid countries, {len(valid_categories['EdLevel'])} valid education levels, {len(valid_categories['DevType'])} valid developer types, {len(valid_categories['Industry'])} valid industries, {len(valid_categories['Age'])} valid age ranges, and {len(valid_categories['ICorPM'])} valid IC/PM values to {valid_categories_path}"
    )

    # Compute currency conversion rates per country, separately per year
    print("\nComputing currency conversion rates per country and year...")
    save_currency_rates(df, valid_categories["Country"], config)

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

    # Show employment distribution
    print("\n💼 Employment Distribution:")
    top_employment = df["Employment"].value_counts()
    for emp, count in top_employment.items():
        print(f"  - {emp}: {count:,} ({count / len(df) * 100:.1f}%)")

    # Show survey year distribution
    print("\n📅 Survey Year Distribution:")
    for year, count in df["SurveyYear"].value_counts().sort_index().items():
        print(f"  - {year}: {count:,} ({count / len(df) * 100:.1f}%)")

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

        train_mape = np.mean(np.abs((y_train - model.predict(X_train)) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - model.predict(X_test)) / y_test)) * 100
        train_scores.append(train_mape)
        test_scores.append(test_mape)
        best_iterations.append(model.best_iteration + 1)
        print(
            f"  Fold {fold}: Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}% (best iter: {model.best_iteration + 1})"
        )

    avg_train = np.mean(train_scores)
    avg_test = np.mean(test_scores)
    std_test = np.std(test_scores)
    avg_best_iter = int(np.mean(best_iterations))
    print(f"\nCV Average Train MAPE: {avg_train:.2f}%")
    print(f"CV Average Test MAPE:  {avg_test:.2f}% (+/- {std_test:.2f}%)")
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
