"""Optuna hyperparameter optimization for the salary prediction model."""

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.preprocessing import prepare_features
from src.train import (
    apply_cardinality_reduction,
    drop_other_rows,
    filter_salaries,
)


def sample_params(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample hyperparameters from the search space using an Optuna trial.

    Args:
        trial: Optuna trial object.
        search_space: Dict mapping parameter names to their search config
                      (type, low, high, optional log).

    Returns:
        Dict of sampled hyperparameter values.
    """
    params = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif param_type == "float":
            log = spec.get("log", False)
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=log
            )
    return params


def build_objective(
    X: pd.DataFrame, y: pd.Series, optuna_config: dict
) -> callable:
    """Build an Optuna objective function for XGBoost CV evaluation.

    Args:
        X: Feature matrix.
        y: Target vector.
        optuna_config: Full optuna config dict with search_space, fixed, study.

    Returns:
        Objective function that takes a trial and returns mean MAPE.
    """
    search_space = optuna_config["search_space"]
    fixed = optuna_config["fixed"]
    cv_splits = optuna_config["study"]["cv_splits"]
    random_state = fixed.get("random_state", 42)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, search_space)
        params.update(fixed)

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        mape_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            preds = model.predict(X_test)
            mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
            mape_scores.append(mape)

        return np.mean(mape_scores)

    return objective


def save_best_params(best_params: dict, config_path: Path) -> None:
    """Save the best hyperparameters to model_parameters.yaml.

    Updates the model: section with tuned values, preserving all other config.

    Args:
        best_params: Dict of best hyperparameter values from Optuna.
        config_path: Path to model_parameters.yaml.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["model"].update(best_params)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    """Run Optuna hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for salary prediction"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of optimization trials (overrides config default)",
    )
    args = parser.parse_args()

    # Load configs
    optuna_config_path = Path("config/optuna_config.yaml")
    with open(optuna_config_path, "r") as f:
        optuna_config = yaml.safe_load(f)

    model_config_path = Path("config/model_parameters.yaml")
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    n_trials = args.n_trials or optuna_config["study"]["n_trials"]

    # Load and preprocess data
    print("Loading data...")
    data_path = Path("data/survey_results_public.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print(
            "Please download the Stack Overflow Developer Survey CSV "
            "and place it in the data/ directory."
        )
        return

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

    df = filter_salaries(df, config)
    df = apply_cardinality_reduction(df)
    df = drop_other_rows(df, config)

    main_label = "ConvertedCompYearly"
    X = prepare_features(df)
    y = df[main_label] * config["data"]["salary_scale"]

    print(f"Feature matrix shape: {X.shape}")
    print(f"\nStarting Optuna optimization with {n_trials} trials...")

    # Run optimization
    objective = build_objective(X, y, optuna_config)
    study = optuna.create_study(
        direction=optuna_config["study"]["direction"],
    )
    study.optimize(objective, n_trials=n_trials)

    # Report results
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best MAPE: {study.best_value:.2f}%")
    print("Best hyperparameters:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    # Save best params to model_parameters.yaml
    save_best_params(study.best_params, model_config_path)
    print(f"\nBest parameters saved to {model_config_path}")


if __name__ == "__main__":
    main()
