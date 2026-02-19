"""Tests for src/tune.py - Optuna hyperparameter optimization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.tune import build_objective, sample_params, save_best_params


SAMPLE_SEARCH_SPACE = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
}


class TestSampleParams:
    """Tests for sample_params()."""

    def test_returns_all_params(self):
        """All search space parameters are returned."""
        trial = MagicMock()
        trial.suggest_int.return_value = 6
        trial.suggest_float.return_value = 0.01
        result = sample_params(trial, SAMPLE_SEARCH_SPACE)
        assert set(result.keys()) == {"max_depth", "learning_rate", "subsample"}

    def test_calls_suggest_int_for_int_type(self):
        """suggest_int is called for int-type parameters."""
        trial = MagicMock()
        trial.suggest_int.return_value = 6
        trial.suggest_float.return_value = 0.01
        sample_params(trial, SAMPLE_SEARCH_SPACE)
        trial.suggest_int.assert_called_once_with("max_depth", 3, 10)

    def test_calls_suggest_float_with_log(self):
        """suggest_float is called with log=True for log-distributed parameters."""
        trial = MagicMock()
        trial.suggest_int.return_value = 6
        trial.suggest_float.return_value = 0.01
        sample_params(trial, SAMPLE_SEARCH_SPACE)
        calls = trial.suggest_float.call_args_list
        lr_call = [c for c in calls if c[0][0] == "learning_rate"][0]
        assert lr_call.kwargs["log"] is True

    def test_calls_suggest_float_without_log(self):
        """suggest_float is called with log=False when not specified."""
        trial = MagicMock()
        trial.suggest_int.return_value = 6
        trial.suggest_float.return_value = 0.8
        sample_params(trial, SAMPLE_SEARCH_SPACE)
        calls = trial.suggest_float.call_args_list
        sub_call = [c for c in calls if c[0][0] == "subsample"][0]
        assert sub_call.kwargs["log"] is False


class TestBuildObjective:
    """Tests for build_objective()."""

    def test_returns_callable(self):
        """build_objective returns a callable."""
        import pandas as pd

        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        optuna_config = {
            "search_space": SAMPLE_SEARCH_SPACE,
            "fixed": {"n_estimators": 10, "random_state": 42, "n_jobs": 1},
            "study": {"cv_splits": 2, "direction": "minimize"},
        }
        result = build_objective(X, y, optuna_config)
        assert callable(result)


class TestSaveBestParams:
    """Tests for save_best_params()."""

    def test_updates_model_section(self):
        """Best params are written to the model section of the config."""
        config = {
            "data": {"min_salary": 1000},
            "model": {
                "n_estimators": 5000,
                "learning_rate": 0.01,
                "max_depth": 6,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            tmp_path = Path(f.name)

        best_params = {"learning_rate": 0.05, "max_depth": 8}
        save_best_params(best_params, tmp_path)

        with open(tmp_path, "r") as f:
            updated = yaml.safe_load(f)

        assert updated["model"]["learning_rate"] == 0.05
        assert updated["model"]["max_depth"] == 8
        tmp_path.unlink()

    def test_preserves_other_config_sections(self):
        """Non-model config sections are preserved."""
        config = {
            "data": {"min_salary": 1000, "test_size": 0.2},
            "model": {"n_estimators": 5000, "max_depth": 6},
            "training": {"verbose": False},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            tmp_path = Path(f.name)

        save_best_params({"max_depth": 8}, tmp_path)

        with open(tmp_path, "r") as f:
            updated = yaml.safe_load(f)

        assert updated["data"]["min_salary"] == 1000
        assert updated["training"]["verbose"] is False
        tmp_path.unlink()

    def test_preserves_existing_model_params(self):
        """Existing model params not in best_params are preserved."""
        config = {
            "model": {
                "n_estimators": 5000,
                "max_depth": 6,
                "n_jobs": -1,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            tmp_path = Path(f.name)

        save_best_params({"max_depth": 8}, tmp_path)

        with open(tmp_path, "r") as f:
            updated = yaml.safe_load(f)

        assert updated["model"]["n_estimators"] == 5000
        assert updated["model"]["n_jobs"] == -1
        assert updated["model"]["max_depth"] == 8
        tmp_path.unlink()
