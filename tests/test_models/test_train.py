"""Tests for model training."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from investor_ml.models.train import build_estimator_from_config, load_trained_artifact, train_models


def test_build_estimator_l1() -> None:
    cfg = {
        "estimator": "LogisticRegression",
        "penalty": "l1",
        "solver": "saga",
        "max_iter": 500,
        "param_grid": {"C": [0.1, 1]},
    }
    est, grid = build_estimator_from_config("l1", cfg)
    assert "model__C" in grid
    assert grid["model__C"] == [0.1, 1]


def test_build_estimator_rf() -> None:
    cfg = {
        "estimator": "RandomForestClassifier",
        "param_grid": {"n_estimators": [10], "max_features": [0.5]},
    }
    est, grid = build_estimator_from_config("rf", cfg)
    assert "model__n_estimators" in grid


def test_train_models_returns_results(sample_raw_df: pd.DataFrame, sample_config: dict) -> None:
    pytest.importorskip("feast", reason="Feast required for training (feature store)")
    with tempfile.TemporaryDirectory() as tmp:
        results = train_models(sample_raw_df, sample_config, artifacts_dir=Path(tmp))
    assert "l1_logistic" in results
    r = results["l1_logistic"]
    assert "best_estimator" in r
    assert "best_score" in r
    assert "X_test" in r and "y_test" in r


def test_load_trained_artifact_after_train(sample_raw_df: pd.DataFrame, sample_config: dict) -> None:
    pytest.importorskip("feast", reason="Feast required for training")
    with tempfile.TemporaryDirectory() as tmp:
        train_models(sample_raw_df, sample_config, artifacts_dir=Path(tmp))
        artifact = load_trained_artifact(Path(tmp), "l1_logistic")
    assert "pipeline" in artifact
    assert "feature_names_in" in artifact or "best_params" in artifact
