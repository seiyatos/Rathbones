"""Tests for evaluation (AUROC from probabilities)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from investor_ml.models.evaluate import compare_models_auroc, evaluate_model


@pytest.fixture
def simple_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500)),
    ])


def test_evaluate_model_uses_proba(simple_pipeline: Pipeline) -> None:
    X = np.random.RandomState(42).randn(100, 5)
    y = (X[:, 0] + X[:, 1] + np.random.RandomState(42).randn(100) > 0).astype(int)
    simple_pipeline.fit(X, y)
    metrics = evaluate_model(simple_pipeline, X, y, positive_class_index=1)
    assert "roc_auc" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert len(metrics["confusion_matrix"]) == 2


def test_compare_models_auroc_returns_dataframe(sample_raw_df: pd.DataFrame, sample_config: dict) -> None:
    pytest.importorskip("feast", reason="Feast required for training (feature store)")
    import tempfile
    from pathlib import Path

    from investor_ml.models.train import train_models
    with tempfile.TemporaryDirectory() as tmp:
        results = train_models(sample_raw_df, sample_config, artifacts_dir=Path(tmp))
    comparison = compare_models_auroc(results, positive_class_index=1)
    assert isinstance(comparison, pd.DataFrame)
    assert "model" in comparison.columns and "test_roc_auc" in comparison.columns
    assert len(comparison) == len(results)
