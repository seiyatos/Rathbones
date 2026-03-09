"""Tests for MLflow tracking (file URI, setup, log_training_run)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from investor_ml.tracking.mlflow_tracking import (
    _local_path_to_file_uri,
    log_training_run,
    setup_mlflow,
)


def test_local_path_to_file_uri_produces_file_scheme() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        uri = _local_path_to_file_uri(Path(tmp) / "mlruns")
    assert uri.startswith("file://")
    assert "mlruns" in uri


def test_setup_mlflow_sets_file_uri(project_root: Path) -> None:
    pytest.importorskip("mlflow")
    config = {
        "mlflow": {"tracking_uri": "mlruns", "experiment_name": "test_exp"},
    }
    setup_mlflow(config, project_root=project_root)
    import mlflow
    uri = mlflow.get_tracking_uri()
    assert uri.startswith("file://") or "mlruns" in uri
    mlflow.set_tracking_uri(None)


def test_log_training_run_logs_params_and_metrics(sample_config: dict) -> None:
    pytest.importorskip("mlflow")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tracking_dir = tmp_path / "mlruns"
        tracking_dir.mkdir()
        sample_config["mlflow"] = {"tracking_uri": str(tracking_dir), "experiment_name": "test", "log_models": False}
        setup_mlflow(sample_config, project_root=None)

        comparison_df = pd.DataFrame([
            {"model": "l1", "test_roc_auc": 0.85, "cv_best_score": 0.82},
        ])
        train_results = {
            "l1": {"best_params": {"model__C": 1.0}, "best_estimator": None},
        }
        log_training_run(
            config=sample_config,
            data_path="/data/raw.csv",
            comparison_df=comparison_df,
            train_results=train_results,
            artifacts_dir=tmp_path,
        )
        assert (tmp_path / "comparison.csv").exists()
    import mlflow
    mlflow.set_tracking_uri(None)
