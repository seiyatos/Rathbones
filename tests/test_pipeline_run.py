"""Tests for the end-to-end pipeline (run_train_evaluate_pipeline)."""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from investor_ml.pipeline.run import run_train_evaluate_pipeline, setup_logging


def test_setup_logging_defaults() -> None:
    """setup_logging runs without error (coverage for pipeline module)."""
    setup_logging(level="WARNING")


def test_setup_logging_with_file() -> None:
    """setup_logging with file_path creates and uses the file."""
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "test.log"
        setup_logging(level="INFO", file_path=str(log_file))
        assert log_file.exists()
        # Close file handlers so Windows can delete the temp dir
        for h in list(logging.getLogger().handlers):
            if getattr(h, "baseFilename", None):
                h.close()
                logging.getLogger().removeHandler(h)


def test_run_train_evaluate_pipeline_minimal(
    project_root: Path, sample_raw_df: pd.DataFrame, sample_config: dict
) -> None:
    """Run full pipeline with temp data and artifacts (no MLflow). Covers pipeline/run.py."""
    pytest.importorskip("feast", reason="Feast required for pipeline (feature store)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "raw.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        config_path = tmp_path / "config.yaml"
        config = {
            **sample_config,
            "data": {
                **sample_config["data"],
                "raw_path": str(csv_path),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "mlflow": None,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        out = run_train_evaluate_pipeline(
            config_path=str(config_path),
            data_path=str(csv_path),
            artifacts_dir=str(tmp_path / "artifacts"),
        )
    assert "config" in out
    assert "train_results" in out
    assert "comparison_df" in out
    assert "l1_logistic" in out["train_results"]
    assert len(out["comparison_df"]) >= 1
