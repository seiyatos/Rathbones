"""End-to-end pipeline: load data, train models, evaluate with AUROC from probabilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from investor_ml.core.config import get_project_root, load_config
from investor_ml.data.load import load_raw_data
from investor_ml.models.evaluate import compare_models_auroc
from investor_ml.models.train import train_models
from investor_ml.tracking.mlflow_tracking import log_training_run, setup_mlflow

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    format_string: str | None = None,
    file_path: str | Path | None = None,
) -> None:
    """Configure root logger for the pipeline (console and optional file)."""
    fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    level_value = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level_value, format=fmt, force=True)
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level_value)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def run_train_evaluate_pipeline(
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    *,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run full pipeline: load config, (optionally override data path), load data, train, compare.

    Train all models, compare by test AUROC (from probabilities).
    Return training results and comparison DataFrame.
    """
    config = load_config(config_path)
    root = get_project_root()
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        format_string=log_cfg.get("format"),
        file_path=log_cfg.get("file"),
    )
    if config.get("mlflow"):
        setup_mlflow(config, project_root=root)

    data_cfg = config.get("data", {})
    path = data_path or data_cfg.get("raw_path")
    if not path:
        raise ValueError("No data path: set data.raw_path in config or pass data_path")
    path = Path(path)
    if not path.is_absolute():
        path = get_project_root() / path

    df = load_raw_data(path)
    artifacts = artifacts_dir or data_cfg.get("artifacts_dir", "artifacts")
    results = train_models(df, config, artifacts_dir=artifacts)

    eval_cfg = config.get("evaluation", {})
    positive_class = eval_cfg.get("probability_positive_class", 1)
    comparison = compare_models_auroc(results, positive_class_index=positive_class)

    if config.get("mlflow"):
        artifacts_path = Path(
            artifacts_dir or data_cfg.get("artifacts_dir", "artifacts")
        )
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
        log_training_run(
            config=config,
            data_path=str(path),
            comparison_df=comparison,
            train_results=results,
            artifacts_dir=artifacts_path,
        )

    return {
        "config": config,
        "train_results": results,
        "comparison_df": comparison,
    }
