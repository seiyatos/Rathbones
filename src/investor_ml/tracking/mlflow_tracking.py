"""MLflow experiment tracking for training runs: params, metrics, artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_mlflow() -> Any:
    try:
        import mlflow
        return mlflow
    except ImportError as e:
        raise ImportError("MLflow is required for experiment tracking. Install with: pip install mlflow") from e


def _local_path_to_file_uri(path: str | Path) -> str:
    """Convert a local path to a file:// URI so MLflow accepts it (avoids model registry error on Windows)."""
    p = Path(path).resolve()
    return p.as_uri()


def setup_mlflow(config: dict[str, Any], project_root: Path | None = None) -> None:
    """Set MLflow tracking URI and experiment name from config."""
    mlflow = _ensure_mlflow()
    mlflow_cfg = config.get("mlflow", {}) or {}
    uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if uri and not Path(uri).is_absolute() and project_root:
        uri = str(project_root / uri)
    if uri:
        # Use file:// URI so MLflow accepts it for tracking and registry (fixes Windows path error)
        if not uri.startswith(("file:", "http:", "https:", "databricks", "postgresql", "mysql", "sqlite", "mssql")):
            uri = _local_path_to_file_uri(uri)
        mlflow.set_tracking_uri(uri)
    exp_name = mlflow_cfg.get("experiment_name", "investor_ml")
    mlflow.set_experiment(exp_name)
    logger.info("MLflow tracking_uri=%s experiment=%s", mlflow.get_tracking_uri(), exp_name)


def log_training_run(
    config: dict[str, Any],
    data_path: str,
    comparison_df: pd.DataFrame,
    train_results: dict[str, Any],
    artifacts_dir: Path,
    run_name: str | None = None,
) -> None:
    """Log one training run: params, metrics per model, comparison CSV, optionally model artifacts and registry."""
    import joblib

    mlflow = _ensure_mlflow()
    mlflow_cfg = config.get("mlflow", {}) or {}
    log_models = mlflow_cfg.get("log_models", True)
    registry_name = mlflow_cfg.get("registry_name", "investor_ml")

    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_param("seed", config.get("seed", 1))
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("test_size", config.get("splits", {}).get("test_size", 0.2))
        mlflow.log_param("cv", config.get("tuning", {}).get("cv", 5))
        mlflow.log_param("scoring", config.get("tuning", {}).get("scoring", "roc_auc"))

        # Metrics (per model and best)
        for _, row in comparison_df.iterrows():
            model_name = str(row["model"])
            mlflow.log_metric(f"test_roc_auc/{model_name}", float(row["test_roc_auc"]))
            mlflow.log_metric(f"cv_best_score/{model_name}", float(row["cv_best_score"]))

        best_row = comparison_df.iloc[0]
        best_name = str(best_row["model"])
        mlflow.log_metric("best_model_test_roc_auc", float(best_row["test_roc_auc"]))
        mlflow.log_param("best_model", best_name)

        # Artifacts: comparison CSV and optional joblib files
        comparison_path = Path(artifacts_dir) / "comparison.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path), artifact_path="evaluation")

        for name, data in train_results.items():
            mlflow.log_params({f"best_params/{name}/{k}": str(v) for k, v in data.get("best_params", {}).items()})
            if log_models:
                model_path = Path(artifacts_dir) / f"model_{name}.joblib"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path), artifact_path=f"models/{name}")

        # Register best model in this same run (so UI shows one run with params, metrics, and model)
        if registry_name and best_name in train_results:
            best_path = Path(artifacts_dir) / f"model_{best_name}.joblib"
            if best_path.exists():
                try:
                    artifact = joblib.load(best_path)
                    pipeline = artifact.get("pipeline")
                    if pipeline is not None:
                        feature_names = list(getattr(pipeline, "feature_names_in_", []))
                        input_example = (
                            pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
                            if feature_names
                            else None
                        )
                        mlflow.sklearn.log_model(
                            pipeline,
                            name="model",
                            registered_model_name=registry_name,
                            input_example=input_example,
                        )
                        logger.info("Registered best model '%s' as '%s' in same run", best_name, registry_name)
                except Exception as e:
                    logger.warning("Could not register best model in run: %s", e)

        logger.info("MLflow run logged: %s", mlflow.active_run().info.run_id if mlflow.active_run() else "none")


def register_best_model(
    comparison_df: pd.DataFrame,
    train_results: dict[str, Any],
    artifacts_dir: Path,
    registry_name: str = "investor_ml",
) -> None:
    """Log the best model (by test AUROC) with mlflow.sklearn and register it for production."""
    if comparison_df.empty or registry_name is None:
        return
    import joblib

    best_name = str(comparison_df.iloc[0]["model"])
    if best_name not in train_results:
        return
    artifact_path = Path(artifacts_dir) / f"model_{best_name}.joblib"
    if not artifact_path.exists():
        logger.warning("Best model artifact not found: %s", artifact_path)
        return
    mlflow = _ensure_mlflow()
    try:
        artifact = joblib.load(artifact_path)
        pipeline = artifact.get("pipeline")
        if pipeline is None:
            return
        # Build input example for signature inference (avoids "no signature" warning)
        feature_names = list(getattr(pipeline, "feature_names_in_", []))
        input_example = (
            pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
            if feature_names
            else None
        )
        with mlflow.start_run(run_name=None):
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                registered_model_name=registry_name,
                input_example=input_example,
            )
        logger.info("Registered best model '%s' as '%s'", best_name, registry_name)
    except Exception as e:
        logger.warning("Could not register best model: %s", e)
