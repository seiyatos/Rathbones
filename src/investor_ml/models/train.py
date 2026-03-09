"""Model training: StandardScaler + GridSearchCV for L1/L2 logistic, RF, GB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ESTIMATOR_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


def build_estimator_from_config(
    name: str,
    candidate_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build a sklearn estimator and its param_grid from config.

    Returns (estimator, param_grid). Param grid keys are prefixed for Pipeline.
    """
    est_name = candidate_cfg.get("estimator", "LogisticRegression")
    cls = ESTIMATOR_MAP.get(est_name)
    if cls is None:
        raise ValueError(f"Unknown estimator: {est_name}")

    params = {k: v for k, v in candidate_cfg.items() if k not in ("estimator", "param_grid")}
    estimator = cls(**params)

    param_grid = candidate_cfg.get("param_grid", {})
    # GridSearchCV with Pipeline: param names must be prefixed by step name
    step_name = "model"
    prefixed = {f"{step_name}__{k}": v for k, v in param_grid.items()}
    return estimator, prefixed


def train_models(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run full training using the feature store as the single source of truth.

    Ingest features into Feast (offline) -> get historical features -> train -> materialize to online.
    All training data flows through the feature store.
    """
    seed = config.get("seed", 1)
    data_cfg = config.get("data", {})
    splits_cfg = config.get("splits", {})
    models_cfg = config.get("models", {})
    tuning_cfg = config.get("tuning", {})
    artifacts_dir = Path(artifacts_dir or data_cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    feast_cfg = config.get("feast", {})
    if not feast_cfg:
        raise ValueError(
            "Feature store (Feast) is required for training. Add a 'feast' section to config."
        )

    from investor_ml.store.feast_store import (
        get_training_data_from_feast,
        run_ingest_and_apply,
    )

    # 1. Feature engineering -> write to Parquet -> apply repo (offline store)
    store, df_with_meta = run_ingest_and_apply(df, config)
    entity_df = df_with_meta[["deal_id", "event_timestamp"]].copy()
    # 2. Training data from feature store (single source of truth)
    X, y = get_training_data_from_feast(store, entity_df, config)
    stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=splits_cfg.get("test_size", 0.2),
        random_state=seed,
        stratify=stratify_arg,
    )
    logger.info("Split: train=%s test=%s", X_train.shape[0], X_test.shape[0])

    candidates = models_cfg.get("candidates", {})
    cv = tuning_cfg.get("cv", 5)
    scoring = tuning_cfg.get("scoring", "roc_auc")
    n_jobs = tuning_cfg.get("n_jobs", -1)

    results: dict[str, Any] = {}
    for name, candidate_cfg in candidates.items():
        estimator, param_grid = build_estimator_from_config(name, candidate_cfg)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        # GridSearchCV expects param_grid for the pipeline (scaler__..., model__...)
        # We only tune model params; scaler has none
        gs = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_pipeline = gs.best_estimator_
        results[name] = {
            "best_estimator": best_pipeline,
            "best_score": float(gs.best_score_),
            "best_params": gs.best_params_,
            "cv_results": gs.cv_results_,
            "X_test": X_test,
            "y_test": y_test,
        }
        logger.info("Model %s best CV %s=%.4f", name, scoring, gs.best_score_)

        # Persist best pipeline and test set for evaluation
        out_path = artifacts_dir / f"model_{name}.joblib"
        joblib.dump(
            {
                "pipeline": best_pipeline,
                "best_params": gs.best_params_,
                "feature_names_in": list(X_train.columns),
            },
            out_path,
        )
        logger.info("Saved artifact: %s", out_path)

    # Save test set once (same for all models)
    joblib.dump(
        {"X_test": X_test, "y_test": y_test, "feature_names": list(X_test.columns)},
        artifacts_dir / "test_set.joblib",
    )

    # 3. Materialize to online store so serving can use the same features
    try:
        from investor_ml.store.feast_store import materialize_to_online
        materialize_to_online(
            store,
            feature_view_name=feast_cfg.get("feature_view_name", "investor_features"),
        )
    except Exception as e:
        logger.warning("Feast materialize_to_online skipped: %s", e)

    return results


def load_trained_artifact(artifacts_dir: str | Path, model_name: str) -> dict[str, Any]:
    """Load a single model artifact."""
    path = Path(artifacts_dir) / f"model_{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
