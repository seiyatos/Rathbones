"""Evaluation: AUROC from predicted probabilities (not hard class predictions)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)


def _get_proba_positive(estimator: Any, X: pd.DataFrame | np.ndarray, positive_class_index: int = 1) -> np.ndarray:
    """Get predicted probability for the positive class (Decline = 1)."""
    proba = estimator.predict_proba(X)
    if proba.shape[1] < 2:
        return proba.ravel()
    return proba[:, positive_class_index]


def evaluate_model(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    y_true: pd.Series | np.ndarray,
    *,
    positive_class_index: int = 1,
) -> dict[str, Any]:
    """Evaluate a binary classifier using AUROC from predicted probabilities.

    Uses probabilities, not hard predictions. Also return confusion matrix.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = _get_proba_positive(estimator, X, positive_class_index=positive_class_index)

    auroc = roc_auc_score(y_true, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "roc_auc": float(auroc),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }
    logger.info("AUROC (from probabilities): %.4f | CM: %s", auroc, cm.tolist())
    return metrics


def compare_models_auroc(
    results: dict[str, Any],
    *,
    positive_class_index: int = 1,
) -> pd.DataFrame:
    """From train_models() results, compute test-set AUROC for each model.

    Uses predicted probabilities. Return a DataFrame sorted by AUROC descending.
    """
    rows = []
    for name, data in results.items():
        pipe = data["best_estimator"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        metrics = evaluate_model(
            pipe, X_test, y_test,
            positive_class_index=positive_class_index,
        )
        rows.append({
            "model": name,
            "test_roc_auc": metrics["roc_auc"],
            "cv_best_score": data["best_score"],
            "confusion_matrix": metrics["confusion_matrix"],
        })
    df = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    logger.info("Model comparison (test AUROC from probabilities):\n%s", df[["model", "test_roc_auc"]].to_string())
    return df
