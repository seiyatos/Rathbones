"""Pytest fixtures and config."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Project root (parent of src/ and tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Raw investor data matching invest.csv schema; enough rows for stratify (2+ per class in test)."""
    base = [
        ("Bank A", "Commit", 300, 40, 2, "Market", 2, 30, 0.0, "Participant", "Bookrunner"),
        ("Bank B", "Decline", 1200, 140, 2, "Market", 2, 115, 20.1, "Bookrunner", "Participant"),
        ("Bank A", "Commit", 900, 130, 3, "Market", 2, 98, 24.4, "Bookrunner", "Bookrunner"),
        ("Bank B", "Decline", 800, 90, 2, "Below", 1, 80, 10.0, "Participant", "Bookrunner"),
        ("Bank A", "Commit", 500, 50, 1, "Market", 0, 40, 2.0, "Bookrunner", "Participant"),
        ("Bank B", "Decline", 1100, 120, 3, "Market", 2, 100, 15.0, "Participant", "Participant"),
        ("Bank A", "Commit", 600, 60, 2, "Market", 1, 55, 5.0, "Participant", "Bookrunner"),
        ("Bank B", "Decline", 1000, 110, 2, "Market", 2, 95, 12.0, "Bookrunner", "Participant"),
        ("Bank A", "Commit", 700, 70, 2, "Market", 2, 65, 8.0, "Bookrunner", "Bookrunner"),
        ("Bank B", "Decline", 950, 100, 3, "Market", 1, 88, 11.0, "Participant", "Bookrunner"),
    ]
    cols = [
        "investor", "commit", "deal_size", "invite", "rating", "int_rate",
        "covenants", "total_fees", "fee_share", "prior_tier", "invite_tier",
    ]
    return pd.DataFrame([dict(zip(cols, row, strict=True)) for row in base])


@pytest.fixture
def sample_config(project_root: Path) -> dict:
    """Minimal config for tests. Training uses the feature store (Feast) as single source of truth."""
    return {
        "seed": 1,
        "data": {
            "raw_path": str(project_root / "data" / "raw" / "invest.csv"),
            "artifacts_dir": str(project_root / "artifacts"),
            "features_parquet": str(project_root / "data" / "processed" / "investor_features.parquet"),
        },
        "feast": {
            "repo_path": str(project_root / "feature_repo"),
            "entity_id_column": "deal_id",
            "event_timestamp_column": "event_timestamp",
            "feature_view_name": "investor_features",
        },
        "splits": {"test_size": 0.2, "stratify_column": "commit_Decline"},
        "feature_engineering": {
            "drop_columns": ["invite_tier", "fee_share", "invite"],
            "target_column": "commit_Decline",
            "drop_after_dummies": ["commit_Commit"],
        },
        "models": {
            "candidates": {
                "l1_logistic": {
                    "estimator": "LogisticRegression",
                    "penalty": "l1",
                    "solver": "saga",
                    "max_iter": 500,
                    "param_grid": {"C": [0.1, 1]},
                },
            },
        },
        "tuning": {"cv": 2, "scoring": "roc_auc", "n_jobs": 1},
        "evaluation": {"probability_positive_class": 1},
    }
