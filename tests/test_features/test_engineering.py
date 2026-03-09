"""Tests for feature engineering."""

import pandas as pd

from investor_ml.features.engineering import (
    apply_feature_engineering,
    get_target_and_features,
)


def test_apply_feature_engineering_adds_derived(sample_raw_df: pd.DataFrame) -> None:
    out = apply_feature_engineering(sample_raw_df, config=None)
    assert "fee_percent" in out.columns
    assert "invite_percent" in out.columns
    assert "invite_tier" not in out.columns
    assert "fee_share" not in out.columns
    assert "invite" not in out.columns


def test_apply_feature_engineering_one_hot(sample_raw_df: pd.DataFrame) -> None:
    out = apply_feature_engineering(sample_raw_df, config=None)
    assert "commit_Decline" in out.columns
    assert "commit_Commit" not in out.columns  # dropped


def test_apply_feature_engineering_safe_divide_zero() -> None:
    df = pd.DataFrame({
        "investor": ["A"],
        "commit": ["Commit"],
        "deal_size": [100],
        "invite": [10],
        "rating": [1],
        "int_rate": ["Market"],
        "covenants": [0],
        "total_fees": [0],
        "fee_share": [5.0],
        "prior_tier": ["X"],
        "invite_tier": ["Y"],
    })
    out = apply_feature_engineering(df, config=None)
    assert "fee_percent" in out.columns
    assert out["fee_percent"].iloc[0] == 0.0  # filled where total_fees=0


def test_get_target_and_features(sample_raw_df: pd.DataFrame) -> None:
    out = apply_feature_engineering(sample_raw_df, config=None)
    X, y = get_target_and_features(out, target_column="commit_Decline")
    assert "commit_Decline" not in X.columns
    assert y.name == "commit_Decline"
    assert len(X) == len(y)
