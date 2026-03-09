"""Feature engineering: derived features, drop columns, one-hot encoding.

Mirrors notebook logic for production use.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from investor_ml.data.validate import validate_raw_schema

logger = logging.getLogger(__name__)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """Element-wise division, filling where denominator is 0."""
    with pd.option_context("mode.chained_assignment", None):
        out = numerator / denominator
    out = out.where(denominator > 0, fill)
    return out


def build_feature_engineering_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract feature-engineering section from config."""
    return config.get("feature_engineering", {})


def apply_feature_engineering(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    drop_columns: list[str] | None = None,
    fee_num: str = "fee_share",
    fee_denom: str = "total_fees",
    invite_num: str = "invite",
    invite_denom: str = "deal_size",
    drop_after_dummies: list[str] | None = None,
) -> pd.DataFrame:
    """Build derived features, drop columns, one-hot encode, then drop post-dummy columns.

    1. fee_percent = fee_share / total_fees
    2. invite_percent = invite / deal_size
    3. Drop invite_tier, fee_share, invite
    4. One-hot encode categoricals (get_dummies)
    5. Drop commit_Commit (and any other drop_after_dummies)
    """
    if config:
        fe = build_feature_engineering_params(config)
        drop_columns = drop_columns or fe.get("drop_columns", [])
        fee_cfg = fe.get("fee_percent", {})
        invite_cfg = fe.get("invite_percent", {})
        fee_num = fee_cfg.get("numerator", fee_num)
        fee_denom = fee_cfg.get("denominator", fee_denom)
        invite_num = invite_cfg.get("numerator", invite_num)
        invite_denom = invite_cfg.get("denominator", invite_denom)
        drop_after_dummies = drop_after_dummies or fe.get("drop_after_dummies", [])
    else:
        drop_columns = drop_columns or ["invite_tier", "fee_share", "invite"]
        drop_after_dummies = drop_after_dummies or ["commit_Commit"]

    validate_raw_schema(df)

    out = df.copy()

    # 1. Derived features (safe division for scalability when total_fees or deal_size can be 0)
    out["fee_percent"] = _safe_divide(out[fee_num], out[fee_denom])
    out["invite_percent"] = _safe_divide(out[invite_num], out[invite_denom])

    # 2. Drop original columns
    to_drop = [c for c in drop_columns if c in out.columns]
    out = out.drop(columns=to_drop, errors="ignore")
    if to_drop:
        logger.debug("Dropped columns: %s", to_drop)

    # 3. One-hot encode categoricals
    categorical_cols = out.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    if categorical_cols:
        out = pd.get_dummies(
            out, columns=categorical_cols, dtype=float, drop_first=False
        )
        logger.debug("One-hot encoded: %s", categorical_cols)

    # 4. Drop commit_Commit (and any other post-dummy columns)
    for c in drop_after_dummies:
        if c in out.columns:
            out = out.drop(columns=[c])
    logger.info("Feature engineering done: shape=%s", out.shape)
    return out


def get_target_and_features(
    df: pd.DataFrame,
    target_column: str = "commit_Decline",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split engineered DataFrame into features (X) and target (y).

    target_column is the binary target; all other columns are features.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not in DataFrame: {list(df.columns)}"
        )
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def build_feature_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Return a small "pipeline" dict that describes the feature steps for training.

    Same config-driven params as apply_feature_engineering.
    """
    return build_feature_engineering_params(config)
