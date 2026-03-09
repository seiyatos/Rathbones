"""Schema and sanity validation for raw and processed data."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns in raw invest.csv (after dropping unnamed index)
RAW_REQUIRED_COLUMNS = frozenset(
    {
        "investor",
        "commit",
        "deal_size",
        "invite",
        "rating",
        "int_rate",
        "covenants",
        "total_fees",
        "fee_share",
        "prior_tier",
        "invite_tier",
    }
)


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Ensure raw DataFrame has required columns and non-empty.

    Raises ValueError on failure.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    missing = RAW_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    logger.debug("Raw schema validation passed: columns=%s", list(df.columns))


def validate_commit_values(df: pd.DataFrame, column: str = "commit") -> None:
    """Ensure commit column contains only Commit/Decline."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    allowed = {"Commit", "Decline"}
    unique = set(df[column].astype(str).str.strip().unique())
    invalid = unique - allowed
    if invalid:
        raise ValueError(f"Column '{column}' has invalid values: {invalid}")


def ensure_numeric_positive(
    df: pd.DataFrame,
    columns: Sequence[str],
    allow_zero: bool = True,
) -> None:
    """Ensure listed columns are numeric and non-negative (or positive if not allow_zero)."""
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")
        if allow_zero:
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' must be >= 0")
        else:
            if (df[col] <= 0).any():
                raise ValueError(f"Column '{col}' must be > 0")
