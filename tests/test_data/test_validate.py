"""Tests for data validation."""

import pandas as pd
import pytest

from investor_ml.data.validate import (
    RAW_REQUIRED_COLUMNS,
    ensure_numeric_positive,
    validate_commit_values,
    validate_raw_schema,
)


def test_validate_raw_schema_pass(sample_raw_df: pd.DataFrame) -> None:
    validate_raw_schema(sample_raw_df)


def test_validate_raw_schema_empty() -> None:
    df = pd.DataFrame(columns=list(RAW_REQUIRED_COLUMNS))
    with pytest.raises(ValueError, match="empty"):
        validate_raw_schema(df)


def test_validate_raw_schema_missing_column(sample_raw_df: pd.DataFrame) -> None:
    sample_raw_df.drop(columns=["fee_share"], inplace=True)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_schema(sample_raw_df)


def test_validate_commit_values_pass(sample_raw_df: pd.DataFrame) -> None:
    validate_commit_values(sample_raw_df, "commit")


def test_validate_commit_values_invalid(sample_raw_df: pd.DataFrame) -> None:
    sample_raw_df = sample_raw_df.copy()
    sample_raw_df.loc[sample_raw_df.index[1], "commit"] = "Maybe"
    with pytest.raises(ValueError, match="invalid values"):
        validate_commit_values(sample_raw_df, "commit")


def test_ensure_numeric_positive_pass(sample_raw_df: pd.DataFrame) -> None:
    ensure_numeric_positive(sample_raw_df, ["deal_size", "invite", "total_fees"], allow_zero=True)


def test_ensure_numeric_positive_negative_fails(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.copy()
    df.loc[df.index[0], "deal_size"] = -1
    with pytest.raises(ValueError, match="must be >= 0"):
        ensure_numeric_positive(df, ["deal_size"], allow_zero=True)


def test_ensure_numeric_positive_non_numeric_fails(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.copy()
    df["deal_size"] = "big"
    with pytest.raises(ValueError, match="must be numeric"):
        ensure_numeric_positive(df, ["deal_size"])
