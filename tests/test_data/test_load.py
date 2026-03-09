"""Tests for data loading."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from investor_ml.data.load import load_raw_data


def test_load_raw_data_creates_dataframe(sample_raw_df: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sample_raw_df.to_csv(f.name, index=True)
        df = load_raw_data(f.name)
    assert df is not None
    assert len(df) == len(sample_raw_df)
    assert "investor" in df.columns and "commit" in df.columns


def test_load_raw_data_drops_unnamed_index(sample_raw_df: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sample_raw_df.to_csv(f.name, index=True)
        df = load_raw_data(f.name)
    unnamed = [c for c in df.columns if "Unnamed" in str(c) or c.strip() == ""]
    assert len(unnamed) == 0


def test_load_raw_data_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(Path("/nonexistent/invest.csv"))
