"""Load raw investor CSV with basic validation and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw investor data from CSV.

    Handles optional index column (unnamed first column).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    # Drop unnamed index column if present
    if df.columns[0].strip() == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    logger.info("Loaded raw data: path=%s shape=%s", path, df.shape)
    return df
