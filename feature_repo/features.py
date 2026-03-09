"""Feast feature repo: entity and feature view for investor commit/decline.

Parquet is written by investor_ml.store.feast_store to ../data/processed/
(relative to this repo dir = project root).
"""

from __future__ import annotations

from pathlib import Path

from feast import Entity, FeatureView, FileSource
from feast.value_type import ValueType

# Path to Parquet: relative to feature_repo dir -> project_root/data/processed/...
_REPO_DIR = Path(__file__).resolve().parent
_FEATURES_PARQUET = (
    _REPO_DIR.parent / "data" / "processed" / "investor_features.parquet"
).resolve()

deal_entity = Entity(
    name="deal_id",
    join_keys=["deal_id"],
    value_type=ValueType.INT64,
    description="Unique deal/invitation identifier",
)

# Schema omitted so Feast infers from Parquet (all columns except deal_id, event_timestamp).
investor_feature_view = FeatureView(
    name="investor_features",
    entities=[deal_entity],
    source=FileSource(
        path=str(_FEATURES_PARQUET),
        timestamp_field="event_timestamp",
    ),
    description="Investor deal features for commit/decline prediction",
)
