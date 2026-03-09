"""Feast feature store integration: offline (training) and online (serving).

Ingest engineered data into the store, get historical features for training,
materialize to online store, and retrieve online features for inference.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from investor_ml.core.config import get_project_root
from investor_ml.features.engineering import apply_feature_engineering

logger = logging.getLogger(__name__)


def _ensure_feast() -> Any:
    """Lazy import Feast to avoid hard dependency at import time."""
    try:
        from feast import FeatureStore
        return FeatureStore
    except ImportError as e:
        raise ImportError(
            "Feast is required for feature store support. Install with: pip install feast[redis]"
        ) from e


def get_feature_store(repo_path: str | Path | None = None) -> Any:
    """Return a Feast FeatureStore instance. repo_path defaults to project feature_repo."""
    FeatureStore = _ensure_feast()
    path = Path(repo_path) if repo_path else get_project_root() / "feature_repo"
    if not path.is_absolute():
        path = get_project_root() / path
    if not (path / "feature_store.yaml").exists():
        raise FileNotFoundError(f"Feast repo not found: {path} (missing feature_store.yaml)")
    return FeatureStore(repo_path=str(path))


def _add_entity_and_timestamp(
    df: pd.DataFrame,
    entity_id_column: str = "deal_id",
    event_timestamp_column: str = "event_timestamp",
) -> pd.DataFrame:
    """Add deal_id (row index) and event_timestamp for Feast."""
    out = df.copy()
    out[entity_id_column] = range(len(out))
    out[event_timestamp_column] = datetime.now(UTC)
    return out


def ingest_from_dataframe(
    df_engineered: pd.DataFrame,
    parquet_path: str | Path,
    *,
    entity_id_column: str = "deal_id",
    event_timestamp_column: str = "event_timestamp",
) -> Path:
    """Write engineered DataFrame to Parquet with entity and timestamp columns.

    Feast feature view source must point to this path. Returns path written.
    """
    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _add_entity_and_timestamp(
        df_engineered, entity_id_column=entity_id_column, event_timestamp_column=event_timestamp_column
    )
    out.to_parquet(path, index=False)
    logger.info("Ingested features to Parquet: path=%s shape=%s", path, out.shape)
    return path


def apply_feast_repo(store: Any, repo_path: str | Path | None = None) -> None:
    """Apply feature repo (register entity and feature view). Idempotent."""
    raw = repo_path or getattr(store, "repo_path", None) or "feature_repo"
    path = Path(raw)
    if not path.is_absolute():
        path = (get_project_root() / path).resolve()
    features_py = path / "features.py"
    if not features_py.exists():
        raise FileNotFoundError(f"Feature repo features.py not found: {features_py}")
    spec = importlib.util.spec_from_file_location("feast_repo_features", features_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {features_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    objects = [mod.deal_entity, mod.investor_feature_view]
    store.apply(objects)
    logger.info("Applied Feast repo (entity + feature view)")


def _entity_and_timestamp_columns(fv: Any) -> set[str]:
    """Column names to exclude from feature refs (entity keys + timestamp)."""
    skip: set[str] = {"deal_id", "event_timestamp"}
    for entity in getattr(fv, "entities", []) or []:
        for key in getattr(entity, "join_keys", []) or []:
            if isinstance(key, str):
                skip.add(key)
    ts_field = getattr(
        getattr(fv, "source", None) or getattr(fv, "batch_source", None),
        "timestamp_field",
        None,
    )
    if ts_field:
        skip.add(ts_field)
    return skip


def _feature_refs_for_view(store: Any, feature_view_name: str) -> list[str]:
    """Build list of feature ref strings for get_historical_features.

    Excludes entity keys and timestamp (they are not features in the projection).
    """
    fv = store.get_feature_view(feature_view_name)
    skip = _entity_and_timestamp_columns(fv)

    if getattr(fv, "schema", None):
        names = [f.name for f in fv.schema if f.name not in skip]
        if names:
            return [f"{feature_view_name}:{n}" for n in names]
    if getattr(fv, "features", None):
        names = [f.name for f in fv.features if f.name not in skip]
        if names:
            return [f"{feature_view_name}:{n}" for n in names]
    source = getattr(fv, "source", None) or getattr(fv, "batch_source", None)
    path = getattr(source, "path", None) if source else None
    if path:
        path = Path(path)
        if not path.is_absolute():
            path = (get_project_root() / path).resolve()
        if path.exists():
            head = pd.read_parquet(path, nrows=0)
            names = [c for c in head.columns if c not in skip]
            if names:
                return [f"{feature_view_name}:{n}" for n in names]
    raise ValueError(
        f"Cannot infer feature refs for view '{feature_view_name}'. "
        "Define schema on the feature view or ensure source path exists."
    )


def get_historical_features(
    store: Any,
    entity_df: pd.DataFrame,
    feature_view_name: str = "investor_features",
    *,
    entity_id_column: str = "deal_id",
    event_timestamp_column: str = "event_timestamp",
  ) -> pd.DataFrame:
    """Get historical features for training.

    entity_df must contain deal_id and event_timestamp.
    Returns a DataFrame with feature columns (and entity/timestamp if present).
    """
    if entity_id_column not in entity_df.columns or event_timestamp_column not in entity_df.columns:
        raise ValueError(
            f"entity_df must contain '{entity_id_column}' and '{event_timestamp_column}'"
        )
    feature_refs = _feature_refs_for_view(store, feature_view_name)
    job = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    )
    result = job.to_df()
    logger.info("Retrieved historical features: shape=%s", result.shape)
    return result


def materialize_to_online(
    store: Any,
    feature_view_name: str = "investor_features",
    end_date: datetime | None = None,
  ) -> None:
    """Materialize the feature view into the online store so get_online_features can serve.

    end_date defaults to now (UTC).
    """
    if end_date is None:
        end_date = datetime.now(UTC)
    store.materialize_incremental(end_date=end_date)
    logger.info("Materialized to online store: view=%s end_date=%s", feature_view_name, end_date)


def get_online_features(
    store: Any,
    entity_ids: list[int],
    feature_view_name: str = "investor_features",
    entity_id_column: str = "deal_id",
  ) -> dict[str, Any]:
    """Retrieve latest feature values for given deal_ids from the online store.

    Call materialize_to_online first so data is available. Returns dict of arrays.
    """
    entity_rows = [{entity_id_column: eid} for eid in entity_ids]
    features = [feature_view_name]
    result = store.get_online_features(features=features, entity_rows=entity_rows)
    return result.to_dict()


def run_ingest_and_apply(
    df_raw: pd.DataFrame,
    config: dict[str, Any],
  ) -> tuple[Any, pd.DataFrame]:
    """Run feature engineering, write to Parquet, apply Feast repo.

    Returns (FeatureStore, engineered_df_with_entity_and_ts).
    """
    df_eng = apply_feature_engineering(df_raw, config)
    data_cfg = config.get("data", {})
    feast_cfg = config.get("feast", {})
    parquet_path = data_cfg.get("features_parquet", "data/processed/investor_features.parquet")
    if not Path(parquet_path).is_absolute():
        parquet_path = get_project_root() / parquet_path
    ingest_from_dataframe(
        df_eng,
        parquet_path,
        entity_id_column=feast_cfg.get("entity_id_column", "deal_id"),
        event_timestamp_column=feast_cfg.get("event_timestamp_column", "event_timestamp"),
    )
    repo_path = feast_cfg.get("repo_path", "feature_repo")
    store = get_feature_store(repo_path)
    apply_feast_repo(store, repo_path=repo_path)
    df_with_meta = _add_entity_and_timestamp(
        df_eng,
        entity_id_column=feast_cfg.get("entity_id_column", "deal_id"),
        event_timestamp_column=feast_cfg.get("event_timestamp_column", "event_timestamp"),
    )
    return store, df_with_meta


def get_training_data_from_feast(
    store: Any,
    entity_df: pd.DataFrame,
    config: dict[str, Any],
  ) -> tuple[pd.DataFrame, pd.Series]:
    """Get historical features from Feast and split into X (features) and y (target).

    entity_df must have deal_id and event_timestamp. Target column from config.
    """
    feast_cfg = config.get("feast", {})
    target_column = config.get("feature_engineering", {}).get("target_column", "commit_Decline")
    view_name = feast_cfg.get("feature_view_name", "investor_features")
    hist = get_historical_features(
        store,
        entity_df,
        feature_view_name=view_name,
        entity_id_column=feast_cfg.get("entity_id_column", "deal_id"),
        event_timestamp_column=feast_cfg.get("event_timestamp_column", "event_timestamp"),
    )
    if target_column not in hist.columns:
        raise ValueError(f"Target '{target_column}' not in Feast output: {list(hist.columns)}")
    y = hist[target_column]
    drop_cols = [
        c for c in hist.columns
        if c in (target_column, "deal_id", "event_timestamp")
        or c.startswith("event_timestamp")
    ]
    X = hist.drop(columns=[c for c in drop_cols if c in hist.columns], errors="ignore")
    return X, y
