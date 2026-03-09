"""Tests for config loading and path resolution."""

from pathlib import Path

import pytest

from investor_ml.core.config import get_project_root, load_config


def test_get_project_root_has_config(project_root: Path) -> None:
    root = get_project_root()
    assert root is not None
    assert (root / "config" / "config.yaml").exists()


def test_load_config_returns_dict(project_root: Path) -> None:
    cfg = load_config(project_root / "config" / "config.yaml")
    assert isinstance(cfg, dict)
    assert "data" in cfg or "models" in cfg


def test_load_config_resolves_data_paths(project_root: Path) -> None:
    cfg = load_config(project_root / "config" / "config.yaml")
    data = cfg.get("data", {})
    for key in ("raw_path", "artifacts_dir", "features_parquet"):
        if key in data and isinstance(data[key], str):
            assert Path(data[key]).is_absolute(), f"data.{key} should be resolved to absolute path"


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config(Path("/nonexistent/config.yaml"))
