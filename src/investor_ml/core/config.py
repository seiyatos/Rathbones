"""Load and validate pipeline configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Env vars that override config (production / 12-factor)
_ENV_CONFIG_PREFIX = "INVESTOR_ML_"
_ENV_OVERRIDES = {
    "DATA_RAW_PATH": ("data", "raw_path"),
    "ARTIFACTS_DIR": ("data", "artifacts_dir"),
    "LOG_LEVEL": ("logging", "level"),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "MODEL_SOURCE": ("production", "model_source"),  # "artifacts" | "registry"
    "MODEL_URI": ("production", "model_uri"),        # e.g. "models:/investor_ml/Production"
    "DEFAULT_MODEL_NAME": ("production", "default_model_name"),
}


def get_project_root() -> Path:
    """Resolve project root (directory containing config/ or pyproject.toml)."""
    # core/config.py -> investor_ml/core/ -> need 4 parents to reach project root
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "config" / "config.yaml").exists():
        return candidate
    if (candidate / "pyproject.toml").exists():
        return candidate
    return candidate


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; paths in config are resolved relative to project root.

    Config path can be overridden by env INVESTOR_ML_CONFIG_PATH (production).
    When the package is installed (e.g. in Docker), project root is not next to
    the package, so we try CWD/config/config.yaml if the default path is missing.
    """
    path_str = os.environ.get(_ENV_CONFIG_PREFIX + "CONFIG_PATH") or config_path
    if path_str:
        path = Path(path_str)
        if not path.is_absolute():
            path = get_project_root() / path
    else:
        root = get_project_root()
        path = root / "config" / "config.yaml"
        if not path.exists():
            # Installed package (e.g. Docker): config is typically at /app/config
            path = Path.cwd() / "config" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    root = path.resolve().parent.parent  # config dir's parent = project root

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root
    for key in ("raw_path", "processed_dir", "artifacts_dir", "features_parquet"):
        if key in cfg.get("data", {}):
            val = cfg["data"][key]
            if isinstance(val, str) and not os.path.isabs(val):
                cfg["data"][key] = str(root / val)
    if "feast" in cfg:
        for key in ("repo_path", "online_store_path"):
            if key in cfg["feast"]:
                val = cfg["feast"][key]
                if isinstance(val, str) and not os.path.isabs(val):
                    cfg["feast"][key] = str(root / val)
    if "logging" in cfg and "file" in cfg["logging"]:
        val = cfg["logging"]["file"]
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["logging"]["file"] = str(root / val)
    mlflow_cfg = cfg.get("mlflow")
    if mlflow_cfg and "tracking_uri" in mlflow_cfg:
        val = mlflow_cfg["tracking_uri"]
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["mlflow"]["tracking_uri"] = str(root / val)

    # Apply env overrides (production)
    for env_suffix, key_path in _ENV_OVERRIDES.items():
        env_key = _ENV_CONFIG_PREFIX + env_suffix
        val = os.environ.get(env_key)
        if val is None:
            continue
        if isinstance(key_path, tuple) and len(key_path) == 2:
            section, subkey = key_path
            cfg.setdefault(section, {})[subkey] = val
        else:
            cfg[key_path] = val
    return cfg
