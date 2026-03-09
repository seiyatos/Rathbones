"""FastAPI application: train, evaluate, predict, and Feast feature store over HTTP."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from investor_ml.core.config import get_project_root, load_config

try:
    from scalar_fastapi import get_scalar_api_reference
except ImportError:
    get_scalar_api_reference = None

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Summary, generate_latest

    _PREDICT_REQUESTS = Counter(
        "investor_ml_predict_requests_total",
        "Total predict requests",
        ["model_source", "model_name"],
    )
    _PREDICT_DURATION = Histogram(
        "investor_ml_predict_duration_seconds",
        "Predict request duration in seconds",
        ["model_source", "model_name"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _PREDICTIONS = Counter(
        "investor_ml_predictions_total",
        "Total predictions by predicted class (for drift: track ratio of 0 vs 1 over time)",
        ["model_source", "model_name", "predicted_class"],
    )
    _PREDICT_PROBA = Summary(
        "investor_ml_predict_probability_decline",
        "Predicted probability of Decline (class 1); use for model drift (distribution shift)",
        ["model_source", "model_name"],
    )
    _REQUEST_FEATURE = Summary(
        "investor_ml_request_feature",
        "Input feature value from predict requests (for data drift: compare mean/quantiles to training baseline)",
        ["feature"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# Numeric request fields to record for data-drift metrics (instances only)
_DRIFT_FEATURE_KEYS = ("deal_size", "invite", "rating", "covenants", "total_fees", "fee_share")

app = FastAPI(
    title="Investor ML API",
    description="Train and predict investor commit/decline (Feast feature store is used internally)",
    version="0.1.0",
)


class TrainRequest(BaseModel):
    """Optional overrides for train run (all optional; config defaults used otherwise)."""

    data_path: str | None = Field(None, description="Override raw data CSV path")
    config_path: str | None = Field(None, description="Override config.yaml path")
    artifacts_dir: str | None = Field(None, description="Override artifacts directory")


class PredictRequest(BaseModel):
    """Request body for prediction. Use **either** deal_ids **or** instances, not both.

    - **deal_ids**: Predict for rows already in the Feast online store (from a previous train).
    - **instances**: Predict from raw feature rows (same columns as your CSV, without 'commit').
    """

    deal_ids: list[int] | None = Field(
        None,
        description="deal_id values to look up in Feast online store (use this OR instances)",
    )
    instances: list[dict[str, Any]] | None = Field(
        None,
        description="Raw feature rows: list of objects with keys investor, deal_size, invite, rating, int_rate, covenants, total_fees, fee_share, prior_tier, invite_tier (no 'commit')",
    )
    model_name: str | None = Field(
        "gradient_boosting",
        description="Saved model to use (e.g. gradient_boosting, l1_logistic, random_forest)",
    )
    return_proba: bool = Field(
        True, description="If true, include probability of Decline (class 1)"
    )


def _predict_ui_html() -> str:
    """Load the predict UI HTML: from file next to this module, or embedded fallback."""
    path = Path(__file__).resolve().parent / "static" / "predict_ui.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        from importlib.resources import files
        return (
            (files("investor_ml.web") / "static" / "predict_ui.html").read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return _PREDICT_UI_HTML_FALLBACK


# Minimal fallback if static file is missing (e.g. in Docker without package_data)
_PREDICT_UI_HTML_FALLBACK = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RathBones</title></head>
<body style="font-family:sans-serif;max-width:520px;margin:2rem auto;padding:1rem;">
<h1>RathBones</h1>
<p>Investor commit / decline prediction</p>
<form id="f"><label>Deal size <input type="number" id="deal_size" value="100"></label>
<label> Invite <input type="number" id="invite" value="10"></label>
<label> Rating <input type="number" id="rating" value="2"></label>
<label> Int rate <input type="number" id="int_rate" value="5" step="0.1"></label>
<label> Covenants <input type="number" id="covenants" value="1"></label>
<label> Total fees <input type="number" id="total_fees" value="20"></label>
<label> Fee share <input type="number" id="fee_share" value="5"></label>
<label> Prior tier <input type="number" id="prior_tier" value="1"></label>
<label> Invite tier <input type="number" id="invite_tier" value="1"></label>
<button type="submit">Predict</button></form>
<div id="out"></div>
<a href="/docs">API docs</a>
<script>
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
instances:[{investor:'A',deal_size:+document.getElementById('deal_size').value,invite:+document.getElementById('invite').value,rating:+document.getElementById('rating').value,int_rate:+document.getElementById('int_rate').value,covenants:+document.getElementById('covenants').value,total_fees:+document.getElementById('total_fees').value,fee_share:+document.getElementById('fee_share').value,prior_tier:+document.getElementById('prior_tier').value,invite_tier:+document.getElementById('invite_tier').value}],return_proba:true})});
const d=await r.json();
document.getElementById('out').innerHTML=r.ok?('<p><b>'+(d.predictions[0]==1?'Decline':'Commit')+'</b>'+(d.probabilities_Decline?' P(Decline)='+(d.probabilities_Decline[0]*100).toFixed(1)+'%':'')+'</p>'):('<p style="color:red">'+d.detail+'</p>');
};
</script>
</body></html>"""


@app.get("/app", response_class=HTMLResponse)
def predict_ui() -> str:
    """Simple UI to run a prediction (Commit vs Decline)."""
    return _predict_ui_html()


@app.get("/")
def root() -> dict[str, Any]:
    """Health and API info. Includes model_source so you can see if serving from registry or artifacts."""
    prod = _get_serving_config().get("production", {})
    source = prod.get("model_source", "artifacts")
    out: dict[str, Any] = {
        "service": "investor-ml",
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "model_source": source,
    }
    if source == "registry":
        out["model_uri"] = prod.get("model_uri") or os.environ.get("INVESTOR_ML_MODEL_URI") or ""
    if get_scalar_api_reference is not None:
        out["scalar"] = "/scalar"
    if _PROMETHEUS_AVAILABLE:
        out["metrics"] = "/metrics"
    return out


if get_scalar_api_reference is not None:

    @app.get("/scalar", include_in_schema=False)
    async def scalar_api_reference() -> HTMLResponse:
        """Scalar API reference (modern alternative to /docs)."""
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title + " - API Reference",
        )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: service is up."""
    return {"status": "ok"}


if _PROMETHEUS_AVAILABLE:

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus metrics for drift and latency (scrape this in Prometheus)."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


def _record_predict_metrics(
    model_source: str,
    model_name: str,
    duration_seconds: float,
    predictions: list[int],
    probabilities_decline: list[float] | None = None,
) -> None:
    """Record Prometheus metrics for predict (drift and latency). No-op if prometheus_client not installed."""
    if not _PROMETHEUS_AVAILABLE:
        return
    _PREDICT_REQUESTS.labels(model_source=model_source, model_name=model_name).inc()
    _PREDICT_DURATION.labels(model_source=model_source, model_name=model_name).observe(duration_seconds)
    for p in predictions:
        _PREDICTIONS.labels(
            model_source=model_source,
            model_name=model_name,
            predicted_class=str(p),
        ).inc()
    if probabilities_decline:
        for prob in probabilities_decline:
            _PREDICT_PROBA.labels(model_source=model_source, model_name=model_name).observe(prob)


def _record_data_drift_metrics(instances: list[dict[str, Any]]) -> None:
    """Record numeric input feature values for data-drift monitoring (instances only)."""
    if not _PROMETHEUS_AVAILABLE:
        return
    for row in instances:
        for key in _DRIFT_FEATURE_KEYS:
            val = row.get(key)
            if isinstance(val, (int, float)):
                _REQUEST_FEATURE.labels(feature=key).observe(float(val))


def _get_serving_config() -> dict[str, Any]:
    """Config with env overrides applied (production)."""
    return load_config(os.environ.get("INVESTOR_ML_CONFIG_PATH"))


def _get_app_root() -> Path:
    """Root directory for the app (parent of config/). In Docker this is /app."""
    config_path = os.environ.get("INVESTOR_ML_CONFIG_PATH")
    if config_path:
        p = Path(config_path).resolve()
        if p.exists():
            return p.parent.parent  # config file -> config/ -> app root
    return get_project_root()


def _load_model_for_serving(model_name: str) -> tuple[Any, list[str], str]:
    """Load pipeline and feature names from artifacts dir or MLflow registry. Returns (pipeline, feature_names_in, model_name)."""
    config = _get_serving_config()
    prod = config.get("production", {})
    source = prod.get("model_source", "artifacts")
    if source == "registry":
        uri = prod.get("model_uri") or os.environ.get("INVESTOR_ML_MODEL_URI")
        if not uri:
            raise ValueError("production.model_uri or INVESTOR_ML_MODEL_URI required when model_source=registry")
        from investor_ml.tracking.mlflow_tracking import setup_mlflow
        setup_mlflow(config, project_root=get_project_root())
        import mlflow
        pipeline = mlflow.sklearn.load_model(uri)
        feature_names_in = list(getattr(pipeline, "feature_names_in_", []))
        return pipeline, feature_names_in, model_name or "registry"
    # Default: load from artifacts dir (resolve relative paths from app root, e.g. /app in Docker)
    from investor_ml.models.train import load_trained_artifact
    artifacts_dir = os.environ.get("INVESTOR_ML_ARTIFACTS_DIR")
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
    else:
        data_cfg = config.get("data", {})
        artifacts_dir = Path(data_cfg.get("artifacts_dir", "artifacts"))
        if not artifacts_dir.is_absolute():
            artifacts_dir = _get_app_root() / artifacts_dir
    artifact = load_trained_artifact(artifacts_dir, model_name)
    pipeline = artifact["pipeline"]
    feature_names_in = list(artifact.get("feature_names_in", []))
    return pipeline, feature_names_in, model_name


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness: model loadable and ready to serve predictions. Returns 503 if not ready.

    Response includes model_source ('registry' or 'artifacts') and model_uri when using registry.
    """
    config = _get_serving_config()
    prod = config.get("production", {})
    default_name = (
        prod.get("default_model_name")
        or os.environ.get("INVESTOR_ML_DEFAULT_MODEL_NAME", "gradient_boosting")
    )
    source = prod.get("model_source", "artifacts")
    try:
        _load_model_for_serving(default_name)
        out: dict[str, Any] = {
            "status": "ok",
            "model_ready": True,
            "default_model": default_name,
            "model_source": source,
        }
        if source == "registry":
            out["model_uri"] = prod.get("model_uri") or os.environ.get("INVESTOR_ML_MODEL_URI") or ""
        return out
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e!s}") from e


def _serve_train_enabled() -> bool:
    """Return True if train endpoint is enabled (disabled when INVESTOR_ML_SERVE_PREDICT_ONLY is set)."""
    return os.environ.get("INVESTOR_ML_SERVE_PREDICT_ONLY", "").lower() not in ("1", "true", "yes")


if _serve_train_enabled():

    @app.post("/train")
    def train(request: TrainRequest | None = None) -> dict[str, Any]:
        """Run full train + evaluate pipeline. Returns comparison of models by test AUROC.

        Uses config defaults unless overrides provided in request body.
        """
        try:
            from investor_ml.pipeline.run import run_train_evaluate_pipeline
        except ImportError as e:
            raise HTTPException(
                status_code=500, detail=f"Pipeline import error: {e!s}"
            ) from e

        req = request or TrainRequest()
        try:
            out = run_train_evaluate_pipeline(
                config_path=req.config_path,
                data_path=req.data_path,
                artifacts_dir=req.artifacts_dir,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {type(e).__name__}: {e!s}",
            ) from e

        comparison = out["comparison_df"]
        return {
            "comparison": comparison[["model", "test_roc_auc", "cv_best_score"]].to_dict(
                orient="records"
            ),
            "message": "Training and evaluation complete.",
        }


def _predict_from_instances(
    instances: list[dict[str, Any]],
    pipeline: Any,
    feature_names_in: list[str],
    config: dict[str, Any],
    return_proba: bool,
) -> dict[str, Any]:
    """Run feature engineering on raw instances and predict. Aligns columns to training schema."""
    from investor_ml.features.engineering import apply_feature_engineering

    df_raw = pd.DataFrame(instances)
    if df_raw.empty:
        return {
            "predictions": [],
            "probabilities_Decline": [] if return_proba else None,
        }
    if "commit" not in df_raw.columns:
        df_raw["commit"] = "Commit"
    df_eng = apply_feature_engineering(df_raw, config)
    target_col = config.get("feature_engineering", {}).get(
        "target_column", "commit_Decline"
    )
    if target_col in df_eng.columns:
        df_eng = df_eng.drop(columns=[target_col])
    X = df_eng.reindex(columns=feature_names_in, fill_value=0.0)
    preds = pipeline.predict(X)
    out: dict[str, Any] = {"predictions": [int(p) for p in preds], "model_name": None}
    if return_proba:
        proba = pipeline.predict_proba(X)
        out["probabilities_Decline"] = [
            float(proba[i, 1]) for i in range(len(instances))
        ]
    return out


def _online_features_to_dataframe(
    features_dict: dict[str, Any], feature_names_in: list[str]
) -> pd.DataFrame:
    """Build a DataFrame from Feast online features dict with columns matching feature_names_in."""
    key_lower = {k.lower(): k for k in features_dict}
    col_to_key = {}
    for col in feature_names_in:
        c = col.lower()
        if c in key_lower:
            col_to_key[col] = key_lower[c]
        elif f"investor_features__{c}" in key_lower:
            col_to_key[col] = key_lower[f"investor_features__{c}"]
        else:
            for k in features_dict:
                if k.endswith("__" + col) or k == col:
                    col_to_key[col] = k
                    break
    if len(col_to_key) != len(feature_names_in):
        missing = set(feature_names_in) - set(col_to_key.keys())
        raise ValueError(f"Online features missing columns: {missing}")
    return pd.DataFrame(
        {col: features_dict[col_to_key[col]] for col in feature_names_in}
    )


@app.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    """Predict Commit (0) or Decline (1). Use either.

    - **deal_ids**: Look up features from Feast online store (rows that were in the last train).
    - **instances**: Send raw feature rows (same columns as your CSV except 'commit'). Features are
      engineered the same way as in training, then the model predicts.
    """
    start = time.perf_counter()
    use_instances = bool(body.instances)
    use_deal_ids = bool(body.deal_ids)

    if not use_instances and not use_deal_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'deal_ids' (look up in Feast) or 'instances' (raw feature rows).",
        )
    if use_instances and use_deal_ids:
        use_deal_ids = False

    config = _get_serving_config()
    model_name = (
        body.model_name
        or config.get("production", {}).get("default_model_name")
        or os.environ.get("INVESTOR_ML_DEFAULT_MODEL_NAME", "gradient_boosting")
    )
    try:
        pipeline, feature_names_in, _ = _load_model_for_serving(model_name)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Run POST /train first. (Looked for: {e!s})",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    prod = config.get("production", {})
    source = prod.get("model_source", "artifacts")
    model_uri = (prod.get("model_uri") or os.environ.get("INVESTOR_ML_MODEL_URI")) if source == "registry" else None

    if use_instances:
        _record_data_drift_metrics(body.instances or [])
        result = _predict_from_instances(
            body.instances,
            pipeline,
            feature_names_in,
            config,
            body.return_proba,
        )
        result["model_name"] = model_name
        result["model_source"] = source
        if model_uri:
            result["model_uri"] = model_uri
        _record_predict_metrics(
            source,
            model_name,
            time.perf_counter() - start,
            result["predictions"],
            result.get("probabilities_Decline"),
        )
        return result

    try:
        from investor_ml.store.feast_store import get_feature_store, get_online_features
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Import error: {e!s}") from e

    feast_cfg = config.get("feast", {})
    repo_path = feast_cfg.get("repo_path", "feature_repo")
    if not Path(repo_path).is_absolute():
        repo_path = str(get_project_root() / repo_path)
    try:
        store = get_feature_store(repo_path)
        features_dict = get_online_features(
            store,
            entity_ids=body.deal_ids or [],
            feature_view_name=feast_cfg.get("feature_view_name", "investor_features"),
            entity_id_column=feast_cfg.get("entity_id_column", "deal_id"),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    target_col = config.get("feature_engineering", {}).get(
        "target_column", "commit_Decline"
    )
    skip = {"deal_id", "event_timestamp", target_col}
    if not feature_names_in:
        feature_names_in = [k for k in features_dict if k not in skip]
    X = _online_features_to_dataframe(features_dict, feature_names_in)

    preds = pipeline.predict(X)
    out = {
        "deal_ids": body.deal_ids,
        "predictions": [int(p) for p in preds],
        "model_name": model_name,
        "model_source": source,
    }
    if model_uri:
        out["model_uri"] = model_uri
    if body.return_proba:
        proba = pipeline.predict_proba(X)
        out["probabilities_Decline"] = [
            float(proba[i, 1]) for i in range(len(body.deal_ids or []))
        ]
    _record_predict_metrics(
        source,
        model_name,
        time.perf_counter() - start,
        out["predictions"],
        out.get("probabilities_Decline"),
    )
    return out
