"""Tests for FastAPI endpoints."""

import tempfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from investor_ml.web.api import app

client = TestClient(app)


def test_root_returns_service_info() -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service") == "investor-ml"
    assert "health" in data
    assert "ready" in data
    assert "model_source" in data
    assert data["model_source"] in ("artifacts", "registry")


def test_health_returns_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_returns_503_when_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without artifacts (or registry), /ready should return 503."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("INVESTOR_ML_ARTIFACTS_DIR", str(Path(tmp) / "empty_artifacts"))
        r = client.get("/ready")
    assert r.status_code == 503
    assert "model" in r.json().get("detail", "").lower()


def test_predict_requires_deal_ids_or_instances() -> None:
    r = client.post("/predict", json={"deal_ids": None, "instances": None})
    assert r.status_code == 400
    assert "either" in r.json().get("detail", "").lower()


def test_predict_rejects_empty_deal_ids_and_no_instances() -> None:
    r = client.post("/predict", json={"deal_ids": [], "instances": None})
    assert r.status_code == 400
    assert "either" in r.json().get("detail", "").lower()


def test_metrics_endpoint_when_available() -> None:
    """GET /metrics returns 200 and Prometheus text when prometheus_client is installed."""
    r = client.get("/metrics")
    if r.status_code == 404:
        pytest.skip("prometheus_client not installed; /metrics not registered")
    assert r.status_code == 200
    assert "investor_ml" in r.text or "prometheus" in r.text.lower()


def test_ready_and_predict_with_instances_when_model_exists(
    project_root: Path,
    sample_raw_df,
    sample_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With real artifacts from a train run, /ready returns 200 and /predict with instances succeeds."""
    pytest.importorskip("feast", reason="Feast required for training")
    from investor_ml.models.train import train_models

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp).resolve()
        train_models(sample_raw_df, sample_config, artifacts_dir=tmp_path)
        config_path = tmp_path / "config.yaml"
        config = {
            **sample_config,
            "data": {**sample_config["data"], "artifacts_dir": str(tmp_path)},
            "production": {"model_source": "artifacts", "default_model_name": "l1_logistic"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        monkeypatch.setenv("INVESTOR_ML_CONFIG_PATH", str(config_path.resolve()))
        r_ready = client.get("/ready")
        assert r_ready.status_code == 200, (r_ready.status_code, r_ready.json())
        assert r_ready.json().get("model_ready") is True
        r_predict = client.post(
            "/predict",
            json={
                "instances": [
                    {
                        "investor": "Bank A",
                        "deal_size": 300,
                        "invite": 40,
                        "rating": 2,
                        "int_rate": "Market",
                        "covenants": 2,
                        "total_fees": 30,
                        "fee_share": 0.0,
                        "prior_tier": "Participant",
                        "invite_tier": "Bookrunner",
                    }
                ],
                "model_name": "l1_logistic",  # we only trained this candidate; request body overrides Pydantic default "gradient_boosting"
                "return_proba": True,
            },
        )
        assert r_predict.status_code == 200, (r_predict.status_code, r_predict.json())
        data = r_predict.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert data.get("model_source") == "artifacts"
        assert "probabilities_Decline" in data
