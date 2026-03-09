# Production deployment

This document covers running RathBones ML in production: config, health checks, model serving, and deployment options.

---

## Environment-based config (12-factor)

The same image can run in dev and prod by overriding config with environment variables.

| Env var | Overrides | Example |
|--------|-----------|--------|
| `INVESTOR_ML_CONFIG_PATH` | Config file path | `/app/config/production.yaml` |
| `INVESTOR_ML_DATA_RAW_PATH` | `data.raw_path` | `/app/data/raw/train.csv` |
| `INVESTOR_ML_ARTIFACTS_DIR` | `data.artifacts_dir` | `/app/artifacts` |
| `INVESTOR_ML_LOG_LEVEL` | `logging.level` | `INFO` or `WARNING` |
| `INVESTOR_ML_MLFLOW_TRACKING_URI` | `mlflow.tracking_uri` | `file:///app/mlruns` or MLflow server URL |
| `INVESTOR_ML_MODEL_SOURCE` | `production.model_source` | `artifacts` (default) or `registry` |
| `INVESTOR_ML_MODEL_URI` | `production.model_uri` | `models:/investor_ml/Production` |
| `INVESTOR_ML_DEFAULT_MODEL_NAME` | `production.default_model_name` | `gradient_boosting` |

---

## Health and readiness

- **`GET /health`** — Liveness: process is up. Use for “is the container running?”
- **`GET /ready`** — Readiness: default model can be loaded and the service can serve predictions. Returns **503** if the model is missing or load fails. Use for load balancers and Kubernetes readiness probes.

Orchestrators should use **/health** for liveness and **/ready** for readiness so traffic is sent only when the model is loadable.

---

## Model serving: artifacts vs registry

- **Artifacts (default)**  
  Models are read from `data.artifacts_dir` (e.g. `artifacts/model_gradient_boosting.joblib`). After `POST /train`, artifacts are written there and **/ready** passes once those files exist.

- **Registry (production)**  
  Set `INVESTOR_ML_MODEL_SOURCE=registry` and `INVESTOR_ML_MODEL_URI=models:/investor_ml/Production`. The app loads the model from the MLflow Model Registry. You must run training (or a separate job) that registers the best model; then promote that version to the `Production` stage in the MLflow UI or API. **/ready** passes when that model can be loaded from the registry.

Training already **registers the best model** (by test AUROC) under the name `investor_ml` (or `config.mlflow.registry_name`). Promote a version to **Production** when you want the API to serve it.

---

## Serving from the registry (no training in the API)

When the model is **not** trained inside the API container, you train elsewhere and have the API load from the MLflow registry.

**1. Shared MLflow backend**  
Training and the API must use the **same** tracking store (same runs and registry). Options:

- **File store**: Train once (e.g. on your machine or a training job), so `mlruns/` is populated and the best model is registered. When you run the API container, **mount that same `mlruns`** into the container (e.g. `-v "$(pwd)/mlruns:/app/mlruns"`) and set the tracking URI to `file:///app/mlruns`.
- **MLflow server**: Run `mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db` (or PostgreSQL, etc.). Point both training and the API at `http://<server>:5000` via `INVESTOR_ML_MLFLOW_TRACKING_URI`. Training registers the model there; the API loads from the same server.

**2. Train and register (outside the API)**  
Run training where you have data and compute (laptop, CI, separate training container). Use the same tracking URI as the API. After the run, in the MLflow UI (or API) promote a version of `investor_ml` to the **Production** stage.

**3. Run the API with registry mode**  
No need to mount `artifacts/` or run training in the container. Set:

```bash
INVESTOR_ML_MODEL_SOURCE=registry
INVESTOR_ML_MODEL_URI=models:/investor_ml/Production
INVESTOR_ML_MLFLOW_TRACKING_URI=<same as training>
```

Example with file store (shared `mlruns` mount):

```bash
docker run -p 8000:8000 \
  -e INVESTOR_ML_MODEL_SOURCE=registry \
  -e INVESTOR_ML_MODEL_URI=models:/investor_ml/Production \
  -e INVESTOR_ML_MLFLOW_TRACKING_URI=file:///app/mlruns \
  -v "/path/to/mlruns:/app/mlruns" \
  investor-ml
```

Example with MLflow server:

```bash
docker run -p 8000:8000 \
  -e INVESTOR_ML_MODEL_SOURCE=registry \
  -e INVESTOR_ML_MODEL_URI=models:/investor_ml/Production \
  -e INVESTOR_ML_MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  investor-ml
```

The API will load the model from the registry at startup (and for each request, depending on implementation); **/ready** returns 200 only if that load succeeds.

---

## Docker

- Build: `docker build -t investor-ml .`
- Run with host paths for data and config:
  ```bash
  docker run -p 8000:8000 \
    -e INVESTOR_ML_ARTIFACTS_DIR=/app/artifacts \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/artifacts:/app/artifacts" \
    -v "$(pwd)/mlruns:/app/mlruns" \
    investor-ml
  ```
- For production, prefer a multi-worker process (e.g. Gunicorn with Uvicorn workers) and set resource limits; see Dockerfile comments or your orchestrator docs.

---

## Checklist

- [ ] Config and secrets via env or mounted config (no secrets in image).
- [ ] Use **/ready** for readiness; **/health** for liveness.
- [ ] Set `INVESTOR_ML_ARTIFACTS_DIR` (and optionally `INVESTOR_ML_MODEL_URI`) so the same image works across environments.
- [ ] If using the registry, train (or a CI job) registers the best model; promote to `Production` when ready.
- [ ] Logging: set `INVESTOR_ML_LOG_LEVEL` (e.g. `WARNING`) in prod; ensure logs go to stdout or your logging backend.
- [ ] HTTPS and auth in front of the API (reverse proxy or API gateway).
