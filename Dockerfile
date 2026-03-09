# RathBones ML API — Docker image
# Build:  docker build -t investor-ml .
#
# Run (default — needs training or pre-loaded artifacts):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/data:/app/data" -v "$(pwd)/config:/app/config" \
#     -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/mlruns:/app/mlruns" \
#     investor-ml
#
# Run (serve from MLflow registry — no training in container; mount mlruns from where you trained):
#   docker run -p 8000:8000 \
#     -e INVESTOR_ML_MODEL_SOURCE=registry \
#     -e INVESTOR_ML_MODEL_URI=models:/investor_ml/Production \
#     -e INVESTOR_ML_MLFLOW_TRACKING_URI=file:///app/mlruns \
#     -v "/path/to/mlruns:/app/mlruns" \
#     -v "$(pwd)/config:/app/config" \
#     investor-ml
#
# Env (all optional): INVESTOR_ML_CONFIG_PATH, INVESTOR_ML_ARTIFACTS_DIR, INVESTOR_ML_MLFLOW_TRACKING_URI,
#   INVESTOR_ML_MODEL_SOURCE, INVESTOR_ML_MODEL_URI, INVESTOR_ML_DEFAULT_MODEL_NAME, INVESTOR_ML_LOG_LEVEL
# Health: GET /health (liveness), GET /ready (readiness)

FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps and build the package
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --no-deps -w dist dist/*.whl

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Runtime deps (e.g. for Feast/some backends)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install built wheel and production dependencies only
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# App code (config, feature_repo) — override with mounts if needed
COPY config/ config/
COPY feature_repo/ feature_repo/

# Create dirs for mounted or generated data (override at run with -v)
RUN mkdir -p data/raw data/processed artifacts logs mlruns

ENV PYTHONUNBUFFERED=1
# So config is found when running from container (app is in site-packages)
ENV INVESTOR_ML_CONFIG_PATH=/app/config/config.yaml
# Predict-only: omit /train endpoint (set to 0 or unset to enable train)
ENV INVESTOR_ML_SERVE_PREDICT_ONLY=1
EXPOSE 8000

# Single-worker default; for production override with multi-worker (e.g. gunicorn + uvicorn workers)
CMD ["uvicorn", "investor_ml.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
