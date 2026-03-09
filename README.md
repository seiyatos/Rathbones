# Investor ML

Production-grade ML pipeline for predicting investor **commit** vs **decline**. Built for scalability, reproducibility, and maintainability.

## Features

- **Feature engineering**: `fee_percent`, `invite_percent`, one-hot encoding, config-driven
- **Feast feature store**: central to the ML process—training reads from the offline store, serving from the online store (single source of truth for features)
- **Models**: L1/L2 logistic regression, Random Forest, Gradient Boosting
- **Tuning**: `GridSearchCV` with configurable param grids; AUROC as the optimization metric
- **Evaluation**: AUROC computed from **predicted probabilities** (not hard class labels)
- **Artifacts**: Saved pipelines (scaler + model) and test set for audit

## Project layout

```
RathBones/
├── config/
│   └── config.yaml          # Paths, seeds, Feast, model config, tuning
├── feature_repo/            # Feast repo (entity, feature view, feature_store.yaml)
├── data/
│   └── raw/
│       └── invest.csv       # Your data (schema below)
├── src/
│   └── investor_ml/
│       ├── core/            # Config (get_project_root, load_config)
│       ├── web/             # FastAPI app (api.py)
│       ├── store/           # Feast feature store
│       ├── tracking/        # MLflow tracking
│       ├── data/            # Load, validate
│       ├── features/        # Engineering
│       ├── models/          # Train, evaluate
│       └── pipeline/        # End-to-end run
├── tests/
├── artifacts/               # Created at run: model_*.joblib, test_set.joblib
├── requirements.txt
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Data format

CSV with columns (matching your sample):

- `investor`, `commit`, `deal_size`, `invite`, `rating`, `int_rate`, `covenants`, `total_fees`, `fee_share`, `prior_tier`, `invite_tier`

Optional: first column can be an unnamed index (it will be dropped).

## Setup

```bash
cd RathBones
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
# Or: pip install -r requirements.txt && pip install -e .
```

Place your dataset at `data/raw/invest.csv` (or set path in config or request body).

## Docker (run API and use /predict)

You need a trained model before the container can serve predictions. From the **project root**:

**1. Train once** (so `artifacts/` has `model_*.joblib`):

```bash
# Windows (PowerShell)
.\\.venv\Scripts\activate; pip install -e .; python -c "from investor_ml.pipeline.run import run_train_evaluate_pipeline; run_train_evaluate_pipeline()"

# Or start API locally and call train:
# uvicorn investor_ml.web.api:app --host 0.0.0.0 --port 8000
# curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{}"
```

**2. Build the image:**

```bash
docker build -t investor-ml .
```

**3. Run the container** (mount config and artifacts; Windows use `%cd%` in CMD or `${PWD}` in PowerShell):

```bash
# PowerShell
docker run -p 8000:8000 -v "${PWD}/config:/app/config" -v "${PWD}/artifacts:/app/artifacts" investor-ml

# CMD
docker run -p 8000:8000 -v "%cd%/config:/app/config" -v "%cd%/artifacts:/app/artifacts" investor-ml
```

**4. Try the API:**

- **http://localhost:8000** — service info and `model_source`
- **http://localhost:8000/docs** — Swagger UI
- **http://localhost:8000/ready** — readiness (model loaded)

**Predict (raw feature rows):**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"instances\": [{\"investor\": \"A\", \"deal_size\": 100, \"invite\": 10, \"rating\": 2, \"int_rate\": 5, \"covenants\": 1, \"total_fees\": 20, \"fee_share\": 5, \"prior_tier\": 1, \"invite_tier\": 1}], \"return_proba\": true}"
```

If you serve from the **MLflow registry** instead of artifacts, see the Dockerfile header and [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Production

For deployment (env-based config, health/ready, MLflow registry, Docker), see **[docs/PRODUCTION.md](docs/PRODUCTION.md)**. For **AWS EC2 + ECR** (run the API image from ECR), see **[docs/DEPLOY-EC2-ECR.md](docs/DEPLOY-EC2-ECR.md)**.

## MLflow

After `POST /train`, each run is logged under `mlruns/` with params, metrics, and the registered model. From the **project root**, run:

```bash
mlflow ui
```

Then open **http://localhost:5000**. If you run `mlflow ui` from another directory, pass the store: `mlflow ui --backend-store-uri mlruns` (or the full path to your `mlruns` folder).

## Run (FastAPI)

Start the API server:

```bash
uvicorn investor_ml.web.api:app --reload --host 0.0.0.0 --port 8000
# or: make run
```

Then open **http://localhost:8000/docs** for interactive docs.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and links |
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (model loadable); returns 503 if not |
| POST | `/train` | Run full train + evaluate; returns model comparison by test AUROC. Ingest and materialize to Feast are done inside this call. Body (optional): `{"data_path": "...", "config_path": "...", "artifacts_dir": "..."}` |
| POST | `/predict` | Predict Commit (0) / Decline (1). Send either `deal_ids` (look up in Feast) or `instances` (raw feature rows). |

**Example (train):**

{
  "data_path": "data/raw/syndicated_revolver.csv",
  "config_path": "config/config.yaml",
  "artifacts_dir": "artifacts"
}

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{}"
```

**Example (predict by deal_ids):** Use when you want to score rows that were in the last train (identified by index 0, 1, 2, …).

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"deal_ids": [0, 1, 2]}'
```

**Example (predict from raw features):** Use when you have new rows with the same columns as your CSV (no `commit`). The API runs the same feature engineering as in training, then predicts.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "instances": [
    {
      "investor": "Goldman Sachs",
      "deal_size": 300,
      "invite": 40,
      "rating": 2,
      "int_rate": "Market",
      "covenants": 2,
      "total_fees": 30,
      "fee_share": 0.0,
      "prior_tier": "Participant",
      "invite_tier": "Bookrunner"
    }
  ],
  "model_name": "gradient_boosting",
  "return_proba": true
}'
```

Response includes `predictions` (0 = Commit, 1 = Decline) and, if `return_proba: true`, `probabilities_Decline`. Run **POST /train** once before predicting.

**Run train once without server:**

```bash
make train
# or: python -c "from investor_ml.pipeline.run import run_train_evaluate_pipeline; run_train_evaluate_pipeline()"
```

## Feast feature store (core to the ML process)

The feature store is **part of the ML process**, not optional. The pipeline:

1. Runs feature engineering and writes features to `data/processed/investor_features.parquet` (with `deal_id` and `event_timestamp`).
2. Applies the Feast repo under `feature_repo/` (entity `deal_id`, feature view `investor_features`).
3. Gets historical features from Feast for training (offline store).
4. After training, materializes the feature view into the online store (SQLite by default).
5. **POST /predict** reads from the online store for the requested `deal_ids` and returns predictions.

Ingest and materialize are not exposed as separate endpoints; they run inside **POST /train**.

**Config** (`config/config.yaml` → `feast`):

- `repo_path`: Feast repo directory (default `feature_repo`)
- `entity_id_column`, `event_timestamp_column`, `feature_view_name`
- `online_store_type`: `sqlite` (default) or `redis` (install `feast[redis]`)
- Training always uses the feature store; a `feast` section in config is required

**Programmatic:**

```python
from investor_ml.feast_store import (
    get_feature_store,
    get_historical_features,
    get_online_features,
    materialize_to_online,
    run_ingest_and_apply,
)
store = get_feature_store()
# ... run_ingest_and_apply(df, config), get_historical_features(store, entity_df), etc.
```

## Configuration

Edit `config/config.yaml` to change:

- `data.raw_path`, `data.artifacts_dir`
- `splits.test_size`, `seed`
- `feature_engineering.drop_columns`, `drop_after_dummies`, `target_column`
- `models.candidates`: estimator names and `param_grid`
- `tuning.cv`, `tuning.scoring`, `tuning.n_jobs`
- `evaluation.probability_positive_class` (1 = Decline as positive for AUROC)

## Tests and CI

```bash
pip install -e ".[dev]"
ruff check src tests
pytest tests -v
# Coverage report (HTML): make test-cov  →  open htmlcov/index.html
```

**Coverage:** Run `make test-cov` to generate the coverage report; open `htmlcov/index.html` in a browser for the line-by-line coverage doc.

CI runs on push/PR: Ruff and pytest (matrix Python 3.10–3.12).

## Reproducibility

- Fixed `seed` in config
- `train_test_split(..., stratify=commit_Decline)` after one-hot
- All dependencies in `requirements.txt` / `pyproject.toml`
- Artifacts include pipeline and feature names for identical preprocessing at inference

## License

MIT.
