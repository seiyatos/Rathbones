# RathBones — Architecture

High-level architecture for the investor commit/decline ML pipeline (FastAPI, Feast, sklearn, MLflow).

---

## System context

```mermaid
flowchart LR
    subgraph Clients
        API[HTTP Client]
    end
    subgraph RathBones["RathBones ML"]
        FastAPI[FastAPI]
    end
    subgraph Storage["Storage & external"]
        Config[config.yaml]
        Raw[data/raw CSV]
        FS[Feast store]
        Art[artifacts]
        MLflow[mlruns]
        Logs[logs/]
    end
    API --> FastAPI
    FastAPI --> Config
    FastAPI --> Raw
    FastAPI --> FS
    FastAPI --> Art
    FastAPI --> MLflow
    FastAPI --> Logs
```

---

## Train pipeline (POST /train)

```mermaid
flowchart TB
    subgraph Input
        Req[TrainRequest]
        Config[config/config.yaml]
    end
    subgraph Load["Load & validate"]
        LoadCSV[load_raw_data]
        Val[validate_raw_schema]
    end
    subgraph Features["Feature engineering"]
        FE[apply_feature_engineering]
    end
    subgraph FeastOffline["Feast (offline)"]
        Ingest[ingest_from_dataframe → Parquet]
        Apply[apply_feast_repo]
        Hist[get_historical_features]
    end
    subgraph Train["Training"]
        Split[train_test_split]
        GS[GridSearchCV: L1 / RF / GB]
        Eval[compare_models_auroc]
    end
    subgraph Output["Output"]
        Joblib[joblib: model_*.joblib, test_set.joblib]
        Mat[materialize_to_online]
        MLflow[log_training_run]
        FileLog[logs/app.log]
    end

    Req --> Config
    Config --> LoadCSV
    LoadCSV --> Val
    Val --> FE
    FE --> Ingest
    Ingest --> Apply
    Apply --> Hist
    Hist --> Split
    Split --> GS
    GS --> Eval
    Eval --> Joblib
    GS --> Mat
    Eval --> MLflow
    Config --> FileLog
```

---

## Predict (POST /predict)

Two paths: **by deal_ids** (Feast online) or **by instances** (raw rows).

```mermaid
flowchart TB
    Req[PredictRequest]
    Req --> Choice{deal_ids or instances?}
    Choice -->|deal_ids| Online[get_online_features]
    Choice -->|instances| Raw[Raw instances]
    Online --> Align1[align to feature_names_in]
    Raw --> FE[apply_feature_engineering]
    FE --> Drop[drop target, reindex]
    Drop --> Align2[align to feature_names_in]
    Align1 --> Load[load_trained_artifact]
    Align2 --> Load
    Load --> Predict[pipeline.predict / predict_proba]
    Predict --> Resp[JSON: predictions, probabilities_Decline]
```

---

## Component map

| Layer        | Component            | Role |
|-------------|----------------------|------|
| **Web**     | `web/api.py`         | FastAPI: `/`, `/health`, `POST /train`, `POST /predict` |
| **Pipeline**| `pipeline/run.py`    | `run_train_evaluate_pipeline`, `setup_logging` |
| **Core**    | `core/config.py`    | `load_config`, `get_project_root` |
| **Data**    | `data/load.py`       | Load raw CSV |
| **Data**    | `data/validate.py`  | Schema & commit validation |
| **Features**| `features/engineering.py` | Derived features, drop, one-hot, target split |
| **Store**   | `store/feast_store.py` | Ingest, apply repo, historical/online features, materialize |
| **Models**  | `models/train.py`   | GridSearchCV, joblib artifacts |
| **Models**  | `models/evaluate.py`| AUROC from probabilities, compare_models_auroc |
| **Tracking**| `tracking/mlflow_tracking.py` | setup_mlflow, log_training_run |

---

## Data flow (training)

```mermaid
flowchart LR
    A[data/raw/*.csv] --> B[Feature engineering]
    B --> C[data/processed/investor_features.parquet]
    C --> D[Feast offline view]
    D --> E[get_historical_features]
    E --> F[Train/Test split]
    F --> G[GridSearchCV]
    G --> H[artifacts/model_*.joblib]
    G --> I[comparison.csv]
    D --> J[Feast online store]
    I --> K[mlruns/]
    H --> K
```

---

## Directory layout (relevant)

```
RathBones/
├── config/
│   └── config.yaml          # data paths, feast, models, tuning, logging, mlflow
├── feature_repo/            # Feast repo (entity + feature view → Parquet)
│   ├── feature_store.yaml
│   └── features.py
├── data/
│   ├── raw/                 # Input CSV
│   └── processed/           # investor_features.parquet (Feast source)
├── artifacts/               # model_*.joblib, test_set.joblib, comparison.csv
├── mlruns/                  # MLflow tracking (file backend)
├── logs/
│   └── app.log              # File logging
└── src/investor_ml/
    ├── core/                # Config (get_project_root, load_config)
    ├── web/                 # FastAPI (api.py)
    ├── store/               # Feast feature store
    ├── tracking/            # MLflow tracking
    ├── data/
    ├── features/
    ├── models/
    └── pipeline/
```

---

*Generated for RathBones ML. View Mermaid diagrams in GitHub, VS Code (Mermaid extension), or [mermaid.live](https://mermaid.live).*
