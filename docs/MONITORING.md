# Monitoring model drift, data drift, and using Grafana + Prometheus

This doc describes how to monitor **model drift** and **data drift** for the Investor ML API, and how to use **Prometheus** and **Grafana** for metrics and dashboards.

---

## Model drift vs data drift

| | Model drift | Data drift |
|---|-------------|------------|
| **What** | Model performance degrades over time (e.g. accuracy/AUROC drops). | Input feature distributions change vs the data the model was trained on. |
| **Cause** | Real-world relationships or label distribution changed; model is no longer well-calibrated. | Population shift, seasonality, new segments, or data pipeline changes. |
| **How to monitor** | Track actual outcomes vs predictions (if you get labels later), or track prediction distribution and compare to a baseline. | Compare incoming request feature stats (mean, std, percentiles) to a reference (e.g. training set). |
| **Action** | Retrain or promote a new model; adjust thresholds. | Retrain on newer data; investigate data pipeline; adjust features. |

---

## What to track for drift

### For model drift

- **Prediction distribution over time**  
  e.g. fraction of predictions that are “Decline” (1) vs “Commit” (0). A sustained shift can indicate concept or prior shift.
- **Mean predicted probability**  
  If you use `return_proba=true`, track the average probability of the positive class (Decline) per time window. Compare to baseline from training/validation.
- **Performance when you have labels**  
  If you later get actual outcomes (e.g. from a feedback loop), compute accuracy, AUROC, or calibration over time and alert on drops.

### For data drift

- **Feature statistics on incoming requests**  
  For `instances` (raw features), track per-feature stats: mean, std, min, max, or histograms (e.g. `deal_size`, `invite`, `rating`, `fee_share`). Compare to reference stats from your training data.
- **Drift score (optional)**  
  Use a library (e.g. Evidently, alibi-detect, or custom KS/PSI) to compute a single “drift score” between reference and current window; expose as a metric.

---

## Using Prometheus and Grafana

**Yes, you can use Grafana and Prometheus for this.**

1. **Prometheus** scrapes metrics from your API (e.g. a `/metrics` endpoint in Prometheus text format).
2. **Grafana** connects to Prometheus as a data source and builds dashboards and alerts.

### Flow

```
Investor ML API (/metrics)  -->  Prometheus (scrape)  -->  Grafana (dashboards & alerts)
```

### Metrics the API exposes

**`GET /metrics`** (when `prometheus_client` is installed) exposes:

| Metric | Type | Labels | Use for |
|--------|------|--------|--------|
| `investor_ml_predict_requests_total` | Counter | `model_source`, `model_name` | Request rate by source/model |
| `investor_ml_predict_duration_seconds` | Histogram | `model_source`, `model_name` | Latency (p50, p95, etc.) |
| `investor_ml_predictions_total` | Counter | `model_source`, `model_name`, `predicted_class` | **Model drift**: ratio of class 0 vs 1 over time |
| `investor_ml_predict_probability_decline` | Summary | `model_source`, `model_name` | **Model drift**: distribution of P(Decline); compare to baseline |
| `investor_ml_request_feature` | Summary | `feature` | **Data drift**: input feature values (deal_size, invite, rating, covenants, total_fees, fee_share) from `instances`; compare mean/quantiles to training baseline |

Model drift: use prediction counts and probability summaries over time. Data drift: use `investor_ml_request_feature_sum` / `investor_ml_request_feature_count` (or quantiles) per `feature` and compare to your training statistics.

You then:

1. Point Prometheus at `http://your-api:8000/metrics` (scrape interval e.g. 15s).
2. In Grafana, create dashboards that plot these metrics over time and, if you have a reference, compare current window to baseline.
3. Set alerts (e.g. “mean predicted probability drops below X” or “request rate for class 1 doubles in 1h”) to catch drift.

### Example Prometheus scrape config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "investor-ml"
    static_configs:
      - targets: ["localhost:8000"]   # or your API host
    metrics_path: /metrics
    scrape_interval: 15s
```

### Example Grafana use

- **Data source:** Add Prometheus (e.g. `http://prometheus:9090`).
- **Dashboard panels:**  
  - **Model drift:** `rate(investor_ml_predictions_total{predicted_class="1"}[1h]) / rate(investor_ml_predictions_total[1h])` (fraction Decline); `rate(investor_ml_predict_probability_decline_sum[1h]) / rate(investor_ml_predict_probability_decline_count[1h])` (mean P(Decline)).  
  - **Data drift:** `rate(investor_ml_request_feature_sum{feature="deal_size"}[1h]) / rate(investor_ml_request_feature_count{feature="deal_size"}[1h])` (mean deal_size in requests); same for `invite`, `rating`, `covenants`, `total_fees`, `fee_share`. Compare to training means.  
  - Request rate and latency as above.  
- **Alerts:** Create alert rules when a metric crosses a threshold (e.g. fraction of class 1 or mean probability moves far from baseline; mean request feature moves far from training baseline).

---

## Enabling Prometheus metrics in this project

The API can expose a **`GET /metrics`** endpoint in Prometheus format when the optional dependency is installed:

```bash
pip install prometheus-client
# or: pip install -e ".[monitoring]"
```

If the dependency is present, the root response (`GET /`) will include `"metrics": "/metrics"` and Prometheus can scrape that URL. If not installed, the app runs as before without the endpoint.

See the table above for metric names. When running with `prometheus_client` installed, `GET /` includes `"metrics": "/metrics"`.

---

## Summary

- **Model drift:** Monitor prediction distribution and, when available, actual performance (accuracy/AUROC).  
- **Data drift:** Monitor feature stats (and optionally a drift score) on incoming requests vs a reference.  
- **Grafana + Prometheus:** Use Prometheus to scrape metrics from the API and Grafana to visualize and alert. The optional `/metrics` endpoint provides request counts, latency, and prediction-related metrics to get started.
