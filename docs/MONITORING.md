# Metrics and drift monitoring

The API can expose **`GET /metrics`** in Prometheus text format when the optional dependency is installed:

```bash
pip install prometheus-client
# or: pip install -e ".[monitoring]"
```

If present, the root response (`GET /`) includes `"metrics": "/metrics"`. You can point your own Prometheus (or other scraper) at `http://your-api:8000/metrics`.

**Example metrics** (when `prometheus_client` is installed):

| Metric | Use |
|--------|-----|
| `investor_ml_predict_requests_total` | Request rate |
| `investor_ml_predict_duration_seconds` | Latency |
| `investor_ml_predictions_total` | Prediction counts by class (for model drift) |
| `investor_ml_predict_probability_decline` | Predicted probability distribution |
| `investor_ml_request_feature` | Feature values on requests (for data drift) |

For **model drift**: track prediction distribution and mean probability over time. For **data drift**: compare feature stats (e.g. mean `deal_size`, `invite`) on incoming requests to your training baseline. Use your own Prometheus/Grafana or other tooling to scrape and visualize.
