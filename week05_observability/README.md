# Week 5 - observability

`inferapi` serves a RandomForest on the Bank Marketing dataset.


## Start everything

```bash
docker compose up

# Pause the load generator
docker compose pause loadgen
docker compose unpause loadgen

# Scale the service
#   Note all of the scaled services must be specified at the same time,
#   otherwise they revert back to their original value (typically 1)
docker compose scale loadgen=5 inferapi=3

```

| Service | What it does |
|---|---|
| `trainer` | trains the model, writes it to a shared volume, exits |
| `inferapi` | serves predictions on `:8080` (and other ports), waits for `trainer` to finish first |
| `loadgen` | calls `/v1/predict` continuously, batch sizes vary, some requests are deliberately broken |
| `otel-collector` | receives OTLP from `inferapi`, exposes Prometheus-scrapeable metrics |
| `prometheus` | scrapes the collector every 5s |
| `grafana` | the dashboard |

## Look around

| URL | What you'll see |
|---|---|
| <http://localhost:3000> | the dashboard |
| <http://localhost:8080/healthz> | `{"status": "ok"}` once the model is loaded |
| <http://localhost:8080/version> | model version and build SHA |
| <http://localhost:8080/metrics> | In prometheus mode (without otel collector), returns metrics |

## The Grafana dashboard

Seven panels: request rate, latency (end-to-end vs model-only, same panel so the gap is visible), error ratio, mean confidence, a confidence heatmap, the prediction mix, and the collector's own health.

Give it a minute after startup, metrics export every 5 seconds.

## Switch the metrics backend

`inferapi` can push metrics over OTLP (the default) or serve `/metrics` itself:

```bash
docker compose -f docker-compose.yaml -f docker-compose.prometheus.yaml up
curl -s http://localhost:8080/metrics | grep inferapi_predictions
```

## Clean up

```bash
docker compose down -v
```
