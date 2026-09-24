"""Serves the model over an instrumented FastAPI app

GENAI COMMENT:

Endpoints:
  GET  /healthz     liveness/readiness, 200 only once the model is in memory
  GET  /version     model version (from the artifact) and build SHA (from the image)
  POST /v1/predict  single row or batch, one {prediction, confidence, model_version} per row
"""

import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, RootModel

from settings import settings
from telemetry import setup_telemetry

print(settings)

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("inferapi")

HOSTNAME = os.environ.get("HOSTNAME", "unknown")

meter = setup_telemetry()

##############################
# Creating our own instruments
##############################

PREDICTIONS_COUNTER = meter.create_counter(
    "inferapi.predictions", unit="{prediction}", description="Predictions served, by outcome."
)
ERRORS_COUNTER = meter.create_counter(
    "inferapi.errors", unit="{error}", description="Requests that failed, by error type."
)

INFERENCE_DURATION = meter.create_histogram(
    "inferapi.inference.duration",
    unit="s",
    description="Time inside predict_proba only.",
    explicit_bucket_boundaries_advisory=[0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5],
)
CONFIDENCE_HISTOGRAM = meter.create_histogram(
    "inferapi.prediction.confidence",
    unit="1",
    description="Winning-class probability.",
    explicit_bucket_boundaries_advisory=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
BATCH_SIZE_HISTOGRAM = meter.create_histogram(
    "inferapi.prediction.batch_size",
    unit="{row}",
    description="Rows per predict request.",
    explicit_bucket_boundaries_advisory=[1, 2, 5, 10, 25, 50, 100],
)

state: dict[str, Any] = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s", settings.model_path)
    artifact = joblib.load(settings.model_path)
    app.state["pipeline"] = artifact["pipeline"]
    app.state["model_version"] = artifact["version"]
    app.state["feature_columns"] = artifact["feature_columns"]
    logger.info("Model %s loaded, %s features", app.state["model_version"], len(app.state["feature_columns"]))
    yield
    # state["pipeline"] = None


app = FastAPI(title="inferapi", lifespan=lifespan)

# Reads OTEL_SEMCONV_STABILITY_OPT_IN and OTEL_PYTHON_FASTAPI_EXCLUDED_URLS straight
# from the process environment, so those two settings live in docker-compose.yaml
# next to every other pinned value rather than being buried here.
FastAPIInstrumentor().instrument_app(app)


@app.middleware("http")
async def add_served_by_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Served-By"] = HOSTNAME
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    ERRORS_COUNTER.add(1, {"error_type": "internal"})
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


class Prediction(BaseModel):
    prediction: str
    confidence: float
    model_version: str


class PredictRequest(RootModel[list[dict[str, Any]] | dict[str, Any]]):
    """Accepts either one row as a JSON object, or a batch as a JSON array of rows."""


def busy_wait(duration_ms: int) -> None:
    """Burn CPU for a randomized multiple of duration_ms."""
    if duration_ms == 0:
        return

    roll = random.random()
    if roll < 0.80:
        wait_ms = duration_ms * random.uniform(0.1, 0.4)
    elif roll < 0.98:
        wait_ms = duration_ms * random.uniform(0.4, 1.5)
    else:
        wait_ms = duration_ms * random.uniform(1.5, 4.0)

    logger.debug("busy_wait burning %.1f ms", wait_ms)
    deadline = time.perf_counter_ns() + wait_ms * 1000
    while time.perf_counter_ns() < deadline:
        pass


def validate_rows(rows: list[dict[str, Any]], feature_columns: list[str]) -> None:
    required = set(feature_columns)
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"missing features: {sorted(missing)}")


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    if request.app.state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/version")
async def version(request: Request) -> dict[str, str]:
    return {"model_version": request.app.state["model_version"], "build_sha": settings.build_sha}



@app.post("/v1/predict", response_model=list[Prediction])
def predict(request: Request, payload: PredictRequest) -> list[Prediction]:
    rows = payload.root if isinstance(payload.root, list) else [payload.root]
    feature_columns = request.app.state["feature_columns"]

    try:
        validate_rows(rows, feature_columns)
        frame = pd.DataFrame(rows)[feature_columns]
    except (ValueError, KeyError) as exc:
        ERRORS_COUNTER.add(1, {"error_type": "validation"})
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pipeline = request.app.state["pipeline"]
    model_version = request.app.state["model_version"]

    start = time.perf_counter()

    # Burn CPU cycles to simulate different work
    busy_wait(settings.simulate_work_ms)

    try:
        probabilities = pipeline.predict_proba(frame)
    except Exception as exc:
        ERRORS_COUNTER.add(1, {"error_type": "model"})
        raise HTTPException(status_code=500, detail="model inference failed") from exc
    INFERENCE_DURATION.record(time.perf_counter() - start)

    BATCH_SIZE_HISTOGRAM.record(len(rows))

    classes = pipeline.named_steps["classify"].classes_
    results = []
    for probs in probabilities:
        best_index = probs.argmax()
        label = classes[best_index]
        confidence = float(probs[best_index])

        PREDICTIONS_COUNTER.add(1, {"model_version": model_version, "outcome": label})

        # ANTI-EXAMPLE, DO NOT UNCOMMENT.
        # Labeling by request path or by a per-client id turns one series into one per distinct value seen.
        # PREDICTIONS_COUNTER.add(1, {"client_id": request.client.host, "raw_path": request.url.path})

        CONFIDENCE_HISTOGRAM.record(confidence, {"model_version": model_version})
        results.append(Prediction(prediction=label, confidence=confidence, model_version=model_version))

    return results


if settings.metrics_export == "prometheus":
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        # PrometheusMetricReader registered itself into prometheus_client's default REGISTRY.
        # generate_latest() therefore picks up every OTel instrument without being handed the reader.
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
