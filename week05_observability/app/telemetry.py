"""OpenTelemetry setup for inferapi, in one place, called once at startup."""

import logging

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider, UpDownCounter
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import DropAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from settings import settings

logger = logging.getLogger("inferapi.telemetry")


# GENAI COMMENT
#     The FastAPI instrumentation also emits a request-size and a response-size histogram.
#     No panel in the dashboard reads either one, and a histogram is the most expensive
#     instrument to carry: about 15 series per label set where a counter costs 1.
#     You cannot delete an instrument declared in someone else's package, so dropping its
#     stream is the only way to stop paying for it.
_UNUSED_SIZE_HISTOGRAMS = "http.server.*.body.size"

# An allow-list
_HTTP_KEPT_ATTRIBUTES = {"http.route", "http.request.method", "http.response.status_code"}

VIEWS = [
    View(instrument_name=_UNUSED_SIZE_HISTOGRAMS, aggregation=DropAggregation()),
    # These names only exist under OTEL_SEMCONV_STABILITY_OPT_IN=http, set in docker-compose.yaml.
    View(instrument_name="http.server.request.duration", attribute_keys=_HTTP_KEPT_ATTRIBUTES),
]

# GENAI COMMENT
#   A View can also re-bucket an instrument you do not own, and the spec makes its bounds
#   win over the ones the instrumentation package advises.
#   We deliberately do not, because the HTTP semantic conventions already advise bounds
#   shaped for request latency in seconds, and those are the right ones here.
#
#   from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
#
#   View(
#       instrument_name="http.server.request.duration",
#       aggregation=ExplicitBucketHistogramAggregation(
#           (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)
#       ),
#   )


_CUMULATIVE_FOR_ALL = {
    Counter: AggregationTemporality.CUMULATIVE,
    UpDownCounter: AggregationTemporality.CUMULATIVE,
    Histogram: AggregationTemporality.CUMULATIVE,
}

_METER_NAME = "inferapi"


def setup_telemetry() -> metrics.Meter:
    """Wire the MeterProvider and TracerProvider, return the meter to instrument with."""
    resource = Resource.create({"service.name": "inferapi", "service.version": settings.model_version})

    if settings.metrics_export == "otlp":
        # Reads the OTEL_EXPORTER_OTLP_ENDPOINT
        exporter = OTLPMetricExporter(preferred_temporality=_CUMULATIVE_FOR_ALL)
        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=settings.otel_metric_export_interval
        )
    elif settings.metrics_export == "prometheus":
        reader = PrometheusMetricReader()
    else:
        raise ValueError

    provider = MeterProvider(resource=resource, metric_readers=[reader], views=VIEWS)
    metrics.set_meter_provider(provider)

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    logger.info("Telemetry configured, metrics_export=%s", settings.metrics_export)
    return metrics.get_meter(_METER_NAME)
