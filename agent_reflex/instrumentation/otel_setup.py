import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_reflex.common.config import Settings

from .redaction import RedactingSpanProcessor


def setup_otel(settings: Settings | None = None) -> trace.Tracer:
    if settings is None:
        settings = Settings()

    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", settings.otel_semconv_opt_in)

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
    })

    provider = TracerProvider(resource=resource)

    if settings.otel_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        if settings.redaction_enabled:
            provider.add_span_processor(RedactingSpanProcessor())
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider.get_tracer("agent_reflex", "0.1.0")
