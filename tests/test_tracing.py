"""Tracing: spans for each request and each model call, with no content in them."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llmgw import tracing

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def clear():
    EXPORTER.clear()
    yield


def spans():
    return {s.name: s for s in EXPORTER.get_finished_spans()}


def all_attribute_values():
    return [str(v) for s in EXPORTER.get_finished_spans() for v in s.attributes.values()]


def test_chat_and_model_call_spans(tmp_path):
    from llmgw.gateway import ChatRequest

    from .conftest import make_gateway, user

    gw = make_gateway(tmp_path)
    gw.chat(
        gw.config.teams["research"], ChatRequest(user("What is 2 + 2? secret-phrase"), model="fast")
    )
    found = spans()
    request = found["llmgw.chat"]
    call = next(s for name, s in found.items() if name.startswith("chat "))
    assert call.parent.span_id == request.context.span_id
    assert request.attributes["llmgw.team"] == "research"
    assert request.attributes["llmgw.status"] == "ok"
    assert call.attributes["gen_ai.operation.name"] == "chat"
    assert "gen_ai.usage.input_tokens" in call.attributes
    assert not any("secret-phrase" in v for v in all_attribute_values())  # never content


def test_refusal_marks_the_span_as_error(tmp_path):
    import pytest as _pytest

    from llmgw.gateway import ChatRequest, PolicyRefused

    from .conftest import make_gateway, user

    gw = make_gateway(
        tmp_path, research={"allowed_models": ["claude-fast"], "data_policy": "local_only"}
    )
    with _pytest.raises(PolicyRefused):
        gw.chat(gw.config.teams["research"], ChatRequest(user("hello"), model="fast"))
    request = spans()["llmgw.chat"]
    assert request.status.status_code.name == "ERROR"
    assert request.attributes["llmgw.status"] == "blocked_policy"


def test_configure_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert tracing.configure() is False


def test_trace_context_from_the_caller_is_continued(tmp_path):
    from fastapi.testclient import TestClient

    from llmgw.api import create_app

    from .conftest import make_gateway

    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    client = TestClient(create_app(make_gateway(tmp_path)))
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer key-research",
            "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01",
        },
        json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert format(spans()["llmgw.chat"].context.trace_id, "032x") == trace_id
