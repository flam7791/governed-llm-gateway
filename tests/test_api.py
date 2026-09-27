"""The HTTP API, through FastAPI's test client."""

from fastapi.testclient import TestClient

from llmgw.api import create_app

from .conftest import make_gateway

AUTH = {"Authorization": "Bearer key-research"}


def client(tmp_path, **overrides):
    return TestClient(create_app(make_gateway(tmp_path, **overrides)))


def test_missing_key_is_rejected_in_openai_error_format(tmp_path):
    response = client(tmp_path).post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_chat_completion_is_openai_compatible_and_explains_the_route(tmp_path):
    response = client(tmp_path).post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "auto", "messages": [{"role": "user", "content": "Capital of Portugal?"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer from claude-fast",
    }
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert "auto: simple request" in body["gateway"]["route_reason"]
    assert response.headers["x-gateway-model"] == "claude-fast"
    assert float(response.headers["x-gateway-cost-usd"]) > 0


def test_content_parts_are_accepted(tmp_path):
    response = client(tmp_path).post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]},
    )
    assert response.status_code == 200


def test_streaming_is_refused_clearly(tmp_path):
    response = client(tmp_path).post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 400
    assert "Streaming" in response.json()["error"]["message"]


def test_budget_exhaustion_returns_402(tmp_path):
    c = client(tmp_path, communications={"monthly_budget_usd": 0.0001})
    response = c.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-comms"},
        json={"model": "fast", "max_tokens": 1000, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 402
    assert response.json()["error"]["code"] == "budget_exceeded"


def test_models_and_usage_endpoints(tmp_path):
    c = client(tmp_path)
    models = [
        m["id"]
        for m in c.get("/v1/models", headers={"Authorization": "Bearer key-legal"}).json()["data"]
    ]
    assert "local" in models and "claude-strong" not in models
    c.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "strong", "messages": [{"role": "user", "content": "Hi"}]},
    )
    usage = c.get("/v1/usage", headers=AUTH).json()
    assert usage["team"] == "research" and usage["spent_usd"] > 0
    assert c.get("/healthz").json() == {"status": "ok", "version": "0.2.0"}
