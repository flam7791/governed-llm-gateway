"""Version 0.2: Azure OpenAI, embeddings, metrics and secrets from the environment."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llmgw.api import create_app
from llmgw.config import GatewayConfig, hash_key
from llmgw.factory import UnavailableProvider, build_providers
from llmgw.gateway import (
    BadRequest,
    ChatRequest,
    EmbeddingRequest,
    PolicyRefused,
    UpstreamFailed,
)
from llmgw.ledger import month_of
from llmgw.providers import AzureOpenAIProvider, ProviderError
from llmgw.router import RoutingError, decide

from .conftest import ROOT, FakeProvider, make_config, make_gateway, user

AZURE = "https://example-resource.openai.azure.com/openai/v1"


def azure_client(seen: list):
    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ],
                    "usage": {"prompt_tokens": 6},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Lisbon"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_azure_openai_uses_the_v1_api_with_an_api_key_header():
    seen = []
    provider = AzureOpenAIProvider(AZURE, api_key="secret", client=azure_client(seen))
    result = provider.complete("my-deployment", user("Capital?"), 50, 0.0)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == f"{AZURE}/chat/completions"
    assert request.headers["api-key"] == "secret" and "authorization" not in request.headers
    assert body["model"] == "my-deployment" and body["max_completion_tokens"] == 50
    assert "temperature" not in body and "max_tokens" not in body
    assert (result.text, result.input_tokens, result.output_tokens) == ("Lisbon", 9, 2)


def test_azure_openai_with_entra_id_sends_a_fresh_bearer_token_each_call():
    seen, tokens = [], iter(["token-1", "token-2"])
    provider = AzureOpenAIProvider(
        AZURE, token_provider=lambda: next(tokens), client=azure_client(seen)
    )
    provider.complete("d", user("a"), 10, 0.0)
    provider.complete("d", user("b"), 10, 0.0)
    assert [r.headers["authorization"] for r in seen] == ["Bearer token-1", "Bearer token-2"]
    assert "api-key" not in seen[0].headers


def test_embeddings_are_returned_in_input_order():
    seen = []
    provider = AzureOpenAIProvider(AZURE, api_key="k", client=azure_client(seen))
    result = provider.embed("embed-deployment", ["first", "second"])
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]] and result.input_tokens == 6
    assert json.loads(seen[0].content) == {
        "model": "embed-deployment",
        "input": ["first", "second"],
    }


def test_azure_example_config_is_valid_and_wires_the_right_providers(monkeypatch):
    monkeypatch.setenv("LLMGW_KEY_RESEARCH", "gw-research-from-secret-store")
    config = GatewayConfig.load(ROOT / "gateway.azure.example.json")
    assert config.teams["research"].key_hash == hash_key("gw-research-from-secret-store")
    assert config.teams["legal"].key_hash == ""  # its variable is not set: no key, no access
    providers = build_providers(config, token_provider_factory=lambda: lambda: "entra-token")
    assert isinstance(providers["azure-strong"], AzureOpenAIProvider)
    assert providers["azure-strong"].token_provider() == "entra-token"
    assert isinstance(providers["claude-strong"], UnavailableProvider)  # no Anthropic key here


def test_azure_with_an_api_key_needs_the_variable(tmp_path, monkeypatch):
    data = json.loads((ROOT / "gateway.azure.example.json").read_text(encoding="utf-8"))
    for model in data["models"]:
        model.pop("auth", None)
    config = GatewayConfig.from_dict(data)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert isinstance(build_providers(config)["azure-fast"], UnavailableProvider)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert build_providers(config)["azure-fast"].api_key == "k"


def test_config_rejects_unsafe_or_inconsistent_models():
    data = json.loads((ROOT / "gateway.azure.example.json").read_text(encoding="utf-8"))
    broken = json.loads(json.dumps(data))
    broken["models"][1].pop("base_url")
    with pytest.raises(ValueError, match="base_url"):
        GatewayConfig.from_dict(broken)
    broken = json.loads(json.dumps(data))
    broken["tier_models"]["fast"] = "azure-embed"
    with pytest.raises(ValueError, match="tier_models"):
        GatewayConfig.from_dict(broken)


def test_embedding_models_cannot_be_used_for_chat(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(RoutingError, match="embedding model"):
        decide("local-embed", user("hi"), config.teams["research"], config, False)


def test_embeddings_go_through_the_governed_path(tmp_path):
    gateway = make_gateway(tmp_path)
    research = gateway.config.teams["research"]
    result = gateway.embed(research, EmbeddingRequest(["budget rules", "records"]))
    assert result.model == "local-embed" and result.vectors[0] == [12.0, 1.0]
    rows = gateway.ledger.monthly_report(month_of(gateway.clock()))
    assert [(r["model"], r["requests"], r["input_tokens"]) for r in rows] == [
        ("local-embed", 1, 14)
    ]
    with pytest.raises(BadRequest):
        gateway.embed(research, EmbeddingRequest(["x"], model="claude-fast"))
    with pytest.raises(BadRequest):
        gateway.embed(research, EmbeddingRequest([""]))


def external_embedding_gateway(tmp_path, **team_overrides):
    gateway = make_gateway(tmp_path, **team_overrides)
    spec = gateway.config.models["local-embed"]
    external = type(spec)(**{**spec.__dict__, "alias": "cloud-embed", "external": True})
    gateway.config.models["cloud-embed"] = external
    gateway.providers["cloud-embed"] = FakeProvider("cloud-embed")
    return gateway


def test_external_embeddings_respect_the_data_policy(tmp_path):
    gateway = external_embedding_gateway(
        tmp_path, legal={"allowed_models": ["*"]}, communications={"allowed_models": ["*"]}
    )
    legal = gateway.config.teams["legal"]
    with pytest.raises(PolicyRefused, match="keeps all data local"):
        gateway.embed(legal, EmbeddingRequest(["contract terms"], model="cloud-embed"))
    comms = gateway.config.teams["communications"]  # local_if_pii
    with pytest.raises(PolicyRefused, match="Personal data"):
        gateway.embed(
            comms,
            EmbeddingRequest(["write to jane.doe@example.org"], model="cloud-embed"),
        )


def test_external_embeddings_are_masked_and_never_fall_back(tmp_path):
    gateway = external_embedding_gateway(tmp_path)
    research = gateway.config.teams["research"]
    result = gateway.embed(
        research, EmbeddingRequest(["reply to jane.doe@example.org"], model="cloud-embed")
    )
    sent = gateway.providers["cloud-embed"].calls[-1]["texts"]
    assert sent == ["reply to [EMAIL_1]"] and result.pii_masked == 1
    gateway.providers["cloud-embed"].fail = True
    with pytest.raises(UpstreamFailed, match="do not fall back"):
        gateway.embed(research, EmbeddingRequest(["x"], model="cloud-embed"))
    assert gateway.providers["local-embed"].calls == []


AUTH = {"Authorization": "Bearer key-research"}


def test_embeddings_endpoint_speaks_the_openai_format(tmp_path):
    c = TestClient(create_app(make_gateway(tmp_path)))
    body = c.post("/v1/embeddings", headers=AUTH, json={"input": "hello"}).json()
    assert body["object"] == "list" and body["model"] == "local-embed"
    assert body["data"] == [{"object": "embedding", "index": 0, "embedding": [5.0, 1.0]}]
    assert body["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    assert c.post("/v1/embeddings", json={"input": "x"}).status_code == 401


def test_metrics_count_answers_refusals_and_budget(tmp_path):
    gateway = make_gateway(tmp_path)
    c = TestClient(create_app(gateway))
    c.post("/v1/chat/completions", headers=AUTH, json={"messages": user("Capital of Peru?")})
    gateway.chat(gateway.config.teams["research"], ChatRequest(user("Capital of Peru?")))  # cache
    c.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-legal"},
        json={"model": "claude-strong", "messages": user("hi")},
    )
    text = c.get("/metrics").text
    assert 'llmgw_requests_total{model="claude-fast",status="ok",team="research"} 2.0' in text
    assert 'llmgw_cache_hits_total{model="claude-fast",team="research"} 1.0' in text
    assert 'llmgw_tokens_total{direction="input",model="claude-fast",team="research"} 100.0' in text
    assert 'status="ok",team="legal"' in text  # legal was routed to its local model instead
    assert 'llmgw_budget_remaining_usd{team="research"}' in text
    assert "llmgw_model_latency_seconds_bucket" in text


def test_metrics_can_require_a_monitoring_token(tmp_path):
    c = TestClient(create_app(make_gateway(tmp_path), metrics_token="scrape-secret"))
    assert c.get("/metrics").status_code == 401
    ok = c.get("/metrics", headers={"Authorization": "Bearer scrape-secret"})
    assert ok.status_code == 200 and "llmgw_budget_used_ratio" in ok.text


def test_every_request_is_logged_as_one_json_line_without_content(tmp_path, caplog):
    gateway = make_gateway(tmp_path)
    with caplog.at_level("INFO", logger="llmgw.usage"):
        gateway.chat(gateway.config.teams["research"], ChatRequest(user("My IBAN secret text")))
    line = json.loads(caplog.records[-1].getMessage())
    assert line["team"] == "research" and line["status"] == "ok"
    assert "secret text" not in caplog.text


def test_ledger_path_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMGW_DB_PATH", str(tmp_path / "volume" / "ledger.db"))
    config = GatewayConfig.load(ROOT / "gateway.example.json")
    assert config.db_path == tmp_path / "volume" / "ledger.db"


def test_entra_id_token_failures_become_provider_errors_so_routing_falls_back():
    def no_identity():
        raise RuntimeError("DefaultAzureCredential failed to retrieve a token\nlong details")

    provider = AzureOpenAIProvider(AZURE, token_provider=no_identity, client=azure_client([]))
    with pytest.raises(ProviderError, match="Entra ID token: DefaultAzureCredential failed"):
        provider.complete("d", user("x"), 10, 0.0)


def test_malformed_replies_become_provider_errors():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    provider = AzureOpenAIProvider(AZURE, api_key="k", client=client)
    with pytest.raises(ProviderError, match="unexpected reply"):
        provider.complete("d", user("x"), 10, 0.0)
    bad_json = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))
    )
    with pytest.raises(ProviderError, match="not JSON"):
        AzureOpenAIProvider(AZURE, api_key="k", client=bad_json).embed("e", ["x"])


def test_a_request_for_a_local_model_never_falls_back_outside(tmp_path):
    gateway = make_gateway(tmp_path)
    gateway.providers["local"].fail = True
    research = gateway.config.teams["research"]  # mask_pii: external models allowed
    with pytest.raises(UpstreamFailed):
        gateway.chat(research, ChatRequest(user("hi"), model="local"))
    assert gateway.providers["claude-strong"].calls == []
    assert gateway.providers["claude-fast"].calls == []
