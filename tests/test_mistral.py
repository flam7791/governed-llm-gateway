"""Version 0.5: Mistral AI as a provider, and an all-Mistral example configuration."""

import json

import httpx
import pytest

from llmgw.config import GatewayConfig
from llmgw.factory import UnavailableProvider, build_providers
from llmgw.gateway import ChatRequest, Gateway
from llmgw.ledger import Ledger
from llmgw.providers import MistralProvider, OpenAICompatibleProvider, ProviderError

from .conftest import ROOT, FakeProvider, user

EXAMPLE = ROOT / "gateway.mistral.example.json"


def mistral_client(seen: list, status: int = 200):
    def handler(request):
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"message": "Requests rate limit exceeded"})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ],
                    "usage": {"prompt_tokens": 8},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Lisbon"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def example_data() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_mistral_provider_calls_the_public_api_with_a_bearer_key():
    seen = []
    provider = MistralProvider(api_key="secret", client=mistral_client(seen))
    result = provider.complete("mistral-small-latest", user("Capital?"), 50, 0.0)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    assert body["model"] == "mistral-small-latest" and body["max_tokens"] == 50
    assert body["temperature"] == 0.0 and body["stream"] is False
    assert (result.text, result.input_tokens, result.output_tokens) == ("Lisbon", 11, 2)


def test_mistral_embeddings_use_the_same_api_and_keep_input_order():
    seen = []
    provider = MistralProvider(api_key="k", client=mistral_client(seen))
    result = provider.embed("mistral-embed", ["first", "second"])
    assert str(seen[0].url) == "https://api.mistral.ai/v1/embeddings"
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]] and result.input_tokens == 8


def test_a_self_hosted_mistral_deployment_is_reached_at_its_own_address():
    seen = []
    provider = MistralProvider(
        "https://mistral.internal.example/v1", api_key="k", client=mistral_client(seen)
    )
    provider.complete("mistral-small", user("hi"), 10, 0.0)
    assert str(seen[0].url) == "https://mistral.internal.example/v1/chat/completions"


def test_mistral_needs_a_key_and_turns_rate_limits_into_provider_errors():
    with pytest.raises(ValueError, match="API key"):
        MistralProvider(api_key=None)
    provider = MistralProvider(api_key="k", client=mistral_client([], status=429))
    with pytest.raises(ProviderError, match="429"):
        provider.complete("mistral-small-latest", user("hi"), 10, 0.0)


def test_example_config_wires_mistral_for_the_external_tiers_and_ollama_for_local(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    config = GatewayConfig.load(EXAMPLE)
    providers = build_providers(config)
    assert isinstance(providers["mistral-fast"], MistralProvider)
    assert providers["mistral-strong"].api_key == "mistral-key"
    assert providers["mistral-fast"].base_url == MistralProvider.DEFAULT_BASE_URL
    local = providers["local"]
    assert isinstance(local, OpenAICompatibleProvider) and not isinstance(local, MistralProvider)
    assert config.models["local"].model == "ministral-3:8b" and not config.models["local"].external


def test_without_the_key_mistral_models_are_unavailable_so_routing_falls_back(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    providers = build_providers(GatewayConfig.load(EXAMPLE))
    assert isinstance(providers["mistral-fast"], UnavailableProvider)
    assert providers["mistral-fast"].reason == "MISTRAL_API_KEY is not set"


def test_the_key_variable_can_be_named_per_model(monkeypatch):
    data = example_data()
    data["models"][1]["api_key_env"] = "MISTRAL_KEY_RESEARCH"
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("MISTRAL_KEY_RESEARCH", "k2")
    providers = build_providers(GatewayConfig.from_dict(data))
    assert providers["mistral-fast"].api_key == "k2"
    assert isinstance(providers["mistral-strong"], UnavailableProvider)


def test_the_public_mistral_api_cannot_be_declared_internal():
    data = example_data()
    data["models"][1]["external"] = False
    with pytest.raises(ValueError, match="outside the organisation"):
        GatewayConfig.from_dict(data)
    # A self-hosted deployment, at its own address, may be internal.
    data["models"][1]["base_url"] = "https://mistral.internal.example/v1"
    assert not GatewayConfig.from_dict(data).models["mistral-fast"].external


def make_mistral_gateway(tmp_path, monkeypatch, **answers):
    for name in ("RESEARCH", "COMMUNICATIONS", "LEGAL"):
        monkeypatch.setenv(f"LLMGW_KEY_{name}", f"key-{name.lower()}")
    data = example_data()
    data["db_path"] = str(tmp_path / "gateway.db")
    config = GatewayConfig.from_dict(data)
    providers = {alias: answers.get(alias) or FakeProvider(alias) for alias in config.models}
    return Gateway(config, providers, Ledger(config.db_path))


def test_data_policy_beats_cost_with_mistral_models_too(tmp_path, monkeypatch):
    echo = FakeProvider("mistral-fast", answer=lambda m: "I will write to [EMAIL_1].")
    gw = make_mistral_gateway(tmp_path, monkeypatch, **{"mistral-fast": echo})
    teams = gw.config.teams

    # research (mask_pii): personal data is masked before it reaches Mistral's API
    request = ChatRequest(user("Reply to jane@example.org"), model="fast")
    result = gw.chat(teams["research"], request)
    sent = echo.calls[0]["messages"][0]["content"]
    assert "jane@example.org" not in sent and "[EMAIL_1]" in sent
    assert result.text == "I will write to jane@example.org."

    # communications (local_if_pii): the same request stays on the local Ministral model
    request = ChatRequest(user("Mail jo@example.org"), model="fast")
    assert gw.chat(teams["communications"], request).model == "local"

    # legal (local_only): never Mistral's API, even when asking for the strong tier
    result = gw.chat(teams["legal"], ChatRequest(user("Draft a memo"), model="strong"))
    assert result.model == "local" and gw.providers["mistral-strong"].calls == []
