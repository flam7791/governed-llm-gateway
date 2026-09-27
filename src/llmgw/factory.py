"""Build providers and the gateway from configuration: the only place real services are wired.

Provider secrets come from environment variables (in a deployment, injected from a secret store):
- anthropic:          ANTHROPIC_API_KEY
- azure_openai:       the model's api_key_env (default AZURE_OPENAI_API_KEY), or, with
                      "auth": "entra_id", a Microsoft Entra ID token from the managed identity
                      (needs the optional azure-identity package: pip install ".[azure]")
- openai_compatible:  the model's api_key_env (default OPENAI_COMPATIBLE_API_KEY), often none
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import GatewayConfig, ModelSpec
from .gateway import Gateway
from .ledger import Ledger
from .providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderError,
    RecordingProvider,
)


class UnavailableProvider:
    """Stands in for a provider that is not configured, so routing falls back cleanly."""

    def __init__(self, reason: str):
        self.reason = reason

    def complete(self, model, messages, max_tokens, temperature):
        raise ProviderError(self.reason)

    def embed(self, model, texts):
        raise ProviderError(self.reason)


def entra_token_provider():  # pragma: no cover - needs Azure credentials
    """A Microsoft Entra ID token provider (managed identity, Azure CLI login, environment)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    scope = os.environ.get("AZURE_OPENAI_TOKEN_SCOPE", AzureOpenAIProvider.ENTRA_SCOPE)
    return get_bearer_token_provider(DefaultAzureCredential(), scope)


def _azure(spec: ModelSpec, token_provider_factory):
    if spec.auth == "entra_id":
        try:
            return AzureOpenAIProvider(spec.base_url, token_provider=token_provider_factory())
        except ImportError:
            return UnavailableProvider('Entra ID auth needs azure-identity: pip install ".[azure]"')
    key_env = spec.api_key_env or "AZURE_OPENAI_API_KEY"
    if not os.environ.get(key_env):
        return UnavailableProvider(f"{key_env} is not set")
    return AzureOpenAIProvider(spec.base_url, api_key=os.environ[key_env])


def build_providers(
    config: GatewayConfig,
    recordings: Path | None = None,
    offline: bool = False,
    token_provider_factory=entra_token_provider,
) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    anthropic = None
    for alias, spec in config.models.items():
        inner: Provider | None
        if offline:
            inner = None
        elif spec.provider == "anthropic":
            if os.environ.get("ANTHROPIC_API_KEY"):
                anthropic = anthropic or AnthropicProvider()
                inner = anthropic
            else:
                inner = UnavailableProvider("ANTHROPIC_API_KEY is not set")
        elif spec.provider == "azure_openai":
            inner = _azure(spec, token_provider_factory)
        elif spec.provider == "openai_compatible":
            inner = OpenAICompatibleProvider(
                spec.base_url or "http://localhost:11434/v1",
                api_key=os.environ.get(spec.api_key_env or "OPENAI_COMPATIBLE_API_KEY"),
            )
        else:
            raise ValueError(f"Unknown provider '{spec.provider}' for model {alias}")

        if recordings is not None:
            providers[alias] = RecordingProvider(alias, inner, recordings / alias, offline=offline)
        else:
            providers[alias] = inner
    return providers


def build_gateway(config: GatewayConfig, **provider_options) -> Gateway:
    return Gateway(config, build_providers(config, **provider_options), Ledger(config.db_path))
