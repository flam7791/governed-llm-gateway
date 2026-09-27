"""Build providers and the gateway from configuration: the only place real services are wired."""

from __future__ import annotations

import os
from pathlib import Path

from .config import GatewayConfig
from .gateway import Gateway
from .ledger import Ledger
from .providers import (
    AnthropicProvider,
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


def build_providers(
    config: GatewayConfig, recordings: Path | None = None, offline: bool = False
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
        elif spec.provider == "openai_compatible":
            inner = OpenAICompatibleProvider(
                spec.base_url or "http://localhost:11434/v1",
                api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY"),
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
