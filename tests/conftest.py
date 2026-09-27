"""Fake model providers and a ready-made gateway, so every test runs offline."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from llmgw.config import EXAMPLE_CONFIG, GatewayConfig, hash_key
from llmgw.gateway import Gateway
from llmgw.ledger import Ledger
from llmgw.providers import Completion, ProviderError

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"research": "key-research", "communications": "key-comms", "legal": "key-legal"}


class FakeProvider:
    """Answers with a function of the prompt; records what it was sent; can be made to fail."""

    def __init__(self, name: str, answer=None, fail: bool = False, tokens=(100, 20)):
        self.name = name
        self.answer = answer or (lambda messages: f"answer from {name}")
        self.fail = fail
        self.tokens = tokens
        self.calls: list[dict] = []

    def complete(self, model, messages, max_tokens, temperature) -> Completion:
        self.calls.append({"model": model, "messages": copy.deepcopy(messages)})
        if self.fail:
            raise ProviderError(f"{self.name} is down")
        return Completion(self.answer(messages), *self.tokens, latency_ms=42)


def make_config(tmp_path: Path, **team_overrides) -> GatewayConfig:
    data = copy.deepcopy(EXAMPLE_CONFIG)
    data["db_path"] = str(tmp_path / "gateway.db")
    for team in data["teams"]:
        team["key_hash"] = hash_key(KEYS[team["name"]])
        team.update(team_overrides.get(team["name"], {}))
    return GatewayConfig.from_dict(data)


def make_gateway(tmp_path: Path, providers=None, **team_overrides) -> Gateway:
    config = make_config(tmp_path, **team_overrides)
    providers = providers or {alias: FakeProvider(alias) for alias in config.models}
    return Gateway(config, providers, Ledger(config.db_path))


@pytest.fixture
def gateway(tmp_path) -> Gateway:
    return make_gateway(tmp_path)


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]
