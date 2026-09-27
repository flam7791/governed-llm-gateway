"""Gateway configuration: models, teams and routing, loaded from one JSON file.

Teams are the unit of access, budget and policy. Each team has:
- an API key, stored only as a SHA-256 hash;
- a monthly budget in USD;
- the models it may use;
- a data policy:
    local_only    never send anything outside (only local models)
    mask_pii      external models allowed; personal data is masked before it leaves
    local_if_pii  external models allowed, but requests containing personal data stay local
- what happens when the budget runs out: "block", or "degrade" to a cheaper model.

Models are chat models (routed by tier) or embedding models (used by /v1/embeddings). Providers:
anthropic (Claude), azure_openai (Azure OpenAI / Microsoft Foundry, v1 API, key or Entra ID) and
openai_compatible (Ollama, vLLM and other OpenAI-format servers).

Secrets never live in this file: provider keys come from environment variables, and a team's
key can be given as an environment variable (key_env) instead of a stored hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_POLICIES = ("local_only", "mask_pii", "local_if_pii")
TIERS = ("local", "fast", "strong")
PROVIDERS = ("anthropic", "azure_openai", "openai_compatible")
KINDS = ("chat", "embedding")


@dataclass(frozen=True)
class ModelSpec:
    alias: str  # the name teams use, e.g. "claude-fast"
    provider: str  # "anthropic", "azure_openai" or "openai_compatible"
    model: str  # the provider's model id (for Azure OpenAI: the deployment name)
    tier: str  # "local", "fast" or "strong" (chat models); "embedding" for embedding models
    price_input: float  # USD per million input tokens
    price_output: float  # USD per million output tokens
    external: bool  # True if data leaves the organisation
    base_url: str | None = None  # openai_compatible and azure_openai endpoints
    kind: str = "chat"  # "chat" or "embedding"
    api_key_env: str | None = None  # environment variable holding the provider key
    auth: str = "api_key"  # azure_openai only: "api_key" or "entra_id" (managed identity)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_input + output_tokens * self.price_output) / 1_000_000


@dataclass(frozen=True)
class TeamPolicy:
    name: str
    key_hash: str
    monthly_budget_usd: float
    allowed_models: tuple[str, ...]  # aliases, or ("*",) for all
    data_policy: str = "mask_pii"
    on_budget_exhausted: str = "degrade"  # or "block"

    def may_use(self, alias: str) -> bool:
        return "*" in self.allowed_models or alias in self.allowed_models


@dataclass(frozen=True)
class GatewayConfig:
    models: dict[str, ModelSpec]
    teams: dict[str, TeamPolicy]
    # Preferred model alias for each tier, and the order in which to fall back.
    tier_models: dict[str, str]
    fallback_order: tuple[str, ...]
    embedding_model: str | None = None  # default model for /v1/embeddings
    db_path: Path = Path("gateway.db")
    cache_ttl_seconds: int = 24 * 3600
    max_output_tokens: int = 4000
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), base=Path(path))

    @classmethod
    def from_dict(cls, data: dict, base: Path | None = None) -> GatewayConfig:
        models = {m["alias"]: ModelSpec(**m) for m in data["models"]}
        teams = {}
        for t in data.get("teams", []):
            key_hash = t.get("key_hash", "")
            if t.get("key_env") and os.environ.get(t["key_env"]):
                key_hash = hash_key(os.environ[t["key_env"]])  # e.g. from a secret store
            teams[t["name"]] = TeamPolicy(
                name=t["name"],
                key_hash=key_hash,
                monthly_budget_usd=float(t["monthly_budget_usd"]),
                allowed_models=tuple(t.get("allowed_models", ["*"])),
                data_policy=t.get("data_policy", "mask_pii"),
                on_budget_exhausted=t.get("on_budget_exhausted", "degrade"),
            )
        # LLMGW_DB_PATH lets a container keep the ledger on a volume, apart from the config.
        db = Path(os.environ.get("LLMGW_DB_PATH") or data.get("db_path", "gateway.db"))
        if base is not None and not db.is_absolute():
            db = base.parent / db
        config = cls(
            models=models,
            teams=teams,
            tier_models=dict(data["tier_models"]),
            fallback_order=tuple(data["fallback_order"]),
            embedding_model=data.get("embedding_model"),
            db_path=db,
            cache_ttl_seconds=int(data.get("cache_ttl_seconds", 24 * 3600)),
            max_output_tokens=int(data.get("max_output_tokens", 4000)),
        )
        config.validate()
        return config

    def chat_model(self, alias: str) -> bool:
        return alias in self.models and self.models[alias].kind == "chat"

    def validate(self) -> None:
        for spec in self.models.values():
            if spec.provider not in PROVIDERS:
                raise ValueError(f"model {spec.alias}: provider must be one of {PROVIDERS}")
            if spec.kind not in KINDS:
                raise ValueError(f"model {spec.alias}: kind must be one of {KINDS}")
            if spec.kind == "chat" and spec.tier not in TIERS:
                raise ValueError(f"model {spec.alias}: tier must be one of {TIERS}")
            if spec.provider == "azure_openai" and not spec.base_url:
                raise ValueError(f"model {spec.alias}: azure_openai needs base_url")
            if spec.auth not in ("api_key", "entra_id"):
                raise ValueError(f"model {spec.alias}: auth must be api_key or entra_id")
        for team in self.teams.values():
            if team.data_policy not in DATA_POLICIES:
                raise ValueError(f"team {team.name}: data_policy must be one of {DATA_POLICIES}")
            if team.on_budget_exhausted not in ("block", "degrade"):
                raise ValueError(f"team {team.name}: on_budget_exhausted must be block or degrade")
        for tier, alias in self.tier_models.items():
            if tier not in TIERS or not self.chat_model(alias):
                raise ValueError(f"tier_models: {tier} -> {alias} is not a known tier and model")
        for alias in self.fallback_order:
            if not self.chat_model(alias):
                raise ValueError(f"fallback_order: unknown chat model {alias}")
        if self.embedding_model and (
            self.embedding_model not in self.models
            or self.models[self.embedding_model].kind != "embedding"
        ):
            raise ValueError(f"embedding_model: {self.embedding_model} is not an embedding model")


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


EXAMPLE_CONFIG = {
    "db_path": "gateway.db",
    "cache_ttl_seconds": 86400,
    "max_output_tokens": 4000,
    "models": [
        {
            "alias": "local",
            "provider": "openai_compatible",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434/v1",
            "tier": "local",
            "price_input": 0.0,
            "price_output": 0.0,
            "external": False,
        },
        {
            "alias": "claude-fast",
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "tier": "fast",
            "price_input": 1.0,
            "price_output": 5.0,
            "external": True,
        },
        {
            "alias": "claude-strong",
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "tier": "strong",
            "price_input": 2.0,
            "price_output": 10.0,
            "external": True,
        },
        {
            "alias": "local-embed",
            "provider": "openai_compatible",
            "model": "nomic-embed-text",
            "base_url": "http://localhost:11434/v1",
            "kind": "embedding",
            "tier": "embedding",
            "price_input": 0.0,
            "price_output": 0.0,
            "external": False,
        },
    ],
    "tier_models": {"local": "local", "fast": "claude-fast", "strong": "claude-strong"},
    "fallback_order": ["claude-strong", "claude-fast", "local"],
    "embedding_model": "local-embed",
    "teams": [
        {
            "name": "research",
            "monthly_budget_usd": 50,
            "allowed_models": ["*"],
            "data_policy": "mask_pii",
            "on_budget_exhausted": "degrade",
        },
        {
            "name": "communications",
            "monthly_budget_usd": 20,
            "allowed_models": ["local", "claude-fast"],
            "data_policy": "local_if_pii",
            "on_budget_exhausted": "block",
        },
        {
            "name": "legal",
            "monthly_budget_usd": 0,
            "allowed_models": ["local"],
            "data_policy": "local_only",
            "on_budget_exhausted": "block",
        },
    ],
}
