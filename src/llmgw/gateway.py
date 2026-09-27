"""The gateway pipeline: one governed path for every model call.

    authenticate -> validate -> detect personal data -> route (policy, tier, allowed models)
    -> budget check (block, or degrade to a cheaper model) -> mask personal data for external
    models -> cache lookup -> call the model, falling back on failure -> restore masked values
    -> record usage and cost (never the content) -> answer, with the routing reason attached

Every refusal is recorded too, so the ledger shows what was blocked and why.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from .config import GatewayConfig, TeamPolicy, hash_key
from .ledger import Ledger, UsageEntry, month_of
from .pii import Masker, contains_pii
from .providers import Provider, ProviderError
from .router import RoutingError, decide

log = logging.getLogger(__name__)
ROLES = {"system", "user", "assistant"}


class GatewayError(Exception):
    status = 500
    code = "gateway_error"


class AuthError(GatewayError):
    status, code = 401, "invalid_api_key"


class BadRequest(GatewayError):
    status, code = 400, "invalid_request"


class PolicyRefused(GatewayError):
    status, code = 403, "policy_refused"


class BudgetExceeded(GatewayError):
    status, code = 402, "budget_exceeded"


class UpstreamFailed(GatewayError):
    status, code = 502, "all_models_failed"


@dataclass
class ChatRequest:
    messages: list[dict]
    model: str = "auto"
    max_tokens: int = 1024
    temperature: float = 0.0
    allow_fallback: bool = True  # evaluations turn this off to measure one model at a time


@dataclass
class ChatResult:
    request_id: str
    text: str
    model: str
    route_reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool
    pii_masked: int
    fallbacks_tried: list[str] = field(default_factory=list)


def estimate_input_tokens(messages: list[dict]) -> int:
    """Rough pre-call estimate (about 4 characters per token), used only for budget checks."""
    return sum(len(m["content"]) // 4 + 8 for m in messages)


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        ledger: Ledger,
        clock=time.time,
    ):
        self.config = config
        self.providers = providers  # model alias -> provider
        self.ledger = ledger
        self.clock = clock

    # ------------------------------------------------------------------ access

    def authenticate(self, api_key: str | None) -> TeamPolicy:
        if api_key:
            presented = hash_key(api_key)
            for team in self.config.teams.values():
                if team.key_hash and hmac.compare_digest(team.key_hash, presented):
                    return team
        raise AuthError("Missing or invalid API key.")

    def budget_status(self, team: TeamPolicy) -> dict:
        spent = self.ledger.spent(team.name, month_of(self.clock()))
        return {
            "team": team.name,
            "month": month_of(self.clock()),
            "budget_usd": team.monthly_budget_usd,
            "spent_usd": round(spent, 4),
            "remaining_usd": round(team.monthly_budget_usd - spent, 4),
        }

    # ------------------------------------------------------------------ pipeline

    def _validate(self, req: ChatRequest) -> None:
        if not req.messages:
            raise BadRequest("messages must not be empty.")
        for m in req.messages:
            if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
                raise BadRequest("Each message needs a role (system/user/assistant) and text.")
        if not any(m["role"] == "user" for m in req.messages):
            raise BadRequest("At least one user message is required.")
        if not 1 <= req.max_tokens <= self.config.max_output_tokens:
            raise BadRequest(f"max_tokens must be 1 to {self.config.max_output_tokens}.")
        if not 0.0 <= req.temperature <= 1.0:
            raise BadRequest("temperature must be between 0 and 1.")

    def _record(self, team, request_id, model, reason, status, **values) -> None:
        self.ledger.record(
            UsageEntry(
                ts=self.clock(),
                team=team.name,
                request_id=request_id,
                model=model,
                route_reason=reason,
                input_tokens=values.get("input_tokens", 0),
                output_tokens=values.get("output_tokens", 0),
                cost_usd=values.get("cost_usd", 0.0),
                latency_ms=values.get("latency_ms", 0),
                cache_hit=values.get("cache_hit", False),
                pii_masked=values.get("pii_masked", 0),
                status=status,
            )
        )

    def chat(self, team: TeamPolicy, req: ChatRequest) -> ChatResult:
        request_id = uuid.uuid4().hex[:12]
        self._validate(req)
        has_pii = contains_pii(req.messages)

        try:
            decision = decide(req.model, req.messages, team, self.config, has_pii)
        except RoutingError as exc:
            self._record(team, request_id, "-", str(exc), "blocked_policy")
            raise PolicyRefused(str(exc)) from exc
        reason = decision.reason
        candidates = [decision.alias, *decision.fallbacks]
        if not req.allow_fallback:
            candidates = candidates[:1]

        # Budget: keep only the models this request can afford; degrade or block if needed.
        remaining = team.monthly_budget_usd - self.ledger.spent(team.name, month_of(self.clock()))
        estimate = estimate_input_tokens(req.messages)
        affordable = [
            a
            for a in candidates
            if self.config.models[a].cost(estimate, req.max_tokens) <= remaining
        ]
        if not affordable or (
            affordable[0] != candidates[0] and team.on_budget_exhausted == "block"
        ):
            message = (
                f"Monthly budget reached for team '{team.name}' "
                f"({team.monthly_budget_usd:.2f} USD; {max(remaining, 0):.4f} USD left)."
            )
            self._record(team, request_id, candidates[0], message, "blocked_budget")
            raise BudgetExceeded(message)
        if affordable[0] != candidates[0]:
            reason += f"; budget nearly used, degraded to {affordable[0]}"
        candidates = affordable

        tried: list[str] = []
        for alias in candidates:
            spec = self.config.models[alias]
            masker = Masker()
            outbound = req.messages
            if spec.external and team.data_policy == "mask_pii":
                outbound = masker.mask_messages(req.messages)
            note = reason + (f"; fell back after {', '.join(tried)} failed" if tried else "")

            # Cache (only for temperature 0, i.e. when the caller wants a repeatable answer).
            key = hashlib.sha256(
                json.dumps([team.name, alias, outbound, req.max_tokens]).encode("utf-8")
            ).hexdigest()
            if req.temperature == 0:
                cached = self.ledger.cache_get(key, self.config.cache_ttl_seconds)
                if cached:
                    self._record(
                        team,
                        request_id,
                        alias,
                        note,
                        "ok",
                        cache_hit=True,
                        pii_masked=masker.total,
                    )
                    return ChatResult(
                        request_id,
                        masker.restore(cached["text"]),
                        alias,
                        note,
                        0,
                        0,
                        0.0,
                        0,
                        True,
                        masker.total,
                        tried,
                    )

            try:
                completion = self.providers[alias].complete(
                    spec.model, outbound, req.max_tokens, req.temperature
                )
            except ProviderError as exc:
                log.warning("model %s failed: %s", alias, exc)
                tried.append(alias)
                continue

            cost = spec.cost(completion.input_tokens, completion.output_tokens)
            if req.temperature == 0:
                self.ledger.cache_put(key, {"text": completion.text})  # stored masked
            self._record(
                team,
                request_id,
                alias,
                note,
                "ok",
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost,
                latency_ms=completion.latency_ms,
                pii_masked=masker.total,
            )
            return ChatResult(
                request_id=request_id,
                text=masker.restore(completion.text),
                model=alias,
                route_reason=note,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=round(cost, 6),
                latency_ms=completion.latency_ms,
                cache_hit=False,
                pii_masked=masker.total,
                fallbacks_tried=tried,
            )

        message = f"All allowed models failed: {', '.join(tried)}."
        self._record(team, request_id, candidates[0], message, "provider_error")
        raise UpstreamFailed(message)
