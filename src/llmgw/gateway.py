"""The gateway pipeline: one governed path for every model call.

    authenticate -> validate -> detect personal data -> route (policy, tier, allowed models)
    -> budget check (block, or degrade to a cheaper model) -> mask personal data for external
    models -> cache lookup -> call the model, falling back on failure -> restore masked values
    -> record usage and cost (never the content) -> answer, with the routing reason attached

Embeddings take the same governed path (access, data policy, masking, budget, ledger), with one
difference: they never fall back to another model, because vectors from different models cannot
be compared with each other.

Every refusal is recorded too, so the ledger shows what was blocked and why. Each ledger entry is
also written as one JSON log line (logger "llmgw.usage") and passed to observers such as the
Prometheus metrics, so the gateway can be monitored without reading the database.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .config import GatewayConfig, TeamPolicy, hash_key
from .ledger import Ledger, UsageEntry, month_of
from .pii import Masker, contains_pii
from .providers import Provider, ProviderError
from .router import Judgment, RoutingError, decide, judge_messages, parse_judgment
from .tracing import tracer

log = logging.getLogger(__name__)
usage_log = logging.getLogger("llmgw.usage")
ROLES = {"system", "user", "assistant"}
MAX_EMBEDDING_INPUTS = 256
MAX_EMBEDDING_CHARS = 32_000


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


@dataclass
class EmbeddingRequest:
    texts: list[str]
    model: str = "auto"  # "auto" means the configured default embedding model


@dataclass
class EmbeddingResult:
    request_id: str
    vectors: list[list[float]]
    model: str
    route_reason: str
    input_tokens: int
    cost_usd: float
    latency_ms: int
    pii_masked: int


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
        self.observers: list[Callable[[UsageEntry], None]] = []  # e.g. Prometheus metrics

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
        entry = UsageEntry(
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
        self.ledger.record(entry)
        usage_log.info(json.dumps(asdict(entry)))
        span = trace.get_current_span()
        span.set_attributes(
            {
                "llmgw.request_id": request_id,
                "llmgw.status": status,
                "llmgw.model_alias": model,
                "llmgw.route_reason": reason,
                "llmgw.cost_usd": entry.cost_usd,
                "llmgw.cache_hit": entry.cache_hit,
                "llmgw.pii_masked": entry.pii_masked,
                "gen_ai.usage.input_tokens": entry.input_tokens,
                "gen_ai.usage.output_tokens": entry.output_tokens,
            }
        )
        if status != "ok":
            span.set_status(Status(StatusCode.ERROR, status))
        for observe in self.observers:
            try:
                observe(entry)
            except Exception:  # monitoring must never break a request
                log.exception("usage observer failed")

    def _judge(self, team: TeamPolicy, req: ChatRequest) -> Judgment | None:
        """Router mode "judged": a local model classifies an "auto" request as simple or complex.

        Bounded: two allowed answers and a confidence. A failed call, an answer outside that set
        or a confidence below the threshold returns a Judgment without a level, and the rules
        decide. The judge is local by configuration, so this call never leaves the
        organisation; and the data policy is applied after it, so a prompt that talks the judge
        into "complex" can raise the cost of its own answer, never change where it may go.
        """
        settings = self.config.router
        if settings.mode != "judged" or req.model not in ("auto", "", None):
            return None
        if team.data_policy == "local_only":
            return None  # every route is local anyway: a judgment would change nothing
        alias = settings.judge_model
        spec = self.config.models[alias]
        with tracer.start_as_current_span(f"judge {spec.model}", kind=SpanKind.CLIENT) as call:
            try:
                completion = self.providers[alias].complete(
                    spec.model, judge_messages(req.messages), 60, 0.0
                )
            except ProviderError as exc:
                log.warning("router judge %s failed: %s", alias, exc)
                return Judgment(None, note="judge unavailable, rules used")
            cost = spec.cost(completion.input_tokens, completion.output_tokens)
            try:
                level, confidence = parse_judgment(completion.text)
            except ValueError:
                return Judgment(None, note="judge answer not valid, rules used", cost_usd=cost)
            call.set_attributes({"llmgw.judge_level": level, "llmgw.judge_confidence": confidence})
        if confidence < settings.min_confidence:
            note = f"judge unsure ({level}, {confidence:.2f}), rules used"
            return Judgment(None, confidence, note, cost)
        return Judgment(level, confidence, cost_usd=cost)

    def chat(self, team: TeamPolicy, req: ChatRequest) -> ChatResult:
        with tracer.start_as_current_span("llmgw.chat", kind=SpanKind.SERVER) as span:
            span.set_attributes(
                {
                    "llmgw.team": team.name,
                    "llmgw.data_policy": team.data_policy,
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": req.model or "auto",
                }
            )
            return self._chat(team, req)

    def _chat(self, team: TeamPolicy, req: ChatRequest) -> ChatResult:
        request_id = uuid.uuid4().hex[:12]
        self._validate(req)
        has_pii = contains_pii(req.messages)

        judgment = self._judge(team, req)
        judge_cost = judgment.cost_usd if judgment else 0.0
        try:
            decision = decide(req.model, req.messages, team, self.config, has_pii, judgment)
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
                        cost_usd=judge_cost,
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
                        round(judge_cost, 6),
                        0,
                        True,
                        masker.total,
                        tried,
                    )

            try:
                with tracer.start_as_current_span(
                    f"chat {spec.model}", kind=SpanKind.CLIENT
                ) as call:
                    call.set_attributes(
                        {
                            "gen_ai.operation.name": "chat",
                            "gen_ai.provider.name": spec.provider,
                            "gen_ai.request.model": spec.model,
                            "gen_ai.request.max_tokens": req.max_tokens,
                            "gen_ai.request.temperature": req.temperature,
                            "llmgw.model_alias": alias,
                            "llmgw.model_external": spec.external,
                        }
                    )
                    completion = self.providers[alias].complete(
                        spec.model, outbound, req.max_tokens, req.temperature
                    )
                    call.set_attributes(
                        {
                            "gen_ai.usage.input_tokens": completion.input_tokens,
                            "gen_ai.usage.output_tokens": completion.output_tokens,
                        }
                    )
            except ProviderError as exc:
                log.warning("model %s failed: %s", alias, exc)
                tried.append(alias)
                continue

            # The judge's cost (zero for an unpriced local model) is part of this request's cost.
            cost = spec.cost(completion.input_tokens, completion.output_tokens) + judge_cost
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

    # ------------------------------------------------------------------ embeddings

    def embed(self, team: TeamPolicy, req: EmbeddingRequest) -> EmbeddingResult:
        with tracer.start_as_current_span("llmgw.embeddings", kind=SpanKind.SERVER) as span:
            span.set_attributes(
                {
                    "llmgw.team": team.name,
                    "gen_ai.operation.name": "embeddings",
                    "gen_ai.request.model": req.model or "auto",
                }
            )
            return self._embed(team, req)

    def _embed(self, team: TeamPolicy, req: EmbeddingRequest) -> EmbeddingResult:
        request_id = uuid.uuid4().hex[:12]
        texts = req.texts
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise BadRequest("input must be a non-empty string or list of non-empty strings.")
        if len(texts) > MAX_EMBEDDING_INPUTS:
            raise BadRequest(f"At most {MAX_EMBEDDING_INPUTS} inputs per request.")
        if any(len(t) > MAX_EMBEDDING_CHARS for t in texts):
            raise BadRequest(f"Each input must be at most {MAX_EMBEDDING_CHARS} characters.")

        alias = req.model if req.model not in ("auto", "", None) else self.config.embedding_model
        spec = self.config.models.get(alias or "")
        if spec is None or spec.kind != "embedding":
            raise BadRequest(f"'{alias}' is not an embedding model.")
        reason = f"embedding model {alias}"

        refusal = None
        if not team.may_use(alias):
            refusal = f"Model {alias} is not allowed for team '{team.name}'."
        elif spec.external and team.data_policy == "local_only":
            refusal = f"Team '{team.name}' keeps all data local; {alias} is external."
        elif (
            spec.external
            and team.data_policy == "local_if_pii"
            and contains_pii([{"role": "user", "content": t} for t in texts])
        ):
            refusal = f"Personal data detected; {alias} is external, so it was not sent."
        if refusal:
            self._record(team, request_id, alias, refusal, "blocked_policy")
            raise PolicyRefused(refusal)

        masker = Masker()
        if spec.external and team.data_policy == "mask_pii":
            texts = [masker.mask(t) for t in texts]
            if masker.total:
                reason += f"; {masker.total} personal data values masked"

        remaining = team.monthly_budget_usd - self.ledger.spent(team.name, month_of(self.clock()))
        estimate = sum(len(t) // 4 + 1 for t in texts)
        if spec.cost(estimate, 0) > remaining:
            message = f"Monthly budget reached for team '{team.name}'."
            self._record(team, request_id, alias, message, "blocked_budget")
            raise BudgetExceeded(message)

        try:
            result = self.providers[alias].embed(spec.model, texts)
        except ProviderError as exc:
            message = f"{alias} failed ({exc}); embeddings do not fall back to another model."
            self._record(team, request_id, alias, message, "provider_error")
            raise UpstreamFailed(message) from exc

        cost = spec.cost(result.input_tokens, 0)
        self._record(
            team,
            request_id,
            alias,
            reason,
            "ok",
            input_tokens=result.input_tokens,
            cost_usd=cost,
            latency_ms=result.latency_ms,
            pii_masked=masker.total,
        )
        return EmbeddingResult(
            request_id=request_id,
            vectors=result.vectors,
            model=alias,
            route_reason=reason,
            input_tokens=result.input_tokens,
            cost_usd=round(cost, 8),
            latency_ms=result.latency_ms,
            pii_masked=masker.total,
        )
