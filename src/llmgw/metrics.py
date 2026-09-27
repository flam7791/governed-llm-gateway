"""Prometheus metrics: what operations teams watch, without reading any content.

Counters are fed from every ledger entry (answers, refusals and failures alike); the budget
gauge is computed from the ledger when metrics are scraped. Counters restart from zero when the
gateway restarts, which is normal for Prometheus; the ledger remains the record for chargeback.

Useful alerts: a team above 80% of its budget, a rising share of provider errors (a model is
down and requests are falling back), latency above an agreed level, refusals by policy.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .ledger import UsageEntry, month_of

LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160)


class Metrics:
    def __init__(self, gateway):
        self.gateway = gateway
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "llmgw_requests",
            "Requests by team, model and outcome (ok, blocked_policy, blocked_budget, "
            "provider_error)",
            ["team", "model", "status"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "llmgw_tokens",
            "Tokens by team, model and direction",
            ["team", "model", "direction"],
            registry=self.registry,
        )
        self.cost = Counter(
            "llmgw_cost_usd", "Model spend in USD", ["team", "model"], registry=self.registry
        )
        self.cache_hits = Counter(
            "llmgw_cache_hits",
            "Answers served from cache",
            ["team", "model"],
            registry=self.registry,
        )
        self.pii = Counter(
            "llmgw_pii_values_masked",
            "Personal data values masked before leaving",
            ["team"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "llmgw_model_latency_seconds",
            "Model call latency (cache hits excluded)",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.budget_used = Gauge(
            "llmgw_budget_used_ratio",
            "Share of this month's budget already spent (1.0 = exhausted)",
            ["team"],
            registry=self.registry,
        )
        self.budget_remaining = Gauge(
            "llmgw_budget_remaining_usd",
            "Budget left this month in USD",
            ["team"],
            registry=self.registry,
        )
        gateway.observers.append(self.observe)

    def observe(self, entry: UsageEntry) -> None:
        self.requests.labels(entry.team, entry.model, entry.status).inc()
        if entry.status != "ok":
            return
        self.tokens.labels(entry.team, entry.model, "input").inc(entry.input_tokens)
        self.tokens.labels(entry.team, entry.model, "output").inc(entry.output_tokens)
        self.cost.labels(entry.team, entry.model).inc(entry.cost_usd)
        self.pii.labels(entry.team).inc(entry.pii_masked)
        if entry.cache_hit:
            self.cache_hits.labels(entry.team, entry.model).inc()
        else:
            self.latency.labels(entry.model).observe(entry.latency_ms / 1000)

    def render(self) -> tuple[bytes, str]:
        month = month_of(self.gateway.clock())
        for team in self.gateway.config.teams.values():
            spent = self.gateway.ledger.spent(team.name, month)
            budget = team.monthly_budget_usd
            self.budget_remaining.labels(team.name).set(budget - spent)
            self.budget_used.labels(team.name).set(spent / budget if budget > 0 else 0.0)
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
