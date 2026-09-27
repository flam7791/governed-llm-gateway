import sqlite3

import pytest

from llmgw.gateway import (
    AuthError,
    BadRequest,
    BudgetExceeded,
    ChatRequest,
    PolicyRefused,
    UpstreamFailed,
)
from llmgw.ledger import UsageEntry, month_of

from .conftest import FakeProvider, make_gateway, user


def providers_for(gateway_config_models, **overrides):
    providers = {alias: FakeProvider(alias) for alias in gateway_config_models}
    providers.update(overrides)
    return providers


def test_authentication(gateway):
    assert gateway.authenticate("key-research").name == "research"
    with pytest.raises(AuthError):
        gateway.authenticate("wrong")
    with pytest.raises(AuthError):
        gateway.authenticate(None)


def test_personal_data_is_masked_for_external_models_and_restored_for_the_user(tmp_path):
    echo = FakeProvider("claude-fast", answer=lambda m: "I will write to [EMAIL_1].")
    gw = make_gateway(
        tmp_path,
        providers=providers_for(["local", "claude-fast", "claude-strong"], **{"claude-fast": echo}),
    )
    result = gw.chat(
        gw.config.teams["research"],
        ChatRequest(user("Reply to jane@example.org please"), model="fast"),
    )
    sent = echo.calls[0]["messages"][0]["content"]
    assert "jane@example.org" not in sent and "[EMAIL_1]" in sent  # never left the gateway
    assert result.text == "I will write to jane@example.org."  # restored for the user
    assert result.pii_masked == 1


def test_local_only_team_is_routed_locally_even_when_asking_for_strong(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.chat(gw.config.teams["legal"], ChatRequest(user("Draft a memo"), model="strong"))
    assert result.model == "local"
    assert gw.providers["claude-strong"].calls == []


def test_budget_blocks_when_policy_is_block(tmp_path):
    gw = make_gateway(tmp_path, communications={"monthly_budget_usd": 0.0001})
    team = gw.config.teams["communications"]
    with pytest.raises(BudgetExceeded):
        gw.chat(team, ChatRequest(user("Hello there"), model="fast", max_tokens=1000))
    rows = gw.ledger.monthly_report(month_of(gw.clock()))
    assert rows[0]["refused_or_failed"] == 1


def test_budget_degrades_to_a_cheaper_model_when_policy_is_degrade(tmp_path):
    gw = make_gateway(tmp_path, research={"monthly_budget_usd": 0.001})
    result = gw.chat(
        gw.config.teams["research"], ChatRequest(user("Draft a strategy"), model="strong")
    )
    assert result.model == "local"
    assert "degraded to local" in result.route_reason


def test_spend_accumulates_until_the_budget_is_used(tmp_path):
    gw = make_gateway(tmp_path, communications={"monthly_budget_usd": 1.0})
    team = gw.config.teams["communications"]
    gw.ledger.record(
        UsageEntry(
            gw.clock(),
            "communications",
            "x",
            "claude-fast",
            "earlier",
            0,
            0,
            0.999,
            0,
            False,
            0,
            "ok",
        )
    )
    with pytest.raises(BudgetExceeded):
        gw.chat(team, ChatRequest(user("Hello"), model="fast", max_tokens=500))


def test_failure_falls_back_to_the_next_model(tmp_path):
    gw = make_gateway(
        tmp_path,
        providers=providers_for(
            ["local", "claude-fast", "claude-strong"],
            **{"claude-strong": FakeProvider("claude-strong", fail=True)},
        ),
    )
    result = gw.chat(gw.config.teams["research"], ChatRequest(user("Draft a plan"), model="strong"))
    assert result.model == "claude-fast"
    assert result.fallbacks_tried == ["claude-strong"]
    assert "fell back" in result.route_reason


def test_all_models_failing_is_reported(tmp_path):
    down = {a: FakeProvider(a, fail=True) for a in ["local", "claude-fast", "claude-strong"]}
    gw = make_gateway(tmp_path, providers=down)
    with pytest.raises(UpstreamFailed):
        gw.chat(gw.config.teams["research"], ChatRequest(user("Hi")))


def test_no_fallback_when_disabled(tmp_path):
    gw = make_gateway(
        tmp_path,
        providers=providers_for(
            ["local", "claude-fast", "claude-strong"],
            **{"claude-fast": FakeProvider("claude-fast", fail=True)},
        ),
    )
    with pytest.raises(UpstreamFailed):
        gw.chat(
            gw.config.teams["research"], ChatRequest(user("Hi"), model="fast", allow_fallback=False)
        )


def test_repeat_requests_are_cached_only_at_temperature_zero(tmp_path):
    gw = make_gateway(tmp_path)
    team = gw.config.teams["research"]
    first = gw.chat(team, ChatRequest(user("Capital of Portugal?"), model="fast"))
    second = gw.chat(team, ChatRequest(user("Capital of Portugal?"), model="fast"))
    assert not first.cache_hit and second.cache_hit and second.cost_usd == 0
    assert len(gw.providers["claude-fast"].calls) == 1

    gw.chat(team, ChatRequest(user("A poem"), model="fast", temperature=0.7))
    gw.chat(team, ChatRequest(user("A poem"), model="fast", temperature=0.7))
    assert len(gw.providers["claude-fast"].calls) == 3


def test_cache_is_not_shared_between_teams(tmp_path):
    gw = make_gateway(tmp_path)
    gw.chat(gw.config.teams["research"], ChatRequest(user("Capital of Portugal?"), model="fast"))
    other = gw.chat(
        gw.config.teams["communications"], ChatRequest(user("Capital of Portugal?"), model="fast")
    )
    assert not other.cache_hit


def test_ledger_records_cost_but_never_the_prompt(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.chat(
        gw.config.teams["research"],
        ChatRequest(user("Confidential merger discussion"), model="strong", temperature=0.5),
    )
    assert result.cost_usd == pytest.approx((100 * 2.0 + 20 * 10.0) / 1_000_000)
    with sqlite3.connect(gw.config.db_path) as db:
        dump = "\n".join(db.iterdump())
    assert "merger" not in dump
    rows = gw.ledger.monthly_report(month_of(gw.clock()))
    assert rows[0]["team"] == "research" and rows[0]["requests"] == 1


def test_policy_refusal_is_recorded(tmp_path):
    gw = make_gateway(tmp_path, legal={"allowed_models": ["claude-fast"]})
    with pytest.raises(PolicyRefused):
        gw.chat(gw.config.teams["legal"], ChatRequest(user("Hi")))


@pytest.mark.parametrize(
    "request_",
    [
        ChatRequest([]),
        ChatRequest([{"role": "tool", "content": "x"}]),
        ChatRequest([{"role": "system", "content": "only system"}]),
        ChatRequest(user("Hi"), max_tokens=0),
        ChatRequest(user("Hi"), temperature=1.5),
    ],
)
def test_invalid_requests_are_rejected(gateway, request_):
    with pytest.raises(BadRequest):
        gateway.chat(gateway.config.teams["research"], request_)
