import pytest

from llmgw.router import RoutingError, complexity, decide

from .conftest import make_config, user


def test_simple_request_goes_to_the_fast_tier(tmp_path):
    config = make_config(tmp_path)
    d = decide(
        "auto", user("What is the capital of Portugal?"), config.teams["research"], config, False
    )
    assert d.alias == "claude-fast" and d.complexity == "simple"


def test_complex_request_goes_to_the_strong_tier_with_reasons(tmp_path):
    config = make_config(tmp_path)
    d = decide(
        "auto",
        user("Compare the three options and draft a recommendation."),
        config.teams["research"],
        config,
        False,
    )
    assert d.alias == "claude-strong"
    assert "compare" in d.reason and "draft" in d.reason


def test_local_only_team_never_leaves_the_building(tmp_path):
    config = make_config(tmp_path)
    d = decide("strong", user("Draft a memo."), config.teams["legal"], config, False)
    assert d.alias == "local" and d.fallbacks == []
    assert "keeps all data local" in d.reason


def test_personal_data_keeps_a_local_if_pii_team_local(tmp_path):
    config = make_config(tmp_path)
    team = config.teams["communications"]
    assert decide("fast", user("Hi"), team, config, has_pii=False).alias == "claude-fast"
    d = decide("fast", user("Hi"), team, config, has_pii=True)
    assert d.alias == "local" and "personal data detected" in d.reason


def test_disallowed_model_falls_to_the_next_allowed_one(tmp_path):
    config = make_config(tmp_path)
    d = decide("strong", user("Draft."), config.teams["communications"], config, False)
    assert d.alias == "claude-fast"
    assert "not allowed" in d.reason


def test_unknown_model_is_refused(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(RoutingError):
        decide("gpt-99", user("Hi"), config.teams["research"], config, False)


def test_team_with_no_usable_model_is_refused(tmp_path):
    config = make_config(tmp_path, legal={"allowed_models": ["claude-fast"]})
    with pytest.raises(RoutingError):
        decide("auto", user("Hi"), config.teams["legal"], config, False)


def test_complexity_signals():
    assert complexity(user("Is 17 prime?"))[0] == "simple"
    level, signals = complexity(user("Budget 100, then 200, then 300: total?"))
    assert level == "complex" and "several numbers" in signals
    assert complexity(user("```python\nprint(1)\n```"))[0] == "complex"
    assert complexity(user("x" * 1300))[0] == "complex"
