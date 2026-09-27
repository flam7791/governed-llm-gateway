import pytest

from llmgw.evaluation import check, load_tasks, report, router_confusion, run_strategy

from .conftest import ROOT, FakeProvider, make_config


@pytest.mark.parametrize(
    "spec, answer, ok",
    [
        ({"type": "exact", "value": "Lisbon"}, " lisbon. ", True),
        ({"type": "exact", "value": ["Atlantic", "Atlantic Ocean"]}, "Atlantic Ocean", True),
        ({"type": "number", "value": 1350}, "It is over by 1,350 EUR", True),
        ({"type": "number", "value": 30}, "31", False),
        ({"type": "regex", "pattern": "chloe\\W*$"}, "Reasoning...\n**Chloe**", True),
        (
            {"type": "contains_all", "values": ["a=7 years", "b=permanent"]},
            "a = 7 years, b = permanent",
            True,
        ),
        (
            {"type": "json_fields", "equals": {"n": 9}, "contains": {"m": "board"}},
            '```json\n{"n": 9, "m": "Digital Governance Board"}\n```',
            True,
        ),
        ({"type": "json_fields", "equals": {"n": 9}}, "not json", False),
        ({"type": "lines", "count": 3, "prefix": "- "}, "- a\n- b\n- c", True),
        ({"type": "lines", "count": 3, "prefix": "- "}, "Here:\n- a\n- b\n- c", False),
        (
            {
                "type": "all_of",
                "checks": [
                    {"type": "max_words", "value": 5},
                    {"type": "regex", "pattern": "pilot"},
                ],
            },
            "A short pilot.",
            True,
        ),
    ],
)
def test_checks(spec, answer, ok):
    assert check(spec, answer) is ok


def test_router_agrees_with_most_human_labels():
    tasks = load_tasks(ROOT / "evals" / "tasks.jsonl")
    confusion = router_confusion(tasks)
    agreement = confusion["complex->complex"] + confusion["simple->simple"]
    assert len(tasks) == 24
    assert agreement >= 20  # measured: 22 of 24 (see README)


def test_strategies_are_compared_on_quality_and_cost(tmp_path):
    config = make_config(tmp_path)
    tasks = [
        {
            "id": "s",
            "difficulty": "simple",
            "prompt": "Capital of Portugal?",
            "check": {"type": "exact", "value": "Lisbon"},
        },
        {
            "id": "c",
            "difficulty": "complex",
            "prompt": "Compare A and B and draft a plan.",
            "check": {"type": "exact", "value": "PLAN"},
        },
    ]

    def smart(messages):
        return "PLAN" if "Compare" in messages[0]["content"] else "Lisbon"

    providers = {
        "claude-strong": FakeProvider("claude-strong", answer=smart, tokens=(100, 50)),
        "claude-fast": FakeProvider("claude-fast", answer=lambda m: "Lisbon", tokens=(100, 50)),
        "local": FakeProvider("local", answer=lambda m: "Lisbon", tokens=(100, 50)),
    }
    results = {s: run_strategy(s, tasks, config, providers) for s in ("strong", "fast", "auto")}

    assert results["strong"].passed == 2 and results["fast"].passed == 1
    assert results["auto"].passed == 2  # routed the complex task to the strong model
    assert results["auto"].routing_correct == 2
    assert results["fast"].cost_usd < results["auto"].cost_usd < results["strong"].cost_usd
    table = report(list(results.values()), tasks)
    assert "| auto | 2/2 (100%)" in table
