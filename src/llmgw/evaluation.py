"""Routing evaluation: does "auto" buy most of the strong model's quality at a fraction of its cost?

Each task has a prompt, a difficulty label a person assigned ("simple" or "complex"), and a
deterministic check of the answer (no model grades another model here, so the scores are
reproducible). Every strategy answers every task:

    strong   always the strong commercial model
    fast     always the fast commercial model
    local    always the local open-weight model
    auto     the gateway's router decides

For each strategy the report gives the pass rate, total and per-task cost, and average latency;
for "auto" it also gives routing accuracy against the human labels.
"""

from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import GatewayConfig, TeamPolicy
from .gateway import ChatRequest, Gateway, GatewayError
from .ledger import Ledger
from .providers import Provider
from .router import complexity

STRATEGIES = ("strong", "fast", "local", "auto")


# --------------------------------------------------------------------------- answer checks


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().strip("`'\"").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.rstrip(".").split())


def _json_payload(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def check(spec: dict, answer: str) -> bool:
    kind = spec["type"]
    if kind == "exact":
        options = spec["value"] if isinstance(spec["value"], list) else [spec["value"]]
        return _norm(answer) in {_norm(o) for o in options}
    if kind == "number":
        found = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        return len(found) >= 1 and abs(float(found[-1]) - spec["value"]) <= spec.get("tol", 1e-6)
    if kind == "regex":
        return re.search(spec["pattern"], answer.strip(), re.IGNORECASE) is not None
    if kind == "contains_all":
        compact = _norm(answer).replace(" =", "=").replace("= ", "=")
        return all(_norm(v) in compact for v in spec["values"])
    if kind == "json_fields":
        try:
            data = _json_payload(answer)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        for key, expected in spec.get("equals", {}).items():
            if isinstance(expected, str):
                if _norm(str(data.get(key, ""))) != _norm(expected):
                    return False
            elif data.get(key) != expected:
                return False
        return all(
            _norm(needle) in _norm(str(data.get(key, "")))
            for key, needle in spec.get("contains", {}).items()
        )
    if kind == "lines":
        lines = [line for line in answer.strip().splitlines() if line.strip()]
        prefix = spec.get("prefix", "")
        return len(lines) == spec["count"] and all(line.startswith(prefix) for line in lines)
    if kind == "max_words":
        return len(answer.split()) <= spec["value"]
    if kind == "all_of":
        return all(check(part, answer) for part in spec["checks"])
    raise ValueError(f"Unknown check type {kind}")


# --------------------------------------------------------------------------- running


@dataclass
class StrategyResult:
    strategy: str
    passed: int = 0
    total: int = 0
    cost_usd: float = 0.0
    latency_ms: list[int] = field(default_factory=list)
    routed: dict[str, int] = field(default_factory=dict)  # model alias -> count
    routing_correct: int = 0
    errors: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> int:
        return int(sum(self.latency_ms) / len(self.latency_ms)) if self.latency_ms else 0


def load_tasks(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("//")]


EVAL_TEAM = TeamPolicy(
    name="evaluation",
    key_hash="",
    monthly_budget_usd=1_000_000.0,
    allowed_models=("*",),
    data_policy="mask_pii",
    on_budget_exhausted="block",
)


def run_strategy(
    strategy: str, tasks: list[dict], config: GatewayConfig, providers: dict[str, Provider]
) -> StrategyResult:
    # A fresh ledger (and so an empty cache) per strategy, so no strategy is billed zero
    # because another one already answered the same prompt.
    with tempfile.TemporaryDirectory() as tmp:
        gateway = Gateway(
            replace(config, db_path=Path(tmp) / "eval.db"),
            providers,
            Ledger(Path(tmp) / "eval.db"),
        )
        result = StrategyResult(strategy)
        for task in tasks:
            result.total += 1
            request = ChatRequest(
                messages=[{"role": "user", "content": task["prompt"]}],
                model=strategy,
                max_tokens=task.get("max_tokens", 300),
                temperature=0.0,
                allow_fallback=False,  # measure the chosen model, never a substitute
            )
            try:
                answer = gateway.chat(EVAL_TEAM, request)
            except GatewayError as exc:
                result.errors.append(f"{task['id']}: {exc}")
                continue
            result.cost_usd += answer.cost_usd
            result.latency_ms.append(answer.latency_ms)
            result.routed[answer.model] = result.routed.get(answer.model, 0) + 1
            tier = config.models[answer.model].tier
            expected_strong = task["difficulty"] == "complex"
            result.routing_correct += (tier == "strong") == expected_strong
            if check(task["check"], answer.text):
                result.passed += 1
            else:
                result.failures.append(f"{task['id']}: {answer.text[:80]!r}")
    return result


def router_confusion(tasks: list[dict]) -> dict[str, int]:
    """How the router's complexity estimate compares with the human labels (no model calls)."""
    counts = {
        "complex->complex": 0,
        "complex->simple": 0,
        "simple->simple": 0,
        "simple->complex": 0,
    }
    for task in tasks:
        level, _ = complexity([{"role": "user", "content": task["prompt"]}])
        counts[f"{task['difficulty']}->{level}"] += 1
    return counts


def report(results: list[StrategyResult], tasks: list[dict]) -> str:
    strong_cost = next((r.cost_usd for r in results if r.strategy == "strong"), None)
    lines = [
        "| Strategy | Pass rate | Cost (USD) | Cost vs strong | Avg latency | Models used |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        relative = f"{r.cost_usd / strong_cost:.0%}" if strong_cost else "n/a"
        used = ", ".join(f"{m} ×{n}" for m, n in sorted(r.routed.items()))
        lines.append(
            f"| {r.strategy} | {r.passed}/{r.total} ({r.pass_rate:.0%}) | {r.cost_usd:.4f} | "
            f"{relative} | {r.avg_latency_ms} ms | {used} |"
        )
    confusion = router_confusion(tasks)
    lines += [
        "",
        "Router versus human difficulty labels (label -> router estimate): "
        + ", ".join(f"{k}: {v}" for k, v in confusion.items()),
    ]
    for r in results:
        if r.errors or r.failures:
            lines += ["", f"**{r.strategy}**"]
            lines += [f"- error {e}" for e in r.errors]
            lines += [f"- failed {f}" for f in r.failures]
    return "\n".join(lines) + "\n"
