"""Routing: which model should answer this request, and why.

The decision is rule-based and explained in one sentence, so every choice can be audited:

1. Data policy first. local_only teams never leave the building; local_if_pii teams stay local
   when the request contains personal data.
2. The requested model: an explicit alias, a tier ("fast", "strong", "local"), or "auto".
3. For "auto", a complexity estimate from visible signals (length, reasoning verbs, several
   numbers, structured-output instructions, code). Complex -> strong tier, simple -> fast tier.
4. If the preferred model is not allowed for the team, the next allowed model in the fallback
   order is used; the remaining ones become fallbacks if the call fails. When the preferred
   model is local, fallbacks are local too: nobody asks for a local model to see their request
   sent outside.

The evaluation (evaluation.py) measures how often "auto" picks the tier a person would have
picked, and what that saves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import TIERS, GatewayConfig, TeamPolicy

REASONING_WORDS = re.compile(
    r"\b(analy[sz]e|compare|evaluate|assess|draft|summari[sz]e|explain why|step by step|"
    r"strategy|plan|reason|justify|which .* (more|most|less|least)|by how (much|many)|"
    r"assume|in the form|json|rules?|extract)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
LONG_PROMPT_CHARS = 1200


class RoutingError(ValueError):
    """No model satisfies the request under the team's policy."""


@dataclass
class RouteDecision:
    alias: str
    reason: str
    fallbacks: list[str] = field(default_factory=list)
    complexity: str | None = None
    signals: list[str] = field(default_factory=list)


def complexity(messages: list[dict]) -> tuple[str, list[str]]:
    """Return ("simple" | "complex", the signals that were found)."""
    text = "\n".join(m["content"] for m in messages if m["role"] in ("user", "system"))
    signals = []
    if len(text) > LONG_PROMPT_CHARS:
        signals.append("long prompt")
    words = sorted({m.group(0).lower() for m in REASONING_WORDS.finditer(text)})
    if words:
        signals.append("reasoning or structure: " + ", ".join(words[:4]))
    if len(NUMBER.findall(text)) >= 3:
        signals.append("several numbers")
    if "```" in text or re.search(r"\b(def |return |select .* from)", text, re.IGNORECASE):
        signals.append("code")
    return ("complex" if signals else "simple"), signals


def _allowed(alias: str, team: TeamPolicy, config: GatewayConfig, local_only: bool) -> bool:
    spec = config.models[alias]
    return team.may_use(alias) and not (local_only and spec.external)


def decide(
    requested: str, messages: list[dict], team: TeamPolicy, config: GatewayConfig, has_pii: bool
) -> RouteDecision:
    local_only = team.data_policy == "local_only" or (
        team.data_policy == "local_if_pii" and has_pii
    )
    policy_note = ""
    if team.data_policy == "local_only":
        policy_note = "team policy keeps all data local; "
    elif local_only:
        policy_note = "personal data detected, so kept on a local model; "

    level, signals = None, []
    if requested in config.models:
        if not config.chat_model(requested):
            raise RoutingError(f"'{requested}' is an embedding model; use /v1/embeddings.")
        preferred, why = requested, f"model {requested} requested"
    elif requested in TIERS:
        preferred, why = config.tier_models[requested], f"tier '{requested}' requested"
    elif requested in ("auto", "", None):
        level, signals = complexity(messages)
        tier = "strong" if level == "complex" else "fast"
        preferred = config.tier_models[tier]
        why = f"auto: {level} request" + (f" ({'; '.join(signals)})" if signals else "")
    else:
        raise RoutingError(f"Unknown model '{requested}'. Use auto, a tier, or a model alias.")

    # A request for a local model never silently leaves the building: its fallbacks stay local.
    keep_local = local_only or not config.models[preferred].external
    order = [preferred] + [a for a in config.fallback_order if a != preferred]
    usable = [a for a in order if _allowed(a, team, config, keep_local)]
    if not usable:
        raise RoutingError(f"No model allowed for team '{team.name}' under its data policy.")
    chosen = usable[0]
    if chosen != preferred:
        why += f"; {preferred} not allowed, using {chosen}"
    return RouteDecision(
        alias=chosen,
        reason=policy_note + why,
        fallbacks=usable[1:],
        complexity=level,
        signals=signals,
    )
