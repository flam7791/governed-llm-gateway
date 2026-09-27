"""Command line: `llmgw init | keys add | serve | ask | report | eval`. Logs go to stderr."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import EXAMPLE_CONFIG, GatewayConfig, hash_key

DEFAULT_CONFIG = "gateway.json"


def _load(args) -> GatewayConfig:
    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"No config at {path}. Create one with: llmgw init --config {path}")
    return GatewayConfig.load(path)


def _init(args) -> int:
    path = Path(args.config)
    if path.exists():
        print(f"{path} already exists; not overwriting.", file=sys.stderr)
        return 1
    path.write_text(json.dumps(EXAMPLE_CONFIG, indent=2), encoding="utf-8")
    print(f"Wrote {path}. Next: llmgw keys add research --config {path}", file=sys.stderr)
    return 0


def _keys_add(args) -> int:
    path = Path(args.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    team = next((t for t in data["teams"] if t["name"] == args.team), None)
    if team is None:
        print(f"No team '{args.team}' in {path}.", file=sys.stderr)
        return 1
    key = f"gw_{args.team}_{secrets.token_urlsafe(24)}"
    team["key_hash"] = hash_key(key)  # only the hash is stored
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(key)
    print(
        "Store this key now: it is not saved anywhere and cannot be shown again.", file=sys.stderr
    )
    return 0


def _serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .factory import build_gateway

    app = create_app(build_gateway(_load(args)))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _ask(args) -> int:
    from .factory import build_gateway
    from .gateway import ChatRequest, GatewayError

    config = _load(args)
    team = config.teams.get(args.team)
    if team is None:
        print(f"No team '{args.team}'.", file=sys.stderr)
        return 1
    try:
        result = build_gateway(config).chat(
            team, ChatRequest(messages=[{"role": "user", "content": args.prompt}], model=args.model)
        )
    except GatewayError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 1
    print(result.text)
    print(
        f"\n[{result.model} | {result.route_reason} | ${result.cost_usd:.6f} | "
        f"{result.latency_ms} ms | cache {'hit' if result.cache_hit else 'miss'} | "
        f"{result.pii_masked} values masked]",
        file=sys.stderr,
    )
    return 0


def _report(args) -> int:
    from .ledger import Ledger

    config = _load(args)
    month = args.month or datetime.now(timezone.utc).strftime("%Y-%m")
    rows = Ledger(config.db_path).monthly_report(month)
    if not rows:
        print(f"No usage recorded for {month}.")
        return 0
    columns = list(rows[0])
    print(f"Usage and chargeback, {month}\n")
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(row[c]) for c in columns))
    for team in sorted({r["team"] for r in rows}):
        total = sum(r["cost_usd"] or 0 for r in rows if r["team"] == team)
        budget = config.teams[team].monthly_budget_usd if team in config.teams else 0
        print(f"\n{team}: {total:.4f} USD of {budget:.2f} USD budget")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}", file=sys.stderr)
    return 0


def _eval(args) -> int:
    from .evaluation import load_tasks, report, run_strategy
    from .factory import build_providers
    from .providers import ReplayMiss

    config = _load(args)
    tasks = load_tasks(Path(args.tasks))
    recordings = Path(args.recordings) if args.recordings else None
    providers = build_providers(config, recordings=recordings, offline=args.offline)
    strategies = [s.strip() for s in args.strategies.split(",")]
    try:
        results = [run_strategy(s, tasks, config, providers) for s in strategies]
    except ReplayMiss as exc:
        print(f"Replay incomplete: {exc} Record again without --offline.", file=sys.stderr)
        return 3
    text = f"# Routing evaluation: {Path(args.tasks).name}\n\n" + report(results, tasks)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval.md").write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmgw", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", default=DEFAULT_CONFIG)
        return p

    with_config(sub.add_parser("init", help="write an example configuration"))

    keys = sub.add_parser("keys", help="manage team API keys")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    p = with_config(keys_sub.add_parser("add", help="create a new API key for a team"))
    p.add_argument("team")

    p = with_config(sub.add_parser("serve", help="run the HTTP gateway"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)

    p = with_config(sub.add_parser("ask", help="send one prompt through the gateway"))
    p.add_argument("prompt")
    p.add_argument("--team", required=True)
    p.add_argument("--model", default="auto")

    p = with_config(sub.add_parser("report", help="monthly usage and chargeback by team"))
    p.add_argument("--month", help="YYYY-MM (default: this month)")
    p.add_argument("--csv", help="also write the report to this CSV file")

    p = with_config(sub.add_parser("eval", help="compare routing strategies on a task set"))
    p.add_argument("tasks")
    p.add_argument("--strategies", default="strong,fast,local,auto")
    p.add_argument("--recordings", help="record model answers here, for offline replay")
    p.add_argument("--offline", action="store_true", help="replay recordings only")
    p.add_argument("--out")

    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "keys":
        return _keys_add(args)
    handlers = {
        "init": _init,
        "serve": _serve,
        "ask": _ask,
        "report": _report,
        "eval": _eval,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
