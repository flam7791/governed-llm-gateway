# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org): a new minor
version adds features without breaking existing configurations.

## Unreleased

- First judged-routing run, Llama 3.1 8B as the judge, recorded and replayed in CI: 21/24 tasks
  passed and 14/24 routed as a person would, against 24/24 and 22/24 for the rules. The judge
  called 22 of 24 tasks simple with confidence 0.9 to 1.0. The rules stay the default; results
  in the README.

## 0.4.0

- Router mode `judged` (off by default): a local model classifies `auto` requests as `simple` or
  `complex` with a confidence. An invalid answer, a low confidence or a failure leaves it to the
  rules, and the route reason says so. An external judge is refused by configuration; local-only
  teams are not judged; the data policy applies after the judgment; the judge's cost is part of
  the request's cost.
- `llmgw eval --router judged --judge-model ALIAS [--min-confidence X]` compares it with the
  rules on the same tasks; the report adds the auto strategy's agreement with the human labels.
  No judged run is recorded yet.
- `AGENTS.md` for coding agents; `CLAUDE.md` imports it.

## 0.3.0

- OpenTelemetry tracing (optional `tracing` extra, in the container image): a `llmgw.chat` span
  per request with team, data policy, routing, status, tokens and cost, and a `chat <model>`
  child span per model call with the generative-AI semantic conventions. Incoming W3C trace
  context is continued. No prompt or answer text on spans; off without an OTLP endpoint.

## 0.2.0

- Azure OpenAI provider (v1 API), with an API key or Microsoft Entra ID (managed identity).
- `POST /v1/embeddings`, through the same governed path; embeddings never fall back.
- Prometheus metrics at `/metrics` (optional scrape token), one JSON log line per request.
- Container image: non-root, health check, ledger on a volume, secrets from the environment.
- Team keys can come from environment variables (`key_env`).
- A request for a local model no longer falls back to an external one.
- Malformed provider replies and token failures become clean fallbacks instead of errors.

## 0.1.0

- Routing (explicit, tier, auto), data policies, reversible masking, budgets with block or
  degrade, cache, fallback, ledger and chargeback report, routing evaluation with record/replay.
