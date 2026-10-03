# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org): a new minor
version adds features without breaking existing configurations.

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
