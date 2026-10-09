# System card: governed-llm-gateway

| | |
|---|---|
| Component | C1 model gateway ([ai-engineering-framework](https://github.com/flam7791/ai-engineering-framework)) |
| Models | Configured per deployment: Claude, Azure OpenAI (Entra ID), local open-weight models through Ollama or any OpenAI-compatible server |

## Intended use

One governed door to every model for an organisation's applications: route each request to the
cheapest adequate model, enforce team budgets and data policies, mask personal data before it
leaves, and report usage and cost per team without storing content.

## Out of scope

Judging whether an answer is correct (the applications evaluate their own outputs); detecting
every kind of personal data (names and addresses are not detected).

## Data

Requests pass through in memory. Personal data (emails, phone numbers, validated IBANs and card
numbers) is masked for external models under `mask_pii`. `local_only` teams never reach an
external model; `local_if_pii` keeps requests with personal data local. The ledger stores tokens,
cost, latency, route reason and status, never prompt or answer text.

## How it can fail

- The complexity router sends a hard task to the fast tier (2 of 24 in the recorded evaluation;
  both passed anyway). In `judged` mode a local model can misjudge too; a low confidence, an
  invalid answer or a failure falls back to the rules, and the data policy is applied after the
  judgment either way. Measured with Llama 3.1 8B as the judge, it agreed with human labels on
  14 of 24 tasks (rules: 22), confidently calling hard tasks simple; `judged` is off by default.
- Personal data the detectors do not cover (names, addresses) reaches an external model under
  `mask_pii`: use `local_only` or `local_if_pii` for such teams.
- Budget checks use a pre-call estimate; the actual cost is recorded after the call.

## Evaluation

24 tasks with deterministic answer checks, run through four strategies (strong, fast, local,
auto) with pass rate, cost and latency; CI replays the recorded run.

## Human oversight

Team budgets, policies and allowed models are set by people in the configuration; usage and cost
are visible per team; Prometheus alerts on budgets, provider failures and latency.
