# AGENTS.md: governed-llm-gateway

Instructions for coding agents (and people) changing this repository. Read this first.

An OpenAI-compatible gateway that every model call in the platform goes through: per-team data
policy, explainable routing to the cheapest model that is good enough, personal-data masking for
external models, budgets enforced before the call, a usage ledger with no content, metrics and
traces.

## Commands

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                                                    # offline, no key, no local model
cp gateway.example.json gateway.json
llmgw eval evals/tasks.jsonl --config gateway.json --recordings evals/recordings --offline
```

## Layout

- `src/llmgw/gateway.py`: the pipeline (auth, policy, budget, masking, cache, fallback, ledger)
- `src/llmgw/router.py`: routing rules, the complexity estimate, and the optional judged router
- `src/llmgw/config.py`: teams, data policies, models and prices (`gateway.example.json`)
- `src/llmgw/pii.py`: masking and restoring personal data
- `src/llmgw/providers.py`: Anthropic, OpenAI-compatible and local providers
- `src/llmgw/evaluation.py`: router agreement with human labels and cost against single models
- `evals/tasks.jsonl`: 24 labelled tasks; `evals/results/eval.md`: the last recorded run

## Invariants: never weaken these

1. **Data policy beats cost** (design decisions 3). A `local_only` team never reaches an
   external model; `local_if_pii` stays local when personal data is detected. This holds for
   every router mode, every fallback and every budget degradation.
2. **A request for a local model never falls back to an external one.**
3. **Every routing decision carries a one-sentence reason** in the answer and the ledger.
4. **Budgets are checked before the call**, from an estimate; over budget blocks or degrades as
   configured.
5. **The ledger, logs, metrics and traces never hold prompt or answer text** (decision 6).
6. **Embeddings never fall back** to another model (decision 9): vectors from different models
   are not comparable.
7. **Personal data is masked before an external call and restored after**; never send unmasked
   text to an external provider to "improve" a route.

## Working rules

- Routing changes are measured on `evals/tasks.jsonl`; report agreement and cost from the
  evaluation output, never from expectation.
- A change to a model, prompt or route makes recordings stale: re-record, never edit them.
- Record every change in `CHANGELOG.md`; a design change goes in `docs/design-decisions.md`.
- Examples use fictional teams. Commits carry no AI co-author trailers.
