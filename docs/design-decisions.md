# Design decisions

## 1. A gateway, not a library

**Context.** Governance rules (who may use which model, with what data, within what budget) are
useless if each application implements them differently or not at all.
**Decision.** A network service that every application calls, speaking the OpenAI
chat-completions format.
**Consequences.** Rules live in one place, and existing SDKs work by changing a base URL. It adds
one network hop and becomes a critical service: it needs monitoring, and a fallback path.

## 2. Rule-based routing first, measured

**Context.** Learned routers exist, but they need logged traffic to train on and are harder to
explain to auditors and budget holders.
**Decision.** Transparent rules: data policy, then the requested tier, then a complexity
estimate from visible signals. Every decision comes with a one-sentence reason. Agreement with
human labels is measured on the task set (22 of 24 today).
**Consequences.** Easy to audit and to tune. It will misroute some requests; the evaluation
shows whether that costs quality. The ledger collects the data a learned router would need
later.

## 3. Data policy beats cost

**Context.** The cheapest or best model is irrelevant if the data may not leave.
**Decision.** Routing applies the team's data policy before anything else. `local_only` teams
can never reach an external model, even when they request one explicitly.
**Consequences.** A sensitive team may get weaker answers. That trade-off is explicit and
visible in the route reason.

## 4. Mask and restore, rather than refuse

**Context.** Much useful work involves an email address or a phone number. Refusing every such
request pushes people to unmanaged tools.
**Decision.** For `mask_pii` teams, detected values are replaced by stable placeholders before
leaving, and restored in the answer. Only validated patterns are masked (IBAN checksum, Luhn),
to avoid mangling ordinary numbers.
**Consequences.** External providers never see those values. Pattern detection misses names and
addresses, which is stated plainly and handled by the stricter policies.

## 5. Budgets enforced before the call

**Context.** Discovering an overspend at month end is too late.
**Decision.** Each request is priced with a conservative pre-call estimate (input size plus the
maximum output) against the team's remaining budget. Teams choose to block, or to degrade to a
cheaper model.
**Consequences.** Spend cannot run away. Estimates are rough; actual costs are recorded after
the call, and budgets are checked against actual spend.

## 6. Record usage, never content

**Context.** Usage data is needed by finance and management; prompt content is often
confidential.
**Decision.** The ledger stores team, model, tokens, cost, latency, route reason and outcome,
but no prompt or answer text. The cache stores masked answers only, scoped per team.
**Consequences.** Chargeback and oversight without a content archive. Debugging relies on
routing reasons rather than transcripts.

## 7. Deterministic checks, recorded runs

**Context.** Model-graded evaluations drift with the grader, and live runs cost money.
**Decision.** Tasks have deterministic answer checks. Every model answer, and every failure, is
recorded, so evaluations replay offline in CI.
**Consequences.** Reproducible numbers, and the task set is limited to checkable tasks.
Open-ended quality (drafting, tone) needs a separate human or model-graded review.

## 8. Open-weight through a standard interface

**Context.** Organisations increasingly run open-weight models on their own infrastructure for
cost and sovereignty.
**Decision.** Local models are reached through the OpenAI-compatible interface that Ollama,
vLLM and similar servers expose, and are configured like any other model.
**Consequences.** Swapping a laptop model for a private-cloud deployment is a configuration
change. Local "cost" is shown as zero in the ledger, but hardware and energy are real costs;
the price fields can carry an internal rate.

## 9. Embeddings never fall back

**Context.** For chat, falling back to another model on failure keeps the service up. For
embeddings, a vector from another model lives in a different space: mixing them silently breaks
search.
**Decision.** Embeddings go through the same access, data-policy, masking, budget and ledger
steps, but a failure is returned as a failure.
**Consequences.** An index is always built and queried with one model. Changing the embedding
model means re-indexing, which is an explicit, planned step.

## 10. Operable by default

**Context.** A component that cannot be monitored or deployed repeatably stays a prototype.
**Decision.** Metrics for Prometheus, one JSON log line per request (never content), a health
check, a non-root container image, secrets only from the environment, and Entra ID for Azure
OpenAI so no model key needs storing. CI builds and checks the image on every push.
**Consequences.** The gateway drops into a standard platform (containers, a secret store, a
metrics stack) without code changes; the cost is one more dependency (prometheus-client).

## 11. A judged router as an option, with a local judge only

**Context.** The rule-based complexity estimate agrees with human labels on 22 of 24 tasks; its
misses are hard tasks that look simple (short, no reasoning verbs). A model reads the request
rather than its surface, but a model in the routing path adds latency and, if it were external,
a route out of the organisation before the data policy has decided anything.
**Decision.** Router mode `judged` (off by default): a model answers `simple` or `complex` with a
confidence, as a bounded judgment. Anything outside those two answers, a confidence below
`min_confidence`, or a failed call leaves the decision to the rules, and the route reason says
which happened. The judge must be a local model (configuration refuses an external one), local-
only teams are not judged (every route is local anyway), and the data policy is applied after
the judgment, so a prompt that talks the judge into "complex" can raise the cost of its own
answer, never change where it may go. The judge's cost is added to the request's cost.
**Consequences.** Better routing only where the evaluation shows it: compare `--router judged`
with the rules on `evals/tasks.jsonl` before turning it on. Until a judged run is recorded the
README reports no result for it. A trained classifier on logged routes stays on the roadmap; the
judged mode gives the labelled data and the evaluation it would need.
