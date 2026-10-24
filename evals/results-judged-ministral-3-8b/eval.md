# Routing evaluation: tasks.jsonl

Router for auto: judged by local, confidence >= 0.7.

| Strategy | Pass rate | Cost (USD) | Cost vs strong | Avg latency | Models used |
|---|---|---|---|---|---|
| auto | 22/24 (92%) | 0.0084 | n/a | 958 ms | claude-fast ×19, claude-strong ×5 |

Rules versus human difficulty labels (label -> rules estimate): complex->complex: 10, complex->simple: 2, simple->simple: 12, simple->complex: 0

Auto routing agreement with human labels: 17/24 (strong tier for complex tasks, fast or local for simple ones)

**auto**
- failed c03: '# Token Cost Analysis\n\n**Tokens used:**\n- Research: 4.2 million\n- Legal: 0.9 mil'
- failed c09: 'The pilot tested an AI assistant with 40 staff, saving two hours weekly but requ'
