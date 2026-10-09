# Routing evaluation: tasks.jsonl

Router for auto: judged by local, confidence >= 0.7.

| Strategy | Pass rate | Cost (USD) | Cost vs strong | Avg latency | Models used |
|---|---|---|---|---|---|
| auto | 21/24 (88%) | 0.0049 | n/a | 867 ms | claude-fast ×22, claude-strong ×2 |

Rules versus human difficulty labels (label -> rules estimate): complex->complex: 10, complex->simple: 2, simple->simple: 12, simple->complex: 0

Auto routing agreement with human labels: 14/24 (strong tier for complex tasks, fast or local for simple ones)

**auto**
- failed c01: '-1450'
- failed c09: 'The pilot tested an AI assistant with 40 staff, saving two hours weekly but requ'
- failed c10: '# R&D Spending Calculation\n\n**Country A:**\n500 billion × 2.1% = 500 × 0.021 = **'
