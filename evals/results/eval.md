# Routing evaluation: tasks.jsonl

| Strategy | Pass rate | Cost (USD) | Cost vs strong | Avg latency | Models used |
|---|---|---|---|---|---|
| strong | 19/24 (79%) | 0.0124 | 100% | 1183 ms | claude-strong ×24 |
| fast | 20/24 (83%) | 0.0046 | 37% | 875 ms | claude-fast ×24 |
| local | 19/24 (79%) | 0.0000 | 0% | 18545 ms | local ×24 |
| auto | 24/24 (100%) | 0.0102 | 83% | 1012 ms | claude-fast ×14, claude-strong ×10 |

Router versus human difficulty labels (label -> router estimate): complex->complex: 10, complex->simple: 2, simple->simple: 12, simple->complex: 0

**strong**
- failed s02: '**Lisbon**'
- failed s04: '**Biblioteca**'
- failed s08: '**Yes**, 17 is a prime number. It is only divisible by 1 and itself (its only fa'
- failed s11: '**Analyses**'
- failed s12: '**Atlantic**'

**fast**
- failed c01: '-1450'
- failed c03: '# Token Cost Analysis\n\n**Tokens used:**\n- Research: 4.2 million\n- Legal: 0.9 mil'
- failed c09: 'The pilot tested an AI assistant with 40 staff, saving two hours weekly but requ'
- failed c10: '# R&D Spending Calculation\n\n**Country A:**\n500 billion × 2.1% = 500 × 0.021 = **'

**local**
- failed c03: '## Step 1: Calculate the cost of the Research team\nThe Research team used 4.2 mi'
- failed c04: "Let's break it down step by step:\n\n1. Anna is older than Ben.\n2. Ben is older th"
- failed c07: 'To find the end date, we need to add 18 months to the start date.\n\n15 January 20'
- failed c10: '## Step 1: Calculate the amount spent on R&D by Country A\nTo find out how much C'
- failed c11: "Based on the provided information, here's a Python solution that categorizes the"
