### Abductive Reasoning Agent: Monster Diagnosis Problem

**Problem Overview**  
The Monster Diagnosis problem is an abductive reasoning task in which an agent must identify the **smallest set of diseases** that explains a patient’s observed vitamin profile. Each disease contributes to changes in vitamin levels, and the agent must determine which combination of diseases best accounts for the observed data.

The challenge lies in searching an **exponentially large hypothesis space** while guaranteeing both **completeness** and **optimality**.

**Approach**  
I implemented an **abductive diagnosis agent** using **Iterative Deepening Depth-First Search (IDDFS)**. The agent searches over all possible subsets of disease candidates, incrementally increasing the allowed subset size to guarantee that the first valid solution found is the **minimum explanation**.

Each search state represents a partial selection of diseases. At every step, the agent:
- Generates include/exclude branches for each disease candidate
- Tests whether the current selection can still explain the observed vitamin profile
- Prunes branches that exceed the current subset limit or cannot possibly match the data

**Key Techniques**
- **IDDFS** for memory-efficient, optimal search
- **Generate-and-Test** over the hypothesis space
- **Early pruning** to eliminate infeasible disease combinations
- Guaranteed minimal explanations without storing large frontiers

**Key Properties**
- Finds the smallest valid disease subset
- Balances optimality and memory efficiency
- Scales better in practice than naive exhaustive search
- Strong example of diagnosis as abductive reasoning

**Human Comparison**
The agent formalizes the generate-and-test process that humans intuitively apply, but executes it systematically and efficiently. While humans struggle as the number of disease candidates grows, the agent remains consistent and reliable.

**Files**
- `MonsterDiagnosisAgent.py` — Abductive diagnosis agent using IDDFS (**omitted for confidentiality; available upon request**)
- `Mini-Project 5 Journal.pdf` — Full explanation, pruning strategy, complexity analysis, and human comparison
  
---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
