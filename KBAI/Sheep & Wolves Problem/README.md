### State-Space Search Agent: Sheep and Wolves Problem

**Problem Overview**  
The Sheep and Wolves problem is a classic state-space search task in AI. A group of sheep and wolves must be transported across a river using a boat with limited capacity. The challenge is to find a sequence of moves such that sheep are never outnumbered by wolves on either bank (when sheep are present), while safely transporting all animals to the opposite side.

**Approach**  
I implemented a **Breadth-First Search (BFS)** agent that systematically explores the valid state space to guarantee an **optimal (minimum-move) solution** when one exists.

Each state is represented as:
- the number of sheep on the left bank
- the number of wolves on the left bank
- the current boat position

At each step, the agent:
- generates all possible boat moves
- filters invalid states using safety and feasibility constraints
- avoids revisiting previously explored states
- explores states layer by layer using a FIFO queue

**Key Properties**
- Guarantees optimal solutions via BFS
- Uses constraint-based pruning to reduce the search space
- Scales efficiently as the number of animals increases
- Matches human reasoning for small cases, but significantly outperforms humans on larger configurations

**Files**
- `SemanticNetsAgent.py` — BFS-based search agent implementation (**omitted for confidentiality; available upon request**)
- `Mini-Project 1 Journal.pdf` — Detailed explanation, performance analysis, and comparison to human reasoning

---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
