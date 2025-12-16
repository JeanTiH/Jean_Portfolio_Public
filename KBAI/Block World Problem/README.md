### Heuristic Search Agent: Block World Problem

**Problem Overview**  
The Block World problem is a classic AI planning task in which blocks stacked on a table must be rearranged from an initial configuration to a target configuration using a minimal number of legal moves. At each step, only the top block of a stack may be moved, either onto the table or onto another stack.

The challenge lies in efficiently searching a large state space while guaranteeing an **optimal (minimum-move) solution**.

**Approach**  
I implemented an **A\* search agent** that finds the minimum-cost solution using an admissible heuristic. Each state represents a configuration of block stacks, and legal moves are generated under strict constraints to reduce unnecessary branching.

The evaluation function is:
f(s) = g(s) + h(s)
where:
- `g(s)` is the number of moves taken so far
- `h(s)` is a heuristic estimating remaining cost

**Heuristic Design**  
The heuristic counts **support mismatches**: for each block, it compares its immediate support (another block or the table) in the current state with its support in the goal state.  
Each mismatch implies at least one required move, making the heuristic **admissible** and guaranteeing optimality.

**Key Optimizations**
- Priority queue–based frontier expansion (A\*)
- Pruning moves that do not advance toward goal-consistent stack prefixes
- Canonical state representation to eliminate duplicates
- Skipping re-expansion of states reached with higher cost

**Key Properties**
- Guarantees optimal solutions with an admissible heuristic
- Efficient pruning keeps search tractable for moderate problem sizes
- Demonstrates informed search and planning beyond uninformed BFS
- Outperforms human intuition as problem size increases

**Files**
- `BlockWorldAgent.py` — A\*-based planning agent (**omitted for confidentiality; available upon request**)
- `Mini-Project 2 Journal.pdf` — Detailed design, heuristic analysis, performance results, and human comparison

---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
