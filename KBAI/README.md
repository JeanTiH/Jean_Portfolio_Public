# Knowledge Based AI (KBAI)

**Overview:**  
This directory contains **six projects** completed as part of the **Knowledge-Based Artificial Intelligence (KBAI)** coursework. These projects emphasize **symbolic representations, explicit knowledge structures, and interpretable reasoning**, rather than data-hungry statistical learning.

Together, they explore foundational KBAI techniques including **state-space search, heuristic planning, rule-based reasoning, natural language understanding, case-based learning, abductive diagnosis, and abstract reasoning**.

---

## Projects

### 1. State-Space Search Agent: Sheep & Wolves Problem
- **Description**: Implements a **Breadth-First Search (BFS)** agent to solve the classic Sheep and Wolves river-crossing problem. The agent systematically explores valid states while enforcing safety constraints to guarantee an **optimal (minimum-move) solution**.
- **KBAI Concepts**: State-space representation, uninformed search, constraint checking, optimality.

---

### 2. Heuristic Search Agent: Block World Problem
- **Description**: An **A\*** search agent for the Block World planning problem. The agent uses an **admissible heuristic** based on support mismatches to efficiently find the minimum number of legal moves required to reach a goal configuration.
- **KBAI Concepts**: Heuristic search, planning, means–ends analysis, pruning.

---

### 3. Rule-Based NLP Agent: Sentence Reading Problem
- **Description**: A **symbolic natural language understanding agent** that answers questions based on simple English sentences. The agent uses **production rules, frame-based representations**, and linguistic preprocessing to extract answers deterministically.
- **KBAI Concepts**: Production systems, frames, symbolic NLP, explainability.

---

### 4. Case-Based Reasoning Agent: Monster Identification Problem
- **Description**: A **case-based learning agent** using **K-Nearest Neighbors (KNN)** with Hamming distance, exponential distance decay, and weighted voting. Designed to reason from **small knowledge bases** using similarity rather than statistical models.
- **KBAI Concepts**: Case-based reasoning, similarity metrics, knowledge reuse.

---

### 5. Abductive Reasoning Agent: Monster Diagnosis Problem
- **Description**: An **abductive diagnosis agent** that identifies the **smallest set of diseases** explaining a patient’s vitamin profile. Uses **Iterative Deepening Depth-First Search (IDDFS)** with pruning to efficiently search an exponential hypothesis space while guaranteeing optimality.
- **KBAI Concepts**: Abduction, hypothesis-space search, generate-and-test, pruning.

---

### 6. Abstract Reasoning Agent: ARC-AGI
- **Description**: A **rule-based abstract reasoning agent** for the Abstraction and Reasoning Corpus (ARC). The agent infers transformations from limited input–output examples using a catalog of **explicit symbolic rules**, including parameterized and multi-stage transformations.
- **KBAI Concepts**: Symbolic abstraction, rule learning, compositional reasoning, limits of generalization.
- **Highlights**:
  - Shape-based reasoning with BFS for structural analysis
  - Parameter inference (rotations, reflections, color mappings)
  - Explicit comparison between agent reasoning and human reasoning

---

## Core KBAI Themes Demonstrated

- Explicit knowledge representations
- Symbolic and rule-based reasoning
- Classical search: **BFS, A\*, IDDFS**
- Heuristic design and admissibility
- Case-based and abductive reasoning
- Interpretability vs. generalization trade-offs
- Efficient pruning of large state and hypothesis spaces

---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
