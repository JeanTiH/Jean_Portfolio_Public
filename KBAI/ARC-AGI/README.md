# ARC-AGI Rule-Based Agent

## Overview

This project implements a **rule-based artificial reasoning agent** for the **Abstraction and Reasoning Corpus (ARC)**, a benchmark designed to test generalization, compositional reasoning, and abstract pattern recognition.

The agent infers transformations from a small number of input–output grid examples and applies the learned rules to unseen test cases. Instead of learning from large datasets, the agent relies on **explicit symbolic rules**, prioritizing **interpretability and structured reasoning** over statistical learning.

---

## Core Idea

The agent operates by:
1. **Analyzing training input–output pairs**
2. **Inferring transformation rules**
3. **Verifying rules against all training examples**
4. **Applying the first valid rule to the test input**

If no rule explains all training pairs, the agent returns a safe fallback (an all-zero grid).

---

## Agent Architecture

### Rule-Based Framework

The agent maintains a **catalog of transformation rules**, each consisting of:
- **Detector**: Verifies whether a rule explains all training pairs
- **Solver**: Applies the rule to generate the output (learning parameters if needed)

Rules are evaluated in order of increasing complexity.

### Rule Categories

- **Basic Rules**
  - Direct, parameter-free transformations  
  - Example: *Connect Ends Rule* (fills a row if both endpoints share the same color)

- **Parameterized Rules**
  - Explore variations of a transformation  
  - Example: rotations (90°, 180°, 270°), mirroring directions

- **Combined Rules**
  - Sequential composition of simpler rules  
  - Example: *Tight Crop* → *Color Swap*

### Structural Reasoning

- Shape detection and geometric pattern recognition
- **Breadth-First Search (BFS)** for connected component analysis
- Classification of tasks into:
  - **Shape-preserving** transformations
  - **Shape-altering** transformations

---

## Performance Summary

- **Training Tasks**: Solved all training problems across milestones
- **Test Tasks**: Successfully solved the majority of hidden test cases
- **Runtime**: Extremely efficient (sub-second execution per milestone)

The agent demonstrates strong performance on:
- Linear scan tasks
- Parameter-dependent transformations
- Multi-stage transformations
- Complex structural reasoning problems

Detailed performance breakdowns are provided in the final report :contentReference[oaicite:1]{index=1}.

---

## Successes

The agent generalizes well on tasks involving:
- Rotation and reflection with learned parameters
- Logical color operations (AND, OR, XOR, NOR)
- Sequential transformations (e.g., crop + recolor)
- Structural segmentation using BFS (interior vs. exterior regions)

These successes highlight the effectiveness of **explicit symbolic reasoning** when task structure is discoverable from examples.

---

## Limitations

The agent struggles when:
- Test cases require patterns not covered by existing rules
- Generalization demands flexible pattern synthesis beyond predefined candidates
- Color mappings depend on more complex spatial relationships

These failures illustrate a core limitation of rule-based systems: **generalization is bounded by the rule catalog**.

---

## Design Philosophy

- Emphasizes **interpretability over black-box learning**
- Incremental rule expansion rather than model replacement
- Human-like reasoning for simple tasks, algorithmic precision for complex ones
- Trades adaptability for transparency and speed

A detailed comparison between agent reasoning and human reasoning is discussed in the final report :contentReference[oaicite:2]{index=2}.

---

## Files

- `ArcAgent.py` — Main rule-based agent implementation (**omitted for confidentiality; available upon request**)
- `ARC-AGI Final Journal.pdf` — Full technical report, experiments, and analysis

---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
