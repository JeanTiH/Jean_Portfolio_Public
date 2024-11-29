# Project 4: Reinforcement Learning

## Overview

This project explores the application of reinforcement learning algorithms to two environments: Blackjack and Frozen Lake. The goal is to implement, evaluate, and compare different approaches, focusing on policy iteration, value iteration, and Q-learning.

### Key Objectives

1. Implement and evaluate reinforcement learning algorithms:
   - Policy Iteration (PI)
   - Value Iteration (VI)
   - Q-Learning (QL)
2. Apply these algorithms to two environments:
   - **Blackjack**: A card-based stochastic environment.
   - **Frozen Lake**: A grid-based environment with slippery surfaces.
3. Analyze algorithm performance in terms of convergence, runtime, and policy quality.

## Highlights

- **Environments**:
  - **Blackjack**: Tests the algorithms in a probabilistic card game scenario.
  - **Frozen Lake**: Tests the algorithms in deterministic and stochastic grid-world settings.
- **Evaluation Metrics**:
  - Convergence speed
  - Policy stability
  - Reward maximization
- **Tools**:
  - Custom Python implementations for the Blackjack environment.
  - OpenAI Gym for the Frozen Lake environment.

### Key Insights

- **Policy Iteration and Value Iteration**:
  - Achieve stable policies in deterministic settings faster than Q-learning.
  - Struggle with stochastic environments without fine-tuning.
- **Q-Learning**:
  - Performs well in stochastic settings like Blackjack.
  - Requires more episodes for convergence compared to iterative methods.
- **Frozen Lake**:
  - Larger grid sizes (16x16) highlight the challenges of scaling reinforcement learning algorithms.

### Files

- `code/`:
  - `blackjack-envP`: Custom implementation of the Blackjack environment.
  - `VIPI-balckjack.py`: Runs Policy Iteration and Value Iteration for Blackjack.
  - `QL-balckjack.py`: Runs Q-Learning for Blackjack.
  - `frozen_lake.py`: Custom implementation of the Frozen Lake environment.
  - `VIPI-frozenlake.py`: Runs Policy Iteration and Value Iteration for Frozen Lake.
  - `QL-frozenlake.py`: Runs Q-Learning for Frozen Lake.
  - `plots.py`: Generates plots for visualization and performance analysis.

- `P4_analysis.pdf`: Detailed report analyzing results of reinforcement learning experiments.
- `README.txt`: Instructions for setting up and running the project.
- `requirement.txt`: Required Python libraries.

## Project Writeup

- `Project4_Writeup.pdf`
