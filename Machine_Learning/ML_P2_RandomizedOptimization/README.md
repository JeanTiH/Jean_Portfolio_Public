# Project 2: Randomized Optimization

## Overview

This project explores the performance of four randomized optimization algorithms applied to three optimization problems and a neural network. The goal is to analyze how these algorithms behave under different problem domains, highlighting their strengths, weaknesses, and trade-offs. The project emphasizes problem-specific algorithm selection, performance tuning, and comparative analysis.

### Key Objectives

1. Evaluate the performance of four randomized optimization algorithms:
   - Randomized Hill Climbing (RHC)
   - Simulated Annealing (SA)
   - Genetic Algorithm (GA)
   - Mutual Information Maximizing Input Clustering (MIMIC)
2. Solve three optimization problems:
   - **Travelling Salesman Problem (TSP)**: Highlights GA's capability in navigating complex search spaces.
   - **Knapsack Problem**: Demonstrates MIMIC's efficiency in resource allocation optimization.
   - **Flipflop Problem**: Showcases SA's ability to balance exploration and exploitation.
3. Optimize a neural network for binary classification using RHC, SA, and GA, comparing their performance with backpropagation (BP).

## Highlights

- **Optimization Problems**:
  - TSP, Knapsack, and Flipflop, each demonstrating unique algorithmic strengths.
  - Results evaluated on fitness scores, runtime, and convergence.
- **Neural Network Experiment**:
  - Dataset: Diabetes (8 features, 1,270 samples, reused from Project 1).
  - Compared RHC, SA, GA, and BP for accuracy, runtime, and convergence.
- **Evaluation Metrics**: Fitness scores, function evaluations, runtime, accuracy, and learning curves.
- **Tools**: Python libraries (e.g., `mlrose_hiive`) for implementing optimization algorithms and grid search for hyperparameter tuning.

### Key Insights

- **Travelling Salesman Problem**: GA consistently achieves the best performance but demands more computational resources.
- **Knapsack Problem**: MIMIC converges quickly with consistent results but has the longest runtime.
- **Flipflop Problem**: SA efficiently balances exploration and exploitation with minimal function evaluations.
- **Neural Network Experiment**:
  - RHC achieved the highest accuracy with multiple restarts.
  - BP is prone to local optima and sensitive to initial weights.
  - SA and GA offer robust global optimization capabilities.

## Files

- `NN/`:
  - `NNCompare.py`: Neural network experiment implementation.
  - `NNcompare_run.py`: Main script for executing neural network experiments.
  - `Data_Diabetes.csv`: Dada file.
- `3Problems/`:
  - `ML_Knapsack.py`: Solves the Knapsack problem.
  - `ML_TSP.py`: Solves the TSP.
  - `ML_Flipflop.py`: Solves the Flipflop problem.
  -Problems.py`: Shared utilities and problem definitions.
- `P2_analysis.pdf`: Detailed report analyzing the performance of the algorithms.
- `README.txt`: Instructions for running the code.
- `requirement.txt`: Required Python libraries.

## Project Writeup

- `Project2_Writeup.pdf`
