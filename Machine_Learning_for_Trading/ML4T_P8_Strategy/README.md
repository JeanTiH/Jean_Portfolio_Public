# Project 8: Strategy Evaluations

## Overview
This project implements and evaluates two trading strategies:
- **Manual Strategy (MS)**: Combines technical indicators to signal buy/sell actions.
- **Learner Strategy (LS)**: Uses a Random Forest classifier trained on technical indicators to make trading decisions.

Both strategies are tested against a benchmark buy-and-hold strategy. The project explores the performance of these strategies under varying conditions, including different impact values.

### Key Objectives
1. Implement and evaluate **Manual Strategy (MS)** using three technical indicators:
   - **Price/SMA Ratio (P/SMA)**
   - **Price Rate of Change (ROC)**
   - **Bollinger Bands (BB)**
2. Implement and evaluate **Learner Strategy (LS)**, leveraging supervised learning techniques.
3. Compare the performance of MS, LS, and the benchmark across in-sample and out-of-sample periods.
4. Analyze the impact of transaction costs and market impacts on strategy performance.

## Files
- `ManualStrategy.py`: Implements the Manual Strategy using three technical indicators.
- `StrategyLearner.py`: Implements the Learner Strategy with a Random Forest classifier.
- `indicators.py`: Defines and calculates the technical indicators used in both strategies.
- `marketsimcode.py`: Simulates trading actions and computes portfolio value for a given strategy.
- `experiment1.py`: Runs and evaluates strategies for Experiment 1 with fixed transaction costs.
- `experiment2.py`: Runs and evaluates strategies for Experiment 2 with varying impact values.
- `testproject.py`: Validates the functionality of strategies and related modules.
- `strategyEval_report.pdf`: Comprehensive report analyzing the performance of both strategies, including statistical results and discussions.

## Project Writeup
https://lucylabs.gatech.edu/ml4t/fall2022/project-8/
