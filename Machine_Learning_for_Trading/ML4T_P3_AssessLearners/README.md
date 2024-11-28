# Project 3: Assess Learners

## Overview
This project evaluates the performance of different machine learning algorithms for regression tasks. Specifically, it compares a **Decision Tree Learner (DTLearner)**, a **Random Tree Learner (RTLearner)**, and a **Bagged Learner**. The project explores key concepts like overfitting, bagging for variance reduction, and trade-offs between training time and accuracy.

### Key Objectives
1. Examine the occurrence of overfitting with respect to leaf size.
2. Investigate how bagging can reduce or eliminate overfitting.
3. Compare the performance of DTLearner and RTLearner in terms of accuracy (using metrics like RMSE and MAE) and training time.
4. Evaluate bagged versions of DTLearner or RTLearner.

## Files
- `DTLearner.py`: Implements a regression tree learner that selects the best feature for splits.
- `RTLearner.py`: Implements a regression tree learner with random feature splits.
- `BagLearner.py`: Implements an ensemble learner using bagging.
- `InsaneLearner.py`: Combines multiple Bagged Learners for experimentation.
- `testlearner.py`: Conducts experiments on training/testing datasets and evaluates performance.
- `assesslearners_report.pdf`: Detailed report analyzing learners' performance, including results, charts, and discussions on overfitting, bagging, and runtime efficiency.

## Project Writeup
https://lucylabs.gatech.edu/ml4t/fall2022/project-3/
