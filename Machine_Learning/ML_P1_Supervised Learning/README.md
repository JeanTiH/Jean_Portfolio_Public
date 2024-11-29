# Project 1: Supervised Learning

## Overview

This project explores the performance of five supervised learning algorithms on binary classification problems. The goal is to analyze how these algorithms behave under different data characteristics, including balanced datasets, noise, and varying levels of complexity. The project emphasizes algorithm tuning, dataset exploration, and comparative analysis.

### Key Objectives

1. Evaluate the performance of five supervised learning algorithms:
   - Decision Trees (with pruning)
   - Neural Networks
   - Gradient Boosting
   - Support Vector Machines (SVM) with multiple kernels
   - k-Nearest Neighbors (kNN)
2. Compare algorithm performance on two datasets:
   - **Data1 (Diabetes)**: Balanced dataset with moderate size and complexity.
   - **Data2 (Sleepiness)**: Larger dataset with higher complexity and noise.
3. Analyze training and testing error rates, training time, and sensitivity to hyperparameters for each algorithm.
4. Assess how dataset characteristics affect algorithm performance and generalization.

### Highlights

- **Datasets**: Two balanced binary classification datasets with varying complexity.
- **Pre-processing**: Included normalization and under-sampling to address imbalance and improve learning.
- **Evaluation Metrics**: Accuracy, precision, recall, and training/testing time.
- **Tools**: Python (scikit-learn), grid search for hyperparameter tuning, and visualization libraries for plotting learning and validation curves.
- **Key Insights**:
  - SVM achieved the highest accuracy on Data1 but required the longest training time.
  - Gradient Boosting performed robustly on noisy data (Data2), showcasing resilience to noise.
  - kNN was computationally efficient but sensitive to noise.
  - Neural Networks required substantial hyperparameter tuning and large datasets to perform effectively.

### Files

- `ML_Algorithms.py`: Implements the supervised learning algorithms and supports hyperparameter tuning.
- `ML.py`: Main script to load data, preprocess it, and execute the experiments.
- `jhan446-analysis.pdf`: Detailed report analyzing the performance of the five supervised learning algorithms on the datasets.
- `data/`: Contains datasets, instructions for running the code (`README.txt`), and required Python libraries (`requirement.txt`).

## Project Writeup
`Project1_Writeup.pdf`
