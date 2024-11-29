# Project 3: Unsupervised Learning and Dimensionality Reduction

## Overview

This project explores clustering and dimensionality reduction algorithms to understand their performance and impact on data. The goal is to analyze how these algorithms behave under different conditions, how dimensionality reduction affects clustering, and how cluster-enhanced data performs in supervised learning tasks.

### Key Objectives

1. Evaluate the performance of two clustering algorithms:
   - Expectation Maximization (EM)
   - K-Means
2. Analyze the effects of four dimensionality reduction techniques:
   - Principal Component Analysis (PCA)
   - Independent Component Analysis (ICA)
   - Sparse Random Projection (SRP)
   - Isomap (IMP)
3. Reapply clustering on dimensionality-reduced data and compare results.
4. Use dimensionality-reduced datasets and cluster-enhanced datasets to train and evaluate a neural network learner.

## Highlights

- **Datasets**:
  - Data1 (Diabetes): 8 features, 1,932 samples, weak collinearity, and significant outliers.
  - Data2 (Rice): 7 features, 3,810 samples, strong collinearity, and balanced distribution.
- **Clustering Insights**:
  - K-Means generally outperforms EM, especially on balanced datasets.
  - EM’s soft clustering handles imbalanced data better but introduces more clusters.
- **Dimensionality Reduction Insights**:
  - PCA and ICA excel with strongly collinear data (Data2).
  - IMP performs better on datasets with non-linear structures (Data1).
  - SRP struggles with datasets containing noise or weak collinearity.
- **Neural Network Results**:
  - Cluster-enhanced datasets improve neural network performance with reduced computational cost.
  - Dimensionality-reduced data requires simpler models but may introduce computational overhead.

### Key Insights

- Dimensionality reduction can effectively improve clustering and supervised learning by reducing noise and highlighting important features.
- K-Means is highly effective on balanced datasets with fewer outliers.
- Combining clusters as features enhances data representation, leading to better supervised learning performance.

### Files

- `code/`:
  - `DR.py`: Implements dimensionality reduction techniques (PCA, ICA, SRP, Isomap).
  - `NNDR.py`: Evaluates neural network performance on dimensionality-reduced data.
  - `Cluster.py`: Implements clustering algorithms (EM, K-Means).
  - `NNCluster.py`: Neural network experiments on cluster-enhanced datasets.
  - `ClusterDR.py`: Combines clustering and dimensionality reduction experiments.
  - `A3_run.py`: Main script for running and evaluating experiments.
- `P3_analysis.pdf`: Detailed report analyzing results of clustering and dimensionality reduction.
- `data/`: Contains datasets, instructions for running the code (`README.txt`), and required Python libraries (`requirement.txt`).

## Project Writeup

- `Project3_Writeup.pdf`
