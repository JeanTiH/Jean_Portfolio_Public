# Project 4: Word Embeddings and Facial Recognition Analysis

## Overview

This project applies AI and machine learning algorithms to two domains: **word embeddings** and **facial recognition**. By leveraging **pre-trained Word2Vec models** and the **UTKFace dataset**, the study evaluates semantic relationships, bias in protected classes, and demographic representation in facial recognition data. The project highlights challenges in fairness and underrepresentation within machine learning datasets.

---

## Learning Outcomes
- Analyzed semantic relationships between words using Word2Vec embeddings, focusing on similarity, analogy, and correlation tasks.
- Evaluated bias in facial recognition datasets by examining the distribution of demographic attributes (age, gender, race).
- Explored the implications of underrepresentation in protected classes for algorithmic fairness and decision-making.

---

## Key Findings

### Word Embeddings
1. **Similarity and Analogy Tests**:
   - Similarity tasks with the words **"man"** and **"woman"** highlighted semantic relationships with terms like **"doctor"**, **"nurse"**, **"king"**, and **"queen"**.
   - Analogy tasks (e.g., **"man is to woman as king is to queen"**) showed the model's ability to capture relationships but revealed subtle gender biases.

2. **Correlation Analysis**:
   - Correlation between human-based and model-based analogy scores was **0.028**, indicating **very weak alignment**.  
   - Discrepancies reflect the influence of pre-trained model biases and the limitations of human interpretation in evaluating semantics.

### Facial Recognition
1. **Demographic Representation**:
   - **Race**: **White** subgroups dominated the dataset, while **Black** subgroups had the least representation.  
   - **Gender**: **Female** images were more frequent than **Male** images.  
   - **Age**: Subgroup **"[81-116)"** was the least represented, contributing to potential bias in age-related tasks.

2. **Bias Implications**:
   - Underrepresented subgroups like **Black** and **Age 81–116** could lead to **algorithmic inaccuracies** and biased outcomes.
   - Models trained on imbalanced data may exhibit lower accuracy and fairness when deployed in real-world applications.

---

## Reflection

- **Bias in Pre-trained Models**:
  - Word embeddings demonstrated gender and cultural biases that affect downstream tasks.
  - Facial recognition datasets revealed significant underrepresentation in protected classes, such as **Black** and **older age groups**.

- **Fairness and Ethics**:
  - Addressing dataset imbalance and embedding biases is critical for improving fairness and accuracy in AI systems.
  - Ensuring diverse and representative datasets is a key step toward ethical AI deployment.

---

## Files
- **`P4_task1.py`**: Code for word embedding analysis, including similarity, analogy, and correlation tasks.
- **`P4_task2.py`**: Code for facial recognition analysis, including demographic bias evaluation.
- **`data/`**:
  - Pre-trained word vectors for word embedding analysis.
  - **Note**: The UTKFace dataset can be downloaded from the [official UTKFace repository](https://susanqq.github.io/UTKFace/).
- For more details, refer to the **[P4_report.pdf](P4_report.pdf)**.
