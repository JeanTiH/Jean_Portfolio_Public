### Case-Based Learning Agent: Monster Identification Problem

**Problem Overview**  
The Monster Identification problem is a supervised classification task in which an agent must determine whether an unseen monster belongs to a target category based on a set of discrete attributes (e.g., size, color, number of limbs, presence of wings or tail).

The challenge is to make accurate decisions from a **very small labeled dataset**, emphasizing similarity-based reasoning rather than statistical learning from large data.

**Approach**  
I implemented a **case-based learning agent** using the **K-Nearest Neighbors (KNN)** algorithm. Each monster is represented by a fixed-length vector of discrete attributes, and classification is performed by retrieving the most similar labeled cases.

Similarity is measured using **Hamming distance** (the number of mismatched attributes). To improve robustness, the agent applies:
- **Exponential distance decay** to weight closer neighbors more strongly
- **Weighted voting** rather than majority voting
- A **margin threshold** to avoid uncertain positive predictions

**Decision Rule**  
The agent sums the weighted support for positive and negative neighbors.  
A monster is classified as *True* only if the difference between positive and negative support exceeds a predefined margin, ensuring confident predictions.

**Key Properties**
- Deterministic and interpretable case-based reasoning
- Linear-time classification in the number of stored cases
- Effective performance with very small datasets
- Careful parameter tuning (k, decay rate, margin) improves robustness on hidden cases

**Human Comparison**
The agent mirrors human reasoning by comparing similarities across attributes, but formalizes the process using distance metrics and weighted voting, making its decisions faster, repeatable, and more consistent than human intuition.

**Files**
- `MonsterClassificationAgent.py` — KNN-based classification agent (**omitted for confidentiality; available upon request**)
- `Mini-Project 4 Journal.pdf` — Full design rationale, parameter tuning, performance analysis, and human comparison
  
---

## Disclaimer
The content (code, report, analyses, writeup, etc.) in this folder are part of the coursework at **Georgia Tech** and are for **demonstration purposes only**. 
Any unauthorized use, reproduction, or distribution may result in a violation of copyright laws and will be subject to appropriate actions.

_**By accessing this folder, you agree to adhere to all copyright policies.**_
