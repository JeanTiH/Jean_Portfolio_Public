# Final Project: Bias Mitigation in Credit Decisions

## Overview

This project explores bias mitigation techniques in credit decision-making using the **Credit Card Approvals dataset**. The analysis evaluates fairness metrics such as **Disparate Impact (DI)** and **Statistical Parity Difference (SPD)** before and after applying bias mitigation techniques like **reweighing**. A **Random Forest classifier** is trained on both original and transformed datasets to assess the effectiveness of bias mitigation and its implications for fairness and financial decision-making.

---

## Learning Outcomes
- Investigated bias in credit decision outcomes across protected classes (e.g., Age, Familial Status).
- Computed fairness metrics (DI, SPD) to evaluate and compare bias in privileged and unprivileged groups.
- Applied bias mitigation techniques and assessed their effectiveness through fairness metrics and classification outcomes.

---

## Key Findings

1. **Protected Classes and Subgroups**:
   - **Age**: Subgroups are **Under 25** (unprivileged) and **25 & Above** (privileged).
   - **Familial Status**: Subgroups are **Married** (privileged) and **Single/Divorced/etc.** (unprivileged).

2. **Fairness Metrics with Dependent Variable 2 Before Bias Mitigation**:
   - **Disparate Impact (DI)**:
     - **Age**: 0.76 (below threshold, favoring the privileged group).
     - **Familial Status**: 0.55 (below threshold, favoring the privileged group).
   - **Statistical Parity Difference (SPD)**:
     - **Age**: -0.10 (within acceptable range).
     - **Familial Status**: -0.17 (below threshold, favoring the privileged group).

3. **Bias Mitigation Using Reweighing**:
   - Improved fairness metrics:
     - DI for Age: Increased to 0.82 (within acceptable range).
     - DI for Familial Status: Increased to 0.61 (still below threshold, favoring the privileged group).
     - SPD for Age: Increased to -0.07 (within acceptable range).
     - SPD for Familial Status: Increased to -0.15 (still below threshold, favoring the privileged group).

4. **Bias Mitigation with Random Forest**:
    - DI for Age: Increased to 0.93 (within acceptable range).
    - DI for Familial Status: Increased to 0.80 (within acceptable range).
    - SPD for Age: Increased to -0.03 (within acceptable range).
    - SPD for Familial Status: Increased to -0.08 (within acceptable range).


---

## Reflection

- **Fairness Metrics**:
  - Both DI and SPD revealed substantial bias against unprivileged groups in the original dataset.
  - Bias mitigation techniques effectively improved fairness metrics, highlighting their importance in ethical AI.

- **Challenges**:
  - Reweighing may overcorrect, introducing reverse bias and affecting fairness for other groups.
  - Random Forest models can amplify existing bias if the training data is not adequately balanced.

---

## Files
- **`Final_Project.ipynb`**: Jupyter notebook containing the code for bias mitigation, fairness metric computation, and Random Forest classification.
- **`clean_dataset.csv`**: Dataset used for bias mitigation and fairness analysis.
- For more details, refer to the **[Final_Project_Report.pdf](Final_Project_Report.pdf)**.
