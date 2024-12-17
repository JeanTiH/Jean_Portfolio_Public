# Project 3: Toxicity Analysis in Protected Classes

## Overview

This project analyzes relationships between protected class variables and toxicity levels in comments from **Wikipedia Talk Pages**. By examining correlations between attributes like **Gender Identity**, **Race**, and **Religion** with toxicity, the study evaluates potential biases in toxicity labeling and explores methods to measure and mitigate bias in AI systems.

---

## Learning Outcomes
- Identified correlations between protected classes and toxicity levels using statistical measures.
- Investigated bias in natural language data and its implications for AI fairness and ethics.
- Assessed how sampling methods affect representation and measurement in data subsets.

---

## Key Findings

1. **Protected Classes and Subgroups**:
   - Subgroups were categorized into six protected classes: **Gender Identity**, **Sexual Orientation**, **Race**, **Religion**, **Age**, and **Disability**.

2. **Toxicity Correlations**:
   - **Highest Correlations**: **Sexual Orientation** showed weak correlations with toxicity levels.
   - **Lowest Correlations**: **Disability** had very weak correlations with toxicity.

3. **Bias in Subgroup Toxicity**:
   - Subgroup **"Blind"** showed the highest toxicity values in **Disability**, potentially due to:
     - Biased data labeling influenced by human perceptions.
     - Offensive usage of the term "blind" in derogatory contexts.
   - Subgroups **"Deaf"** and **"Paralyzed"** in **Disability** exhibited the lowest toxicity values, attributed to their neutral or descriptive usage.

4. **Sampling Impact**:
   - Random sampling of 10% and 50% datasets preserved toxicity distributions across most protected classes.
   - Smaller sample sizes led to noticeable selection bias and reduced representativeness of subgroups like **Disability**.

---

## Visualizations

### Correlation Coefficients Table
| **Protected Class**     | **Toxicity Correlation** | **Correlation Strength**  |
|-------------------------|--------------------------|---------------------------|
| Gender Identity         | 0.213                    | Weak                      |
| Sexual Orientation      | 0.358                    | Weak                      |
| Race                    | 0.222                    | Weak                      |
| Religion                | 0.160                    | Very Weak                 |
| Age                     | 0.163                    | Very Weak                 |
| Disability              | 0.100                    | Very Weak                 |

### Toxicity by Protected Class Subgroups
**1. Gender Identity**  
![Average Toxicity by Gender Identity](pic/GenderIdentity.png)

**2. Race**  
![Average Toxicity by Race](pic/Race.png)

**3. Sexual Orientation**  
![Average Toxicity by Sexual Orientation](pic/SexualOrientation.png)

---

## Reflection

- **Bias in Data**:
  - Toxicity values reflect human bias in data labeling, particularly against subgroups like **"Blind"** in **Disability**.
  - Offensive contexts amplify toxicity for specific terms, highlighting societal biases embedded in datasets.
- **Ethical Implications**:
  - Bias in toxicity labeling has significant implications for AI fairness, leading to potential discrimination in automated systems.
  - Addressing these biases requires neutral toxicity assessment frameworks, robust data selection processes, and regular evaluations.

---

## Files
- **`P3.py`**: Code for statistical analysis and visualization of toxicity data.
- For more details, refer to the **[P3_report.pdf](P3_report.pdf)**.
