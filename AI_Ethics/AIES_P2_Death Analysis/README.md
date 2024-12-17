# Project 2: Deaths in Custody Analysis

## Overview

This project investigates patterns and potential biases in deaths that occur in custody or during the process of arrest in California. By analyzing a dataset of deaths in custody, the study explores relationships between legally protected class variables (Age, Gender, Race) and outcomes like **Manner of Death** and **Custody Status**. It evaluates whether observed patterns suggest fairness or bias and examines the implications of random sampling on dataset distributions and protected class representation.

---

## Learning Outcomes
- Analyzed relationships between protected class variables (Age, Gender, Race) and outcomes (Manner of Death, Custody Status) using statistical measures and visualizations.
- Investigated fairness and bias hypotheses by presenting manipulated data views.
- Assessed the impact of random sampling on dataset distributions and protected class representation.

---

## Key Findings

1. **Protected Class Variables**:
   - **Age**: Divided into two groups: **Under 40** and **40 & Above**.
   - **Race**: Categories include White, Black, Hispanic, and Other (e.g., Asian, Pacific Islander, etc.).
   - **Gender**: Male and Female.

2. **Manner of Death**:
   - Grouped into six categories: Suicide, Natural, Homicide, Accidental, Cannot be Determined (CD), and Other.
   - Analysis revealed:
     - Younger individuals (Under 40) had higher rates of **Homicide**.
     - Males accounted for the majority of deaths across all manners.

3. **Custody Status**:
   - Classified into five categories: Sentenced, Awaiting Booking (AB), Booked-Awaiting Trial (B-AT), Booked-No Charges Filed (B-NCF), and Other.
   - Older individuals (40 & Above) were more likely to die while **Sentenced**.
   - White individuals showed disproportionate representation while **Sentenced**.

4. **Data Manipulation**:
   - The dataset was manipulated to test fairness and bias hypotheses:
     - Fair Hypothesis: **Sentenced Deaths** do not depend on **Age**.
     - Bias Hypothesis: A clear dependency of **Sentenced Deaths** on **Age** was observed.
   - Random sampling maintained similar distributions between the original and reduced datasets, ensuring no group was favored or harmed.

---

### Summary Statistics Tables
#### Age vs Manner of Death (CD: Cannot be Determined)
| **Age Group** | **Homicide** | **Suicide** | **Natural** | **Accident** | **CD** | **Other** |
|---------------|--------------|-------------|-------------|--------------|--------|-----------|
| Under 40      | 994          | 482         |  442        | 347          | 59     | 148       |
| 40 & Above    | 510          | 374         | 4493        | 336          | 56     | 131       |

#### Race vs Custody Status
| **Race**  | **Sentenced** | **Awaiting Booking (AB)** | **Booked-Awaiting Trial (B-AT)** | **Booked-No Charges Filed (B-NCF)** | **Other** |
|-----------|---------------|---------------------------|-----------------------------------|-------------------------------------|-----------|
| White     | 2249          | 45                        | 471                               | 92                                  | 582       |
| Black     | 1316          | 20                        | 211                               | 43                                  | 382       |
| Hispanic  | 1365          | 31                        | 286                               | 54                                  | 757       |
| Other     |  265          |  1                        |  63                               | 14                                  | 124       |

---

## Reflection

- **Fairness and Bias**:
  - Data manipulation can introduce bias, influencing outcomes and leading to varying or potentially misleading conclusions.

- **Random Sampling**:
  - Random sampling preserved the overall distribution of protected class variables, ensuring no group was favored or harmed.
  - This highlights the importance of representative sampling in fairness assessments.

---

## Files
- **`P2.py`**: Code for the analysis and visualization of deaths in custody data.
- For more details, refer to the **[P2_report.pdf](P2_report.pdf)**, 
