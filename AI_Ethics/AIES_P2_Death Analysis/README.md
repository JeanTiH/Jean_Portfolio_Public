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
   - **Age**: Divided into two groups: **Under 25** and **25 & Above**.
   - **Race**: Categories include White, Black, Hispanic, and Other (e.g., Asian, Pacific Islander, Native American).
   - **Gender**: Male and Female.

2. **Manner of Death**:
   - Grouped into six categories: Suicide, Natural, Homicide, Accidental, Cannot be Determined (CD), and Other.
   - Analysis revealed:
     - Younger individuals (Under 25) had higher rates of **Homicide** and **Accidents**.
     - Males accounted for the majority of deaths across all manners.

3. **Custody Status**:
   - Classified into five categories: Sentenced, Awaiting Booking (AB), Booked-Awaiting Trial (B-AT), Booked-No Charges Filed (B-NCF), and Other.
   - Older individuals (25 & Above) were more likely to die while **Sentenced**.
   - Black and Hispanic individuals showed disproportionate representation in **Awaiting Booking** and **B-NCF** categories.

4. **Data Manipulation**:
   - The dataset was manipulated to test fairness and bias hypotheses:
     - Fair Hypothesis: **Sentenced Deaths** do not depend on Age.
     - Bias Hypothesis: A clear dependency of Sentenced Deaths on Age was observed, supporting the bias hypothesis.
   - Random sampling maintained similar distributions between the original and reduced datasets, ensuring no group was favored or harmed.

---

### Summary Statistics Tables
#### Age vs Manner of Death
| **Age Group** | **Homicide** | **Suicide** | **Natural** | **Accident** | **CD** | **Other** |
|---------------|--------------|-------------|-------------|--------------|--------|-----------|
| Under 25      | 552          | 147         | 243         | 426          | 255    | 123       |
| 25 & Above    | 1240         | 675         | 2567        | 849          | 610    | 543       |

#### Race vs Custody Status
| **Race**  | **Sentenced** | **Awaiting Booking (AB)** | **Booked-Awaiting Trial (B-AT)** | **Booked-No Charges Filed (B-NCF)** | **Other** |
|-----------|---------------|---------------------------|-----------------------------------|-------------------------------------|-----------|
| White     | 1255          | 210                      | 774                               | 570                                 | 490       |
| Black     | 824           | 135                      | 440                               | 425                                 | 310       |
| Hispanic  | 987           | 140                      | 506                               | 273                                 | 359       |
| Other     | 243           | 60                       | 163                               | 122                                 | 98        |

---

## Reflection

- **Fairness and Bias**:
  - Younger individuals (Under 25) were more likely to experience **Homicide** or **Accidents**, raising questions about potential biases.
  - Black and Hispanic individuals showed disproportionate representation in custody outcomes, suggesting systemic disparities.

- **Random Sampling**:
  - Random sampling preserved the overall distribution of protected class variables, ensuring no group was favored or harmed.
  - This highlights the importance of representative sampling in fairness assessments.

---

## Files
- **`P2.py`**: Code for the analysis and visualization of deaths in custody data.
- **`deaths-in-custody.csv`**: Dataset of deaths in custody in California.
- For more details, refer to the **[P2_report.pdf](P2_report.pdf)**, 
