# SOTA-Movie-Recommendation-

# Fair and Equitable AI in Recommendation Systems

**Author:** Abraham Ho  
**Last Updated:** August 2025  
**Status:** Active Development  


---

## 📌 Overview

This project develops a **fairness-aware recommendation system** that aims to provide relevant and personalized results **without reinforcing existing biases**.  

By combining:
- **Exploratory Data Analysis (EDA)**
- **Bias detection & mitigation techniques**
- **Fairness metrics**
- **MLflow experiment tracking**

…we ensure that both **accuracy** and **equity** are integral to the system.

---

## 🎯 Objectives

1. **Detect and quantify bias** in recommendation outputs.  
2. **Mitigate disparate impacts** while preserving recommendation quality.  
3. Track experiments with **MLflow** for transparency and reproducibility.  
4. Provide a **reproducible pipeline** for fairness-aware recommender systems.

---

## 📊 Key Fairness Concepts

- **Disparate Impact** – Avoid situations where one group systematically receives less favorable recommendations.
- **Exposure Parity** – Ensure balanced visibility for items from diverse providers.
- **Calibration** – Maintain consistency between predicted relevance and actual outcomes across groups.

---

## 🗂 Project Structure

```plaintext
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── src/                 # Source code for recommendation algorithms
├── mlruns/              # MLflow experiment logs
├── README.md            # Project documentation
└── requirements.txt     # Dependencies


```
<img width="1271" height="426" alt="image" src="https://github.com/user-attachments/assets/320600b4-3672-4445-952e-8a8f99e0122e" />
