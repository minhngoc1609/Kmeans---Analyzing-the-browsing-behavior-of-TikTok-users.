# TikTok User Engagement Analysis & Like Prediction

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Model Performance](https://img.shields.io/badge/ROC--AUC-0.8313-green.svg)]()

## 📌 Project Overview
This project analyzes TikTok-style short-video interaction data to understand user engagement behavior and predict whether a user is likely to "like" a video. The dataset contains **over 1.7 million interaction records**, capturing rich user-video dynamics including completion behavior, music IDs, duration, temporal patterns, and geographic/IP signals.

### 🎯 Objectives
*   **Behavioral Insight:** Identify core engagement patterns and inequality in interaction dynamics.
*   **Predictive Modeling:** Build a robust classification model to estimate the probability of a user liking a video.
*   **Imbalance Handling:** Since liked videos account for **less than 1% of all interactions**, the project heavily prioritizes imbalanced-classification metrics over raw accuracy.

---

## 🛠️ Tools & Methodology

*   **Languages & Core Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn
*   **Machine Learning (Scikit-Learn):** K-Means Clustering, Logistic Regression, Naive Bayes, Decision Tree, Random Forest
*   **Optimization:** GridSearchCV, Threshold Tuning, Feature Importance Analysis
*   **Workflow:** End-to-end ML pipeline (Data Cleaning $\rightarrow$ EDA $\rightarrow$ K-Means Segmentation $\rightarrow$ Feature Engineering $\rightarrow$ Model Training $\rightarrow$ Leakage Control $\rightarrow$ Optimization).

---

## 📈 Key Insights from EDA
*   **The 80/20 Rule:** Engagement is heavily concentrated; **80% of total likes originate from just 20% of users**, showing strong inequality in content consumption.
<img width="2236" height="956" alt="image 2" src="https://github.com/user-attachments/assets/f1a9acfc-020b-43f7-88b6-fd91f9508133" />
<img width="2372" height="826" alt="image (1)" src="https://github.com/user-attachments/assets/5e2aa2fd-2b6f-448f-b8d6-0cd22791b987" />

*   **The Golden Hooks:** Most users drop off within the first **1–12 seconds** of watching a video. Capturing attention early is critical.
*   <img width="2396" height="742" alt="image (2)" src="https://github.com/user-attachments/assets/3baabc4b-339b-4d88-822c-6f83cdd6ede1" />

*   **Location Relevance:** Users maintain highly consistent IP addresses. Recommending videos based on **IP/Location signals** yields extremely high engagement potential.
*   <img width="2306" height="858" alt="image (3)" src="https://github.com/user-attachments/assets/e09f3b9c-6111-4005-9a68-f450a3a61084" />


---
## 👥 User & Author Segmentation (K-Means)
To map the platform's ecosystem, K-Means clustering was applied to segment both users and authors into 4 distinct groups based on behavioral metrics:
<img width="1086" height="569" alt="image (4)" src="https://github.com/user-attachments/assets/eee7e31d-99a9-4511-af00-9696a0986e9a" />
<img width="1602" height="370" alt="image (5)" src="https://github.com/user-attachments/assets/30324633-95d5-4c0d-990c-1a96005992a5" />

### User Clusters
| Cluster | Persona | Behavioral Profile |
| :--- | :--- | :--- |
| **Group 1** | *The Ghosts* | Low views, 0 likes $\rightarrow$ New or inactive accounts. |
| **Group 2** | *The Lurkers* | High views but very low likes $\rightarrow$ Passive consumers. |
| **Group 3** | *The Fans* | Average views, high likes $\rightarrow$ Highly active engagers. |
| **Group 4** | *The Superstars*| Extremely high metrics across all aspects $\rightarrow$ Core power users. |

### Author Clusters
Authors were similarly mapped into 4 clusters evaluating metrics such as *total views, total likes, post days, unique music count, and finish rate*, allowing the platform to separate casual creators from high-volume, highly active influencers.
<img width="1095" height="565" alt="image (6)" src="https://github.com/user-attachments/assets/cd2ac585-e63a-4909-85cb-d40b6cf3018d" />
<img width="1770" height="306" alt="image (7)" src="https://github.com/user-attachments/assets/62101cfa-10f3-4e58-9a9b-2f134acef04f" />

---

## 🎛️ Feature Engineering & Leakage Control
*   **Behavioral Features:** Created historical interaction features at the *user*, *author*, and *item* levels (e.g., historical view count, completion rates, and historical popularity).
*   **ID Strategy:** Raw identifier variables (`uid`, `item_id`, `author_id`, `music_id`) were treated cautiously to emphasize interpretable behavioral indicators over sparse raw IDs.
*   **Data Leakage Prevention:** Strictly audited target-related statistics (like user like rates) to ensure metrics were not calculated using future target data before the train-test split.

---

## 🤖 Model Evaluation & Optimization

Given the severe class imbalance ($\approx 0.97\%$ positive rate), **PR-AUC** and **ROC-AUC** served as primary evaluation metrics rather than basic accuracy.

### 📊 Final Model Performance
After feature selection (top 10 non-leakage behavioral features) and `GridSearchCV` hyperparameter tuning, the **Optimized Random Forest** was selected as the final production model:

| Metric | Value | Comparison to Baseline |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8313** | Strong discriminative ability |
| **PR-AUC** | **0.1767** | Substantially higher than the $0.97\%$ random baseline |

> 💡 **Threshold Tuning:** The classification threshold was dynamically tuned and selected based on the optimal **F1-score** to find a practical business balance between capturing true engagement (Recall) and minimizing false alarms (Precision).

---

## 🧠 Reflection & Lessons Learned
1.  **Accuracy is a Trap:** In highly imbalanced contexts (likes < 1%), a naive model predicting all zeros achieves >99% accuracy but provides zero business value. PR-AUC and F1-score provide the true reality.
2.  **Leakage is Dangerous:** Target-related features (e.g., item like counts) can artificially bloat evaluation metrics during training if not carefully isolated, leading to model failure in production.

---

## 🚀 Next Steps
*   [ ] **Advanced Architectures:** Evaluate gradient boosting frameworks like **XGBoost** and **LightGBM**.
*   [ ] **Validation Strategy:** Implement a strict time-based train-test split to simulate real production rollouts.
*   [ ] **Categorical Encoding:** Implement target encoding or embedding techniques for high-cardinality IDs (`author_id`, `music_id`).
*   [ ] **Deployment:** Wrap the optimized Random Forest into a lightweight FastAPI or Streamlit dashboard.
