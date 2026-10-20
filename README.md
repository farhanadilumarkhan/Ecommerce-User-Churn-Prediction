# 🛒 Predicting E-Commerce User Churn Using Machine Learning

An end-to-end Data Science & Machine Learning decision-support system designed to identify e-commerce platform users at risk of churning. By analyzing behavioral, transactional, and demographic indicators, this project trains and evaluates multiple classification algorithms to proactively retain high-risk customers.

---

## 📌 Project Overview
Customer acquisition costs continue to rise, making user retention vital for e-commerce profitability. Traditional retention strategies rely on lagging indicators (e.g., quarterly revenue drops), missing the opportunity for proactive intervention.

This project implements a complete Machine Learning pipeline in Python—from data preprocessing and exploratory data analysis (EDA) to model evaluation and feature importance extraction—focusing heavily on **Recall** and **ROC-AUC** metrics to minimize false negatives (unnoticed churning users).

---

## 📊 Dataset Attributes
The dataset contains **3,941 customer records** with the following key behavioral attributes:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Tenure` | Numerical | Months since customer registration |
| `WarehouseToHome` | Numerical | Distance from warehouse to customer's home |
| `NumberOfDeviceRegistered` | Numerical | Total devices logged into the user account |
| `PreferedOrderCat` | Categorical | Preferred order category (e.g., Mobile Phone, Fashion) |
| `SatisfactionScore` | Discrete | Customer rating score (1 to 5) |
| `MaritalStatus` | Categorical | Customer marital status (Single, Married, Divorced) |
| `NumberOfAddress` | Numerical | Number of saved delivery addresses |
| `Complain` | Binary | Whether customer filed a complaint (1 = Yes, 0 = No) |
| `DaySinceLastOrder` | Numerical | Inactivity period in days since last purchase |
| `CashbackAmount` | Numerical | Average cashback earned |
| **`Churn` (Target)** | **Binary** | **Platform abandonment status (1 = Churned, 0 = Retained)** |

---

## 🛠️ Project Architecture & Pipeline

1. **Data Preprocessing & Cleaning:**
   * Resolved categorical name inconsistencies (e.g., merged `'Mobile'` into `'Mobile Phone'`).
   * Imputed missing numerical values (`Tenure`, `WarehouseToHome`, `DaySinceLastOrder`) using median strategies.
   * Encoded categorical attributes via One-Hot Encoding (`pd.get_dummies`).
   * Normalized numerical features using `StandardScaler`.
   * Split data using stratified 80-20 train-test ratio to handle target class imbalance (~17% churn rate).

2. **Exploratory Data Analysis (EDA):**
   * Visualized target class distribution and feature relationships.
   * Constructed correlation heatmaps and boxplots highlighting key drop-off signals.

3. **Model Development & Comparison:**
   * Evaluated 6 Machine Learning algorithms:
     * **Logistic Regression**
     * **Decision Tree**
     * **Random Forest**
     * **K-Nearest Neighbors (KNN)**
     * **Naive Bayes**
     * **Support Vector Machine (SVM)**

4. **Model Evaluation & Interpretability:**
   * Generated Confusion Matrices, Precision, Recall, F1-Score, and ROC-AUC curves.
   * Extracted Feature Importances from the best-performing model.

---

## 📈 Model Performance & Results

| Model | Accuracy | Precision | **Recall** | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9354** | **0.7917** | **0.8444** | **0.8172** | **0.9666** |
| **SVM** | 0.8542 | 0.5490 | 0.8296 | 0.6608 | 0.9179 |
| **Logistic Regression** | 0.8010 | 0.4542 | 0.8074 | 0.5813 | 0.8756 |
| **Decision Tree** | 0.9303 | 0.8175 | 0.7630 | 0.7893 | 0.8639 |
| **Naive Bayes** | 0.7376 | 0.3647 | 0.7185 | 0.4838 | 0.8118 |
| **KNN** | 0.8910 | 0.7952 | 0.4889 | 0.6055 | 0.9013 |

> **Best Model:** **Random Forest Classifier** achieved the highest Recall (**~84.4%**) and ROC-AUC (**~0.966**), ensuring high sensitivity for detecting churning users.

---

## 💡 Key Business Insights

* **Primary Churn Drivers:**
  1. **Tenure:** Newer users (low tenure) present the highest probability of platform drop-off.
  2. **Cashback Amount:** Lower cashback incentives correlate with higher user churn rates.
  3. **Customer Complaints:** Active complaints are a strong direct predictor of churn.
  4. **Days Since Last Order:** Extended inactivity signals imminent abandonment.

* **Actionable Retention Strategy:**
  * Implement personalized onboarding incentives for new users (< 6 months tenure).
  * Prioritize rapid resolution and follow-up support for users filing complaints.
  * Trigger targeted cashback discount campaigns for users exhibiting high inactivity (`DaySinceLastOrder`).

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Jupyter Notebook / VS Code

### Installation & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/ecommerce-user-churn-prediction.git](https://github.com/farhanadilumarkhan/Ecommerce-User-Churn-Prediction.git)
   cd ecommerce-user-churn-prediction