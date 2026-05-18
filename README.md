# Bank Customer Churn Prediction

An end-to-end Machine Learning project designed to predict customer attrition (churn) for a banking institution. This model enables proactive customer retention strategies by identifying high-risk profiles based on demographic factors, credit relationships, and transaction behavioral patterns.

## 📌 Executive Summary
Customer churn is a critical problem for financial institutions. Acquiring a new customer costs significantly more than retaining an existing one. This project utilizes an automated machine learning pipeline built on a dataset of credit card customers to accurately classify potential churners. 

To overcome the challenges of severe **Class Imbalance** (~16% churned vs ~84% existing customers) and prevent **Data Leakage**, the pipeline strictly implements:
* **Stratified Splitting** to protect test set integrity.
* **Synthetic Minority Over-sampling Technique (SMOTE)** exclusively applied to the training partition.
* **Standardization Scaling** fitted solely on training distributions.

---

## 📊 Dataset Overview
The dataset contains information on 10,127 credit card holders. The target variable is `Attrition_Flag` which contains two classes:
* **Existing Customer** (Majority Class)
* **Attrited Customer** (Minority Class — Churned)

### Key Features Analyzed:
* **Demographics:** `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category`
* **Account Relationship:** `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`
* **Credit & Transaction Behavior:** `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio`

---

## 🛠️ Project Architecture & Pipeline

### 1. Data Cleaning & Feature Engineering
* Removed uninformative tracking columns (`CLIENTNUM`) and algorithmic artifacts (`Naive_Bayes_Classifier_*`).
* Handled structural outliers and encoded categorical text into numeric identifiers.

### 2. Categorical Encoding Strategies
* **Nominal Features (`Gender`, `Marital_Status`):** Formatted using One-Hot Encoding (`pd.get_dummies`) to eliminate arbitrary numeric rankings.
* **Ordinal Features (`Education_Level`, `Income_Category`, `Card_Category`):** Explicitly mapped using continuous integer rank-scaling to capture the intrinsic hierarchy of levels.

### 3. Production-Ready Machine Learning Pipeline
To maintain professional standard validation and guard against overfitting, the data transformation follows this exact progression: [ Raw Data DataFrame ]
│
▼
[ Stratified Train/Test Split (80/20) ] ────► [ Pure X_test, y_test ] (Held Back)
│
▼
[ SMOTE Over-sampling (X_train, y_train Only) ]
│
▼
[ StandardScaler fit_transform() ] ──────────► [ StandardScaler transform() ]
│                                                      │
▼                                                      ▼
[ Balanced & Scaled X_train_res ]                      [ Pure & Scaled X_test ]
│                                                      │
▼                                                      ▼
[ Model Training (RF / SVM) ] ──────────────────────► [ Production Inference Evaluator ]


