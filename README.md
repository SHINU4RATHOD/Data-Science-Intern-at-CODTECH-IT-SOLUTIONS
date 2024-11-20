# Data-Science-Intern-at-CODTECH-IT-SOLUTIONS

# Visa Approval Prediction
### Table of Contents
1. Abstract
2. Introduction
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Building and Training
6. Results and Discussion
7. Conclusion
8. Future Work
9. Appendix


# Abstract
Visa applications for employment, study, and travel often involve a complex approval process. This project analyzes historical visa application data to identify factors influencing approval outcomes and builds predictive models to assist in the decision-making process.


# Introduction
## Background
Visa approval is influenced by multiple factors such as wages, job titles, and employers. This project aims to explore these patterns and predict outcomes using machine learning models.

## Objective
* Perform exploratory data analysis (EDA) to extract meaningful patterns.
* Engineer features to enhance prediction accuracy.
* Build machine learning models to predict visa approval status.
* Provide actionable insights based on model results.


# Exploratory Data Analysis (EDA)
## Dataset Overview
#### Features:
* Case_Status: Target variable indicating approval or denial.
* Attributes include employer name, job title, prevailing wage, and application date.

## Key Insights
1. Approved cases have higher average wages than denied cases.
2. Job titles significantly influence approval rates.
3. Certain employers have consistently high approval rates.

#### Visualizations:
* Distribution plots for wages and approval trends.
* Heatmaps to identify correlations between features.


# Feature Engineering
### Steps Taken
1. Data Cleaning: Handled missing values and standardized employer and job title names.
2. Feature Creation: Generated new features like Wage_Per_Hour for better insights.
3. Encoding and Scaling: Categorical variables were one-hot encoded, and numeric features were normalized.
** Challenge Addressed: Balanced the dataset using SMOTE to handle class imbalance.


# Model Building and Training
### Models Implemented
* Logistic Regression: Baseline model for comparison.
* Random Forest: Captured non-linear relationships effectively.
* XGBoost: Optimized model for best performance.

# Performance Metrics
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	82%	80%	78%	79%	0.85
Random Forest	88%	86%	85%	85%	0.91
XGBoost	91%	89%	88%	89%	0.93



# Results and Discussion
### Key Findings:
* High wages and specific job titles significantly influence approval.
* XGBoost achieved the best performance with a ROC-AUC score of 0.93.
### Challenges:
Standardizing employer names and handling class imbalance were key hurdles.


# Conclusion
This project demonstrates the power of data-driven approaches to understanding and predicting visa approvals. By leveraging EDA, feature engineering, and machine learning, i have developed a framework capable of making accurate predictions.


# Appendix
* EDA Notebook: Detailed data exploration and visualizations.
* Feature Engineering Notebook: Steps for data cleaning and feature creation.
* Model Training Notebook: Implementation and evaluation of machine learning models.


