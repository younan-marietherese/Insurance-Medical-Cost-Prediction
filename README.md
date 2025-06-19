# Insurance Medical Cost Prediction

A machine learning project that predicts individual medical insurance charges and smoker status based on demographic and health-related data. The project walks through the complete Data Science Lifecycle using both custom and library-based implementations of regression and classification models.

---

## Overview

This notebook-based project explores a U.S. health insurance dataset to uncover relationships between factors like age, BMI, smoking habits, and medical charges. It is divided into two main parts:

- **Part 1**: Predicting medical charges (a regression task)
- **Part 2**: Predicting smoker status (a classification task)

Both parts include implementations using:
- Gradient Descent from scratch (loop-based and vectorized)
- Scikit-learn models for comparison

---

## Objective

- **Regression Task**: Predict the `charges` (individual medical costs).
- **Classification Task**: Predict the `smoker` status (yes/no).

---

## Dataset Description

The dataset includes the following features:

- `age`: Age of the beneficiary  
- `sex`: Gender (male/female)  
- `bmi`: Body Mass Index  
- `children`: Number of dependents  
- `smoker`: Smoking status (yes/no)  
- `region`: Residential region  
- `charges`: Annual medical costs (target for regression)

---

## Exploratory Data Analysis (EDA)

- Descriptive statistics and data types explored using `info()` and `describe()`.
- Pairplots and histograms used to visualize relationships between variables.
- Found that **smoker status** and **BMI** are strong influencers of insurance charges.

---

## Models & Methodology

### Part 1: Medical Charges Prediction

- **Model 1**: Linear Regression from scratch using Gradient Descent (loops → vectorized)
- **Model 2**: Linear Regression using `scikit-learn`

### Part 2: Smoker Status Prediction

- **Model 1**: Logistic Regression from scratch using Gradient Descent
- **Model 2**: Logistic Regression using `scikit-learn`

---

## Results & Evaluation

- **Medical Charges Prediction**:
  - High accuracy using both models
  - Gradient Descent (vectorized) yielded performance close to `scikit-learn`
- **Smoker Status Prediction**:
  - Achieved strong classification results
  - Important features: `charges`, `bmi`, and `age`
  - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Technologies Used

- **Language**: Python  
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Future Enhancements

- Add regularization (Ridge/Lasso) to improve generalization
- Extend classification to predict obesity risk
- Deploy models via Flask or Streamlit for user-friendly interface
