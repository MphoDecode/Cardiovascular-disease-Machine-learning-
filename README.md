# CardioCare – Cardiovascular Disease Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-lightgrey.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

**CardioCare** is a supervised machine learning pipeline that predicts cardiovascular disease (CVD) onset using clinical, genetic, and lifestyle data. The project compares five classification models and implements a stacked ensemble to achieve optimal predictive performance.

CVD accounts for nearly 18 million deaths annually worldwide – 32% of all global deaths. Up to 80% of premature CVD mortality is preventable with early detection. CardioCare aims to identify high-risk individuals before symptoms appear, enabling timely intervention.

## Key Results

| Metric | Stacked Ensemble Performance |
|--------|------------------------------|
| **Accuracy** | 85% |
| **Sensitivity** | 91% |
| **Specificity** | 74% |
| **Precision** | 91% |
| **F1 Score** | 77% |
| **AUC-ROC** | 0.89 |

### Critical Clinical Insight

The model identified **exercise-induced angina** and **ST depression** as stronger predictors of heart disease than traditional factors like cholesterol alone – a finding that could reshape early screening protocols.

## Models Compared

- Logistic Regression (baseline)
- Decision Tree
- AdaBoost
- Neural Network (1 hidden layer)
- **Stacked Ensemble** (meta-classifier combining top models)

## Dataset

Public dataset containing **13 clinical features**:

| Feature | Description |
|---------|-------------|
| Age | Patient age (numeric) |
| Sex | Male / Female |
| Chest Pain Type | 4 categories |
| BP | Blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar >120 mg/dl |
| EKG Results | Resting electrocardiogram results |
| Max HR | Maximum heart rate achieved |
| Exercise Angina | Angina induced by exercise |
| ST Depression | ST segment depression |
| Slope of ST | Slope of ST segment |
| Number of Vessels | Vessels seen on fluoroscopy |
| Thallium | Thallium stress test results |

**Target variable:** Heart Disease diagnosis (binary)

## Project Structure
