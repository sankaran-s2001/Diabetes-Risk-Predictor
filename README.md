# Diabetes Risk Predictor

![Project Banner](https://via.placeholder.com/800x200/006064/FFFFFF?text=Diabetes+Risk+Prediction+System)

A machine learning-based web application that predicts diabetes risk using health metrics, with Streamlit UI and a prediabetes warning system.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
This project demonstrates an end-to-end machine learning pipeline for diabetes prediction:
1. Data preprocessing and feature engineering
2. Model training and evaluation
3. Streamlit web interface deployment
4. Risk stratification with clinical thresholds

**Key Highlights**:
- Achieved 77.3% accuracy with SVM
- Implemented prediabetes warning system (35-50% risk threshold)

## Dataset
**Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Features**:
| Column | Description | Medical Threshold |
|--------|-------------|-------------------|
| Pregnancies | Number of pregnancies | - |
| Glucose | Plasma glucose concentration (mg/dL) | ≥140 (Prediabetes) |
| BloodPressure | Diastolic BP (mmHg) | ≥90 (Hypertension) |
| SkinThickness | Triceps skinfold thickness (mm) | - |
| Insulin | 2-Hour serum insulin (μU/mL) | - |
| BMI | Body mass index (kg/m²) | ≥30 (Obese) |
| DiabetesPedigree | Diabetes likelihood function | - |
| Age | Years | - |
| Outcome | Class variable (0/1) | - |

**Data Challenges**:
- Handled impossible zero values (Glucose=0 → median)
- Addressed class imbalance (500:268 ratio)

## Technical Approach
### 1. Data Preprocessing
```python
# Handle missing values disguised as 0s
for col in ['Glucose', 'BloodPressure']:
    df[col] = df[col].replace(0, df[col].median())
