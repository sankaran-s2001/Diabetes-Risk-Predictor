# 🩺 Diabetes Risk Prediction System

![Project Banner](https://static.vecteezy.com/system/resources/previews/046/262/714/non_2x/diabetes-pink-word-concept-chronic-illness-symptoms-and-treatment-blood-glucose-levels-typography-banner-illustrationwith-title-text-editable-icons-color-vector.jpg)

A machine learning-powered web application that predicts diabetes risk using key health indicators. Built with **Python**, **scikit-learn**, and **Streamlit**, this tool provides a real-time and interactive risk assessment interface — including a warning system for *prediabetes*.

---

## 📌 Table of Contents
- [🔍 Project Overview](#project-overview)
- [📊 Dataset](#dataset)
- [🧠 Technical Approach](#technical-approach)
- [⚙️ Installation](#installation)
- [🚀 Usage](#usage)
- [✨ Features](#features)
- [📈 Model Performance](#model-performance)
- [🖼️ Screenshots](#screenshots)
- [🔧 Future Improvements](#future-improvements)
- [📄 License](#license)

---

## 🔍 Project Overview

This project demonstrates a complete **end-to-end ML pipeline** to predict diabetes:

1. **Data preprocessing** and handling missing values
2. **Feature engineering** using medical thresholds
3. **Model training** and evaluation (Support Vector Machine - SVM)
4. **Deployment** using Streamlit
5. **Prediabetes warning** system integration

> ⚠️ **Disclaimer**: This tool is intended for **educational purposes only**. Always consult a certified healthcare professional for medical advice and diagnosis.

---

## 📊 Dataset

**📁 Source**: [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Feature            | Description                                 | Medical Threshold         |
|--------------------|---------------------------------------------|---------------------------|
| `Pregnancies`      | Number of pregnancies                       | -                         |
| `Glucose`          | Plasma glucose concentration (mg/dL)        | ≥140 → Prediabetes        |
| `BloodPressure`    | Diastolic blood pressure (mmHg)             | ≥90 → Hypertension        |
| `SkinThickness`    | Triceps skinfold thickness (mm)             | -                         |
| `Insulin`          | 2-Hour serum insulin (μU/mL)                | -                         |
| `BMI`              | Body mass index (kg/m²)                     | ≥30 → Obesity             |
| `DiabetesPedigree` | Diabetes pedigree function                  | -                         |
| `Age`              | Patient age in years                        | -                         |
| `Outcome`          | Target class (0: Non-diabetic, 1: Diabetic) | -                         |

**Data Preprocessing**:
- Replaced zeros in `Glucose`, `BloodPressure`, `BMI`, and `Insulin` with median values
- Addressed **class imbalance** (500:268) using resampling

---

## 🧠 Technical Approach

- **Language**: Python
- **Frameworks & Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` for model building (SVM)
  - `joblib` for model serialization
  - `Streamlit` for UI

- **Pipeline**:
  - Scaling using `StandardScaler`
  - Training using **SVM** (Support Vector Machine)
  - Predicting the probability of diabetes
  - Categorizing risk:
    - **Low Risk**: 0–35%
    - **Prediabetic Warning**: 35–50%
    - **High Risk**: 50%+

---

## ⚙️ Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# Step 2: Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install required libraries
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run app.py
