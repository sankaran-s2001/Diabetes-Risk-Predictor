import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load model and scaler
scaler = joblib.load('./scaler.pkl')
model = joblib.load('./model.pkl')

# Page configuration
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", layout="centered")

# Custom styling
st.markdown("""
<style>
.warning {
    background-color: rgba(255, 165, 0, 0.1); /* Soft orange background */
    color: #FFD700; /* Bright yellow text */
    padding: 15px;
    border-left: 6px solid #FFA500;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: 500;
}
.safe {
    background-color: rgba(0, 128, 0, 0.1); /* Soft green background */
    color: #00FF7F; /* Bright green text */
    padding: 15px;
    border-left: 6px solid #32CD32;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: 500;
}
h4 {
    color: #FFFFFF;  /* White headline */
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #00FFEF;  /* Bright cyan for dark theme */
    margin-bottom: 30px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)

st.write("""
This tool uses a machine learning model to assess your risk of diabetes based on clinical data. 
Please fill in the details below to receive your prediction.
""")

st.sidebar.title("📘 Feature Information")

st.sidebar.markdown("""
**🔹 Pregnancies**  
*How many times the woman has been pregnant*  
📈 eg values: 0 to 5  

**🔹 Glucose**  
*Sugar level in blood (tested after 2 hours)*  
📈 eg values: 0 to 199 (normal: 70–140)  

**🔹 BloodPressure**  
*Pressure in your blood vessels (diastolic)*  
📈 eg values: 0 to 122 (normal: 60–80)  

**🔹 SkinThickness**  
*Skin fold thickness at triceps (body fat measure)*  
📈 eg values: 0 to 99 mm  

**🔹 Insulin**  
*Insulin level (2-hour serum insulin)*  
📈 eg values: 0 to 846 μU/ml  

**🔹 BMI**  
*Body Mass Index (weight/height)*  
📈 eg values: 0 to 67.1 (normal: 18.5–24.9)  

**🔹 DiabetesPedigreeFunction**  
*Genetic diabetes risk*  
📈 eg values: 0.078 to 2.42  

**🔹 Age**  
*Person’s age in years*  
📈 eg values: 21 to 81  
 
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Developed by <b><i>Sankaran S</i></b></small>",
    unsafe_allow_html=True
)


# Input form
with st.form("prediction_form"):
    st.subheader("📋 Enter Your Health Details")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 90)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", 0, 1000, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age", 1, 120, 30)
    
    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
    input_df = pd.DataFrame([features], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    diabetic_risk = proba[1] * 100

    st.subheader("📊 Prediction Results")
    st.metric(
        label="Prediction",
        value="Diabetic" if prediction == 1 else "Not Diabetic",
        delta=f"{diabetic_risk:.1f}% Risk"
    )

    # Message blocks
    if prediction == 0 and diabetic_risk > 35:
        st.markdown("""
        <div class="warning">
            <h4>⚠️ At Risk of Prediabetes</h4>
            <ul>
                <li>Risk detected: {:.1f}%</li>
                <li>Consider an HbA1c test</li>
                <li>Improve lifestyle and monitor health</li>
            </ul>
        </div>
        """.format(diabetic_risk), unsafe_allow_html=True)
    
    elif prediction == 1 and diabetic_risk < 50:
        st.markdown("""
        <div class="warning">
            <h4>⚠️ Borderline Diabetes Detected</h4>
            <ul>
                <li>Early intervention is key</li>
                <li>Consult your doctor</li>
                <li>Review medications and diet</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif prediction == 1:
        st.markdown("""
        <div class="warning">
            <h4>🚨 High Diabetes Risk</h4>
            <ul>
                <li>Immediate medical advice recommended</li>
                <li>Strict dietary and fitness plan needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe">
            <h4>✅ Low Diabetes Risk</h4>
            <p>Your current estimated risk is <b>{diabetic_risk:.1f}%</b>.</p>
            <p>Keep up a healthy lifestyle!</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature impact bar chart
    st.subheader("📌 Feature Impact")
    impact_df = pd.DataFrame({
        'Feature': input_df.columns,
        'Impact': abs(scaled[0])
    }).sort_values(by='Impact', ascending=False)
    
    st.bar_chart(impact_df.set_index("Feature"))

# Footer
st.markdown("---")
st.caption("⚠️ The results provided by this tool are for informational use only. Always consult a certified medical professional for accurate diagnosis and treatment.")
