import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
rf_model = joblib.load('Models/random_forest_model.pkl')
cox_male = joblib.load('Models/cox_male.pkl')
cox_female = joblib.load('Models/cox_female.pkl')

# Title
st.title("Type 2 Diabetes Risk Prediction System")

st.write("""
This application estimates the **risk of Type 2 Diabetes** using two analytical approaches:

• **Machine Learning (Random Forest)** — predicts whether an individual is currently at high risk.  
• **Survival Analysis (Cox Proportional Hazard Model)** — estimates the **future progression risk** of diabetes.

The system uses metabolic indicators and lifestyle factors commonly associated with diabetes development.
""")

st.markdown("---")

# Patient information
st.header("Patient Information")

age = st.number_input("Age (years)", 20, 90)

gender = st.selectbox("Gender", ["Male","Female"])

hba1c = st.number_input("HbA1c (Average blood glucose over 3 months)", step=0.1)

pp2 = st.number_input("Post Prandial Blood Sugar (2 hours after meal)")

fbs = st.number_input("Fasting Blood Sugar")

bmi = st.number_input("Body Mass Index (BMI)")

waist_hip = st.number_input("Waist to Hip Ratio")

triglyceride = st.number_input("Triglyceride Level")

hdl = st.number_input("HDL Cholesterol")

chol_hdl = st.number_input("Total Cholesterol / HDL Ratio")

ldl_hdl = st.number_input("LDL / HDL Ratio")

tobacco = st.selectbox("Tobacco Usage", ["Never","Low","Moderate","High"])

alcohol = st.selectbox("Alcohol Consumption", ["Never","Low","Moderate","High"])

activity = st.selectbox("Physical Activity", ["Low","Moderate"])

# Encoding
gender_val = 1 if gender == "Male" else 0

Tobacco_map = {'Never':0,'Low':1,'Moderate':2,'High':3}
Alcohol_map = {'Never':0,'Low':1,'Moderate':2,'High':3}
Activity_map = {'Low':0,'Moderate':1}

tobacco = Tobacco_map[tobacco]
alcohol = Alcohol_map[alcohol]
activity = Activity_map[activity]

# Create dataframe
patient = pd.DataFrame({

"AGE":[age],
"GENDER":[gender_val],
"HBA1C":[hba1c],
"PP2 [<140 mg/dl]":[pp2],
"FBS [74-110 mg/dl]":[fbs],
"BMI":[bmi],
"WAIST TO HIP RATIO":[waist_hip],
"TOBACCO":[tobacco],
"PHYSICAL ACTIVITY":[activity],
"ALCOHOL":[alcohol],
"SERUM TRIGLYCERIDE [<150 mg/dl]":[triglyceride],
"HDL CHOLESTEROL[200 mg/dl]":[hdl],
"CHOL/HDL":[chol_hdl],
"LDL/HDL":[ldl_hdl]

})

# Align ML features
patient = patient[rf_model.feature_names_in_]

if st.button("Predict Risk"):

    st.subheader("Machine Learning Prediction")

    ml_pred = rf_model.predict(patient)[0]
    ml_prob = rf_model.predict_proba(patient)[0][1]

    if ml_pred == 1:
        st.error(f"High Diabetes Risk (Probability: {ml_prob:.2f})")
    else:
        st.success(f"Low Diabetes Risk (Probability: {ml_prob:.2f})")

    st.write("""
The machine learning model estimates the probability that the individual belongs
to the diabetes risk group based on metabolic indicators.
""")

    # Choose Cox model
    if gender == "Male":
        cox_model = cox_male
    else:
        cox_model = cox_female

    patient_cox = patient[cox_model.params_.index]

    st.subheader("Survival Analysis Risk")

    risk = cox_model.predict_partial_hazard(patient_cox)

    st.write("Relative Risk Score:", round(float(risk),4))

    st.write("""
A relative risk value greater than **1** indicates higher likelihood of developing diabetes
compared to the average population, while values below **1** indicate lower risk.
""")

    # Survival curve
    patient_survival = cox_model.predict_survival_function(patient_cox)

    fig, ax = plt.subplots()

    patient_survival.plot(ax=ax,label="Patient Risk",color="red")

    plt.title("Probability of Remaining Diabetes-Free Over Time")

    plt.xlabel("Age")

    plt.ylabel("Probability of Remaining Non-Diabetic")

    st.pyplot(fig)

    st.write("""
The survival curve represents how the probability of remaining diabetes-free
changes as age increases for the given patient characteristics.
A steeper decline indicates higher future risk.
""")

st.markdown("---")

st.write("""
**Disclaimer**

This tool is developed for research and educational purposes.
It should not be used as a substitute for professional medical diagnosis or treatment.
""")