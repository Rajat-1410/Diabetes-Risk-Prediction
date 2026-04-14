import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    rf = joblib.load('Models/random_forest_model.pkl')
    cox_m = joblib.load('Models/cox_male.pkl')
    cox_f = joblib.load('Models/cox_female.pkl')
    return rf, cox_m, cox_f

rf_model, cox_male, cox_female = load_models()

# -------------------- TITLE --------------------
st.title("🩺 Type 2 Diabetes Risk Prediction System")
st.markdown("""
Predict **current risk (Machine Learning)** and **future risk (Survival Analysis)**  
using clinical and lifestyle parameters.
""")

st.divider()

# -------------------- INPUT --------------------
st.header("Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 90, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    waist_hip = st.number_input("Waist-Hip Ratio", 0.5, 2.0, 0.9)

with col2:
    hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.5)
    fbs = st.number_input("Fasting Blood Sugar", 50, 300, 100)
    pp2 = st.number_input("Post-Prandial Sugar", 50, 400, 140)
    triglyceride = st.number_input("Triglycerides", 50, 600, 150)

with col3:
    hdl = st.number_input("HDL", 20, 100, 50)
    chol_hdl = st.number_input("Chol/HDL Ratio", 1.0, 10.0, 4.0)
    ldl_hdl = st.number_input("LDL/HDL Ratio", 0.5, 10.0, 2.5)
    tobacco = st.selectbox("Tobacco", ["Never","Low","Moderate","High"])
    alcohol = st.selectbox("Alcohol", ["Never","Low","Moderate","High"])
    activity = st.selectbox("Physical Activity", ["Low","Moderate"])

# -------------------- ENCODING --------------------
gender_val = 1 if gender == "Male" else 0

map_dict = {'Never':0,'Low':1,'Moderate':2,'High':3}
activity_map = {'Low':0,'Moderate':1}

tobacco = map_dict[tobacco]
alcohol = map_dict[alcohol]
activity = activity_map[activity]

# -------------------- DATAFRAME --------------------
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

patient = patient[rf_model.feature_names_in_]

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    if hba1c == 0 or bmi == 0:
        st.error("HbA1c and BMI must be non-zero.")
        st.stop()

    colA, colB = st.columns(2)

    # ---------- ML MODEL ----------
    with colA:
        st.subheader("📊 Current Risk (ML Model)")

        ml_pred = rf_model.predict(patient)[0]
        ml_prob = rf_model.predict_proba(patient)[0][1]

        if ml_prob > 0.7:
            st.error(f"🔴 High Risk ({ml_prob:.2%})")
        elif ml_prob > 0.4:
            st.warning(f"🟠 Moderate Risk ({ml_prob:.2%})")
        else:
            st.success(f"🟢 Low Risk ({ml_prob:.2%})")

        st.progress(float(ml_prob))

    # ---------- COX MODEL (CONSISTENT) ----------
    with colB:
        st.subheader("📈 Future Risk (Survival Analysis)")

        cox_model = cox_male if gender == "Male" else cox_female

        # Ensure correct column order
        patient_cox = patient[cox_model.params_.index]

        # Extract risk ONCE (stable)
        risk_patient = cox_model.predict_partial_hazard(patient_cox).values[0]

        if risk_patient > 1.5:
            st.error(f"🔴 High Relative Risk: {risk_patient:.2f}")
        elif risk_patient > 1:
            st.warning(f"🟠 Moderate Risk: {risk_patient:.2f}")
        else:
            st.success(f"🟢 Low Risk: {risk_patient:.2f}")

    # ---------- SURVIVAL CURVE (CONSISTENT) ----------
    st.subheader("📉 Survival Curve Comparison")

    baseline_survival = cox_model.baseline_survival_

    # Use SAME risk value (no recomputation)
    risk_clipped = np.clip(risk_patient, 0.5, 2.0)

    risk_low = risk_clipped * 0.8
    risk_high = risk_clipped * 1.2

    # Generate curves
    patient_survival = baseline_survival ** risk_clipped
    low_surv = baseline_survival ** risk_low
    high_surv = baseline_survival ** risk_high

    # Plot
    fig, ax = plt.subplots()

    baseline_survival.plot(ax=ax)
    patient_survival.plot(ax=ax)
    low_surv.plot(ax=ax)
    high_surv.plot(ax=ax)

    ax.set_title("Diabetes-Free Survival Probability")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survival Probability")

    ax.legend(["Population", "Patient", "Lower Risk", "Higher Risk"])

    st.pyplot(fig)

    # ---------- INTERPRETATION ----------
    st.expander("📘 How to understand this graph"):
    st.markdown("""
    This graph shows how diabetes risk may change with age.

    **Lines in the graph:**
    - **🔵 Blue (Population):** Average risk in the general population  
    - **🟠 Orange (You):** Your predicted risk  
    - **🟢 Green:** Lower-risk scenario  
    - **🔴 Red:** Higher-risk scenario  

    **How to read it:**
    - If your line is **below blue** → higher risk than average  
    - If your line is **above blue** → lower risk than average  
    - Faster drop → higher risk  

    ⚠️ This shows relative risk trends, not exact prediction.
    """)
st.markdown(" The Different variables and their meaning are:-")
st.markdown("""
    - **Age**: Used as a proxy for time in survival analysis  
    - **HbA1c(Glycated Haemoglobin )**: Long-term blood sugar indicator  
    - **FBS**: Fasting blood glucose level  
    - **PP2**: Blood sugar level after meals  
    - **BMI**: Indicator of body fat  
    - **Waist-Hip Ratio**: Measures fat distribution  
    - **Triglycerides**: Type of fat in blood  
    - **HDL**: “Good” cholesterol  
    - **Chol/HDL**: Cardiovascular risk indicator  
    - **LDL/HDL**: Balance of bad vs good cholesterol  
    - **Tobacco/Alcohol**: Lifestyle risk factors  
    - **Physical Activity**: Level of daily activity  
    """)
# -------------------- FOOTER --------------------
st.divider()
st.caption("⚠️ For educational purposes only. Not a medical diagnosis tool.")
st.caption("Note: Age is used as a proxy for time. Curves are scaled for interpretability.")
