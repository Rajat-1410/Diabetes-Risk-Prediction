import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

rf_model=joblib.load('Models/random_forest_model.pkl')
cox_male=joblib.load('Models/cox_male.pkl')
cox_female=joblib.load('Models/cox_female.pkl')

st.title('Type 2 Diabetes prediction System')
st.write('This system combines **Survival Analysis(Cox PH models)** and **Machine learning(Random Forest)** to estimate diabetes risk')

st.header('Patient Information')

age= st.number_input('Age',20,90)
gender= st.selectbox('Gender',['Male','Female'])
hba1c=st.number_input('HBA1c(Glycated Haemoglobin)')
pp2= st.number_input('Post prandial after 2 hours')
fbs=st.number_input('Fasting Blood Sugar')
bmi=st.number_input('Body Mass Index')
waist_hip=st.number_input('Waist to hip ratio')
triglyceride= st.number_input('Triglycerides level')
hdl=st.number_input('HDL Cholestrol')
chol_hdl=st.number_input('Chol/HDL')
ldl_hdl=st.number_input('LDL/HDL')
tobacco=st.selectbox('Tobacco Usage',['Never','Low','Moderate','High'])
alcohol=st.selectbox('Alcohol consumption',['Never','Low','Moderate','High'])
activity=st.selectbox('Physical Activity',['Low','Moderate'])

gender_val=1 if gender=='Male' else 0

Tobacco_map = {'Never':0,'Low':1,'Moderate':2,'High':3}
Alcohol_map = {'Never':0,'Low':1,'Moderate':2,'High':3}
Activity_map = {'Low':0,'Moderate':1}


tobacco=Tobacco_map[tobacco]
alcohol=Alcohol_map[alcohol]
activity=Activity_map[activity]

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
baseline = pd.DataFrame({
    "AGE":[50],
    "BMI":[25],
    "WAIST TO HIP RATIO":[0.9],
    "HBA1C":[5.5],
    "PP2 [<140 mg/dl]":[120],
    "FBS [74-110 mg/dl]":[95],
    "SERUM TRIGLYCERIDE [<150 mg/dl]":[130],
    "HDL CHOLESTEROL[200 mg/dl]":[45],
    "TOBACCO":[0],
    "PHYSICAL ACTIVITY":[1],
    "ALCOHOL":[0],
    "CHOL/HDL":[3],
    "LDL/HDL":[2]
})

patient = patient[rf_model.feature_names_in_]
if st.button('Predict Risk'):
    st.subheader('Machine Learning Prediction')

    ml_pred=rf_model.predict(patient)[0]
    ml_prob= rf_model.predict_proba(patient)[0][1]

    if ml_pred ==1:
        st.error(f'High Diabetes Risk(Probability: {ml_prob:.2f})')
    else:
        st.success(f'Low Diabetes Risk(Probability:{ml_prob:.2f})')
    if gender == 'Male':
        cox_model= cox_male
    else:
        cox_model=cox_female
    patient_cox = patient[cox_model.params_.index]
        
    st.subheader('Survival Analysis Risk')
    risk= cox_model.predict_partial_hazard(patient)

    st.write('Relative Risk Score:',round(float(risk),4))
    patient_survival = cox_model.predict_survival_function(patient)
    fig, ax= plt.plot()
    patient_survival.plot(ax=ax,label="Patient Risk",color="red")
    plt.title('Diabetes Risk Over Time')
    plt.xlabel('Age')
    plt.ylabel('Probability of Remaining Non-Diabetic')
    st.pyplot(fig)

    st.write(
            "The survival curve shows how the probability of remaining non-diabetic "
            "changes over time for the given patient characteristics."
        )

