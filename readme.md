# Diabetes Risk Prediction using Survival Analysis and Machine Learning

## Overview

This project develops a predictive framework for assessing the risk of Type 2 Diabetes using both statistical survival analysis and machine learning techniques. The aim of the study is to analyze clinical and lifestyle indicators associated with diabetes and to build models that can help identify individuals who are currently at risk as well as those who may develop the disease earlier than others.

The project combines two analytical approaches. Survival analysis using the Cox Proportional Hazard model is used to estimate the relative progression risk of diabetes over time, while machine learning models are used to classify whether a patient belongs to a diabetic risk group based on metabolic and lifestyle characteristics.

By integrating both approaches, the system provides a more comprehensive understanding of diabetes risk from both a predictive and temporal perspective.


## Objectives

The main objectives of this project are:

- Identify metabolic and lifestyle factors associated with Type 2 Diabetes.
- Estimate relative diabetes risk using Cox Proportional Hazard models.
- Compare multiple machine learning algorithms for diabetes risk classification.
- Build a simple predictive system where users can input patient data and obtain risk estimates.
- Demonstrate an end-to-end data science workflow including data processing, database management, statistical modelling, machine learning and application development.


## Dataset

The dataset used in this project contains demographic, metabolic and lifestyle variables associated with diabetes risk. The data was processed through multiple stages:

1. Raw dataset collection  
2. Data cleaning and preprocessing  
3. Feature transformation and encoding  
4. Storage of structured datasets in a SQLite database  

Multiple versions of the dataset were stored as tables in the database to represent different stages of preprocessing.

Key variables used in the analysis include:

- Age
- Gender
- Body Mass Index (BMI)
- Waist–Hip Ratio
- HbA1c
- Fasting Blood Sugar (FBS)
- Post-Prandial Blood Sugar (PP2)
- Serum Triglycerides
- HDL Cholesterol
- Lifestyle indicators such as tobacco use, alcohol consumption and physical activity.


## Methodology

### Survival Analysis

Cox Proportional Hazard models were used to estimate the relative hazard of developing diabetes. Separate models were constructed for male and female populations in order to capture potential gender-specific risk patterns.

The model estimates a relative hazard value that indicates how much faster or slower an individual may develop diabetes compared to a baseline individual.

### Machine Learning Models

Several machine learning algorithms were trained and evaluated for diabetes classification:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine  
- Gaussian Naive Bayes  

Model performance was evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- F1 Score  
- Area Under the ROC Curve (AUC)

Among the evaluated models, Random Forest demonstrated the strongest predictive performance.


## Database Design

To simulate a real-world data pipeline, the processed datasets were stored in a SQLite database. Different stages of the data processing workflow were stored as separate tables, including raw data, cleaned data and encoded datasets used for modelling.

This structure allows analysis notebooks to retrieve data directly from the database rather than relying on spreadsheet files.


## Application

A simple interactive application was developed using Streamlit to demonstrate how the trained models can be used in practice.

The application allows users to enter patient information and obtain:

- Machine learning based diabetes risk prediction  
- Estimated relative hazard from the Cox survival model  
- A survival probability curve showing risk progression over time  

This demonstrates how statistical models and machine learning techniques can be integrated into a practical decision support tool.


## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Lifelines (Survival Analysis)  
- SQLite  
- Streamlit  
- Matplotlib  
- Seaborn  


## Project Structure

Diabetes_Risk_Prediction_Project

Data  
- Raw and processed datasets  
- SQLite database  

Notebooks  
- Data preprocessing and database creation  
- Cox survival analysis models  
- Machine learning model comparison  

Models  
- Random Forest classifier  
- Cox model for male population  
- Cox model for female population  

Output  
- Model evaluation plots and results  

Application  
- Streamlit interface for diabetes risk prediction


## Key Insights

The analysis highlights several metabolic indicators associated with increased diabetes risk. The machine learning models successfully identify patients who belong to a diabetic risk group, while the Cox survival models provide insight into how quickly diabetes may develop for different individuals.

By combining both approaches, the system supports both early detection and risk progression analysis, which may assist healthcare professionals in identifying individuals who require early intervention.


## Future Improvements

Future improvements to this project may include:

- Using larger and more diverse datasets
- Incorporating additional clinical variables
- Deploying the predictive system as a publicly accessible web application
- Improving model interpretability using explainable AI techniques
