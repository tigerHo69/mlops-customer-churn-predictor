import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard predicts the probability of a customer churning using a trained XGBoost model.
It communicates with a FastAPI backend for real-time inference.
""")

# Sidebar for inputs
st.sidebar.header("Customer Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender (0: Female, 1: Male)", (0, 1))
    senior = st.sidebar.selectbox("Senior Citizen", (0, 1))
    partner = st.sidebar.selectbox("Partner", (0, 1))
    dependents = st.sidebar.selectbox("Dependents", (0, 1))
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone = st.sidebar.selectbox("Phone Service", (0, 1))
    multiple = st.sidebar.selectbox("Multiple Lines", (0, 1, 2))
    internet = st.sidebar.selectbox("Internet Service (0: DSL, 1: Fiber, 2: No)", (0, 1, 2))
    security = st.sidebar.selectbox("Online Security", (0, 1, 2))
    backup = st.sidebar.selectbox("Online Backup", (0, 1, 2))
    protection = st.sidebar.selectbox("Device Protection", (0, 1, 2))
    support = st.sidebar.selectbox("Tech Support", (0, 1, 2))
    tv = st.sidebar.selectbox("Streaming TV", (0, 1, 2))
    movies = st.sidebar.selectbox("Streaming Movies", (0, 1, 2))
    contract = st.sidebar.selectbox("Contract (0: Month-to-month, 1: One year, 2: Two year)", (0, 1, 2))
    billing = st.sidebar.selectbox("Paperless Billing", (0, 1))
    payment = st.sidebar.selectbox("Payment Method", (0, 1, 2, 3))
    monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total = st.sidebar.slider("Total Charges ($)", 18.0, 9000.0, 800.0)
    
    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": billing,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    return data

input_data = user_input_features()

# Main UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Request")
    st.write(pd.DataFrame([input_data]))
    
    if st.button("Predict Churn Risk"):
        try:
            # API Call
            response = requests.post("http://localhost:8000/predict", json=input_data)
            result = response.json()
            
            prob = result['churn_probability']
            risk = result['message']
            
            st.metric("Churn Probability", f"{prob*100:.2f}%")
            if prob > 0.5:
                st.error(f"Prediction: {risk}")
            else:
                st.success(f"Prediction: {risk}")
                
        except Exception as e:
            st.warning("Could not connect to FastAPI. Is 'python src/api.py' running?")

with col2:
    st.subheader("Model Interpretability (SHAP)")
    shap_path = "data/shap_summary_plot.png"
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Global Feature Importance")
    else:
        st.info("Run 'python src/explain.py' to generate SHAP visualizations.")

st.divider()
st.subheader("Historical Context (EDA)")
eda_path = "data/numerical_distributions.png"
if os.path.exists(eda_path):
    st.image(eda_path, caption="Distribution of key features in training data")
