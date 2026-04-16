# Telco Customer Churn Prediction & MLOps

An end-to-end data science project demonstrating predictive modeling, model interpretability, and MLOps principles.

## Project Overview
This project predicts whether a customer will churn (cancel their service) based on demographics, account info, and usage patterns.

### Key Features
- **Exploratory Data Analysis (EDA)**: Understanding trends and correlations.
- **MLflow Integration**: Tracking experiments, parameters, and metrics.
- **Model Explainability**: Using SHAP to interpret local and global model behavior.
- **API Deployment**: A FastAPI serving layer for real-time predictions.
- **Interactive UI**: A Streamlit dashboard for business users.

## Project Structure
```text
├── app/            # Streamlit dashboard
├── data/           # Local data storage
├── notebooks/      # Jupyter notebooks for EDA and R&D
├── src/            # Core logic (training, inference, API)
├── requirements.txt
└── README.md
```

## Getting Started
1. Install dependencies: `python3 -m pip install -r requirements.txt`
2. Run EDA: `python3 src/eda.py`
3. Train model: `python3 src/train.py`
4. Run API: `python3 src/api.py`
5. Run Dashboard: `python3 -m streamlit run app/dashboard.py`
