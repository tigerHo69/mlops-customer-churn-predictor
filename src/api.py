from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Telco Churn Prediction API")

# Model Loading
MODEL_PATH = "data/model.pkl"

def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Input Schema
class CustomerData(BaseModel):
    gender: int  # 0 or 1
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is running. Visit /docs for Swagger UI."}

@app.post("/predict")
def predict(data: CustomerData):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Please run training first.")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "message": "High risk of churn" if prediction == 1 else "Low risk of churn"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
