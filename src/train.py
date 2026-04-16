import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess_data(df):
    """
    Simple preprocessing for Telco Churn dataset.
    - Handle missing values in TotalCharges.
    - Label encode categorical variables.
    - Drop customerID.
    """
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.fillna(df['TotalCharges'].mean())
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

def train_model():
    # 1. Start MLflow Run
    mlflow.set_experiment("Telco_Churn_Experiment")
    
    with mlflow.start_run():
        # 2. Load and Preprocess
        data_path = "data/raw_telco_churn.csv"
        if not os.path.exists(data_path):
            print("Data file not found. Please run src/download_data.py first.")
            return
            
        df = pd.read_csv(data_path)
        df_processed = preprocess_data(df)
        
        # 3. Split Data
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Define and Train Model
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Model trained. Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        # 6. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, "churn_model_xgboost")
        
        # 7. Local Save for API/Explainability
        with open("data/model.pkl", "wb") as f:
            pickle.dump(model, f)
            
        print("Model and metrics logged to MLflow and saved to data/model.pkl.")

if __name__ == "__main__":
    train_model()
