import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def run_explainability():
    print("Generating SHAP explainability plots...")
    
    # 1. Load Data
    data_path = "data/raw_telco_churn.csv"
    if not os.path.exists(data_path):
        print("Data file not found. Please run src/download_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Simple preprocessing (Sync with train.py)
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.fillna(df['TotalCharges'].mean())
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Load the trained model from disk
    model_path = "data/model.pkl"
    if not os.path.exists(model_path):
        print("Model file not found. Please run src/train.py first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # 3. SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 4. Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance (Impact on Churn)")
    
    # Save the plot
    plot_path = "data/shap_summary_plot.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to {plot_path}")
    
    # 5. Global Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("data/shap_feature_importance_bar.png", bbox_inches='tight')
    print("SHAP global importance bar plot saved to data/shap_feature_importance_bar.png")

if __name__ == "__main__":
    try:
        run_explainability()
    except Exception as e:
        print(f"Error generating SHAP values: {e}")
        print("Note: This script requires 'shap' and 'xgboost' installed.")
