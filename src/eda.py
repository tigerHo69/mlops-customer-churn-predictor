import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def run_eda():
    # 1. Load Data
    df = pd.read_csv('data/raw_telco_churn.csv')
    print("Dataset Loaded Successfully")
    
    # 2. Inspect Data
    print("\n--- Basic Info ---")
    print(df.info())
    
    # 3. Data Cleaning
    # Convert TotalCharges to numeric (handle empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']) # Dropping the few rows with missing TotalCharges
    
    # 4. Target Variable Analysis
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Churn Distribution (Target Variable)')
    plt.savefig('data/churn_distribution.png')
    
    # 5. Numerical Features Analysis
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, feature in enumerate(numerical_features):
        sns.histplot(data=df, x=feature, hue='Churn', kde=True, ax=axes[i], palette='magma')
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('data/numerical_distributions.png')

    # 6. Categorical Features Analysis
    # Let's look at Contract type and Internet Service
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0], palette='pastel')
    axes[0].set_title('Churn by Contract Type')
    sns.countplot(x='InternetService', hue='Churn', data=df, ax=axes[1], palette='pastel')
    axes[1].set_title('Churn by Internet Service')
    plt.savefig('data/categorical_churn.png')

    print("\nEDA plots saved to the data/ directory.")

if __name__ == "__main__":
    try:
        run_eda()
    except Exception as e:
        print(f"Error running EDA: {e}")
        print("Note: Ensure you have installed requirements and run the download script first.")
