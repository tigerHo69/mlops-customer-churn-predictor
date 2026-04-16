import pandas as pd
import os

def download_data():
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_path = "data/raw_telco_churn.csv"
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    print(f"Downloading data from {url}...")
    df = pd.read_csv(url)
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path}")
    print(f"Dataset Shape: {df.shape}")
    
if __name__ == "__main__":
    download_data()
