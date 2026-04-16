import requests
import json

def test_api():
    url = "http://localhost:8000/predict"
    
    # Mock customer data (using numeric codes based on LabelEncoding)
    sample_payload = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "OnlineBackup": 1,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 0,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=sample_payload)
        response.raise_for_status()
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure 'python src/api.py' is running in another terminal.")

if __name__ == "__main__":
    test_api()
