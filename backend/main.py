from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Create FastAPI App
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API that detects fraudulent credit card transactions using Machine Learning",
    version="1.0.0"
)

# Load Model and Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "robust_scaler.pkl"))
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Endpoint 1  Home

@app.get("/")
def home():
    return {
        "message": "Welcome to Credit Card Fraud Detection API!",
        "version": "1.0.0",
        "team": "Ptechsoft",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

# Endpoint 2 Health Check
@app.get("/health")
def health():
    return {
        "status": "API is running",
        "model": "XGBoost Classifier",
        "port": 8007
    }

# Pydantic Input Schema 30 Transaction Features

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Endpoint 3 Predict Fraud

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Step 1 — Convert input to numpy array
        data = transaction.dict()
        input_array = np.array(list(data.values())).reshape(1, -1)

        # Step 2 — Scale only Time and Amount
        time_amount = np.array([[data['Time'], data['Amount']]])
        scaled_time_amount = scaler.transform(time_amount)

        # Step 3 — Replace Time and Amount with scaled values
        input_array[0][0] = scaled_time_amount[0][0]
        input_array[0][29] = scaled_time_amount[0][1]

        # Step 4 — Make prediction
        probability = model.predict_proba(input_array)[0][1]

        # Step 5 — Use custom threshold of 0.3
        # Default threshold is 0.5 but fraud detection needs lower threshold
        # to catch more frauds
        threshold = 0.3
        prediction = 1 if probability >= threshold else 0

        # Step 6 — Return result
        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "result": "🚨 FRAUDULENT TRANSACTION DETECTED" if prediction == 1 else "✅ LEGITIMATE TRANSACTION",
            "confidence": f"{round(float(probability) * 100, 2)}%",
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )