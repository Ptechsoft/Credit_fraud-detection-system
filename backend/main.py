from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Credit Card Fraud Detection API

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

# Pydantic Input Schema — 30 Transaction Features
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

# Endpoint 1 — Home

@app.get("/")
def home():
    return {
        "message": "Welcome to Credit Card Fraud Detection API!",
        "version": "1.0.0",
        "team": "3MTT C2 DATA SCIENCE(112)",
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
        "model": "Random Forest Classifier",
        "port": 8007
    }


# Endpoint 3 Predict Fraud
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        #  Get all transaction values
        data = transaction.dict()

        # Scale Amount and Time using Patrick's RobustScaler
        amount_time = np.array([[data['Amount'], data['Time']]])
        scaled = scaler.transform(amount_time)
        scaled_amount = scaled[0][0]
        scaled_time = scaled[0][1]

        # Build feature array in correct order
        features = []
        for i in range(1, 29):
            features.append(data[f'V{i}'])
        features.append(scaled_amount)
        features.append(scaled_time)

        input_array = np.array(features).reshape(1, -1)

        #  Make prediction using Random Forest model
        probability = model.predict_proba(input_array)[0][1]

        # Use custom threshold of 0.3
        threshold = 0.3
        prediction = 1 if probability >= threshold else 0

        #  Return prediction result
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
