import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# -----------------------
# Load Model
# -----------------------
try:
    model = joblib.load("payment_failure_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# -----------------------
# Initialize App
# -----------------------
app = FastAPI(title="Payment Failure Prediction API", description="API to predict payment failure risk based on transaction metadata.")

# -----------------------
# Request Schema
# -----------------------
class PredictionRequest(BaseModel):
    occupation: str
    purposeOfTransaction: str
    sourceOfFunds: str
    countryOfBirth: str
    nationality: str
    receiver_address_countryCode: str # Mapped from receiver.address.countryCode
    age: int
    id_verified: int # 0 or 1
    cross_border: int # 0 or 1

class PredictionResponse(BaseModel):
    payment_failure_risk: int
    failure_probability: float

# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def home():
    return {"message": "Payment Failure Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    
    # Create DataFrame from request
    data = {
        "occupation": [request.occupation],
        "purposeOfTransaction": [request.purposeOfTransaction],
        "sourceOfFunds": [request.sourceOfFunds],
        "countryOfBirth": [request.countryOfBirth],
        "nationality": [request.nationality],
        "receiver.address.countryCode": [request.receiver_address_countryCode],
        "age": [request.age],
        "id_verified": [request.id_verified],
        "cross_border": [request.cross_border]
    }
    
    df = pd.DataFrame(data)
    
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return PredictionResponse(
            payment_failure_risk=int(prediction),
            failure_probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
