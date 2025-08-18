"""
FastAPI application for Customer Churn Prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="1.0.0"
)

# Load configuration
with open("ml_pipeline/config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Global variables for model and preprocessors
model = None
feature_engineer = None

class CustomerData(BaseModel):
    """Input schema for customer data"""
    customer_id: int
    age: int
    gender: str
    tenure_months: int
    contract_type: str
    payment_method: str
    internet_service: str
    monthly_charges: float
    total_charges: float
    # Optional fields (will be set to 0 if not provided)
    total_transaction_amount: float = 0.0
    avg_transaction_amount: float = 0.0
    transaction_count: int = 0
    avg_resolution_time: float = 0.0
    support_ticket_count: int = 0
    avg_monthly_usage_gb: float = 0.0
    streaming_services_used: int = 0

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    customer_id: int
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load the trained model and preprocessors on startup"""
    global model, feature_engineer

    try:
        # Load the best trained model
        model_path = Path("ml_pipeline/models/best_model.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error("Model file not found. Please train a model first.")
            return

        # Load feature engineer (if saved separately)
        # For now, we'll recreate it
        from ml_pipeline.src.feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer(config)

        logger.info("API startup complete")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a single customer"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert customer data to DataFrame
        customer_df = pd.DataFrame([customer.dict()])

        # Apply feature engineering (without target variable)
        if feature_engineer:
            # Create a dummy target column for feature engineering
            customer_df['churn'] = 0
            X, _ = feature_engineer.engineer_features(customer_df, fit=False, select_features=True)
        else:
            # Simple preprocessing if feature engineer not available
            X = customer_df.drop(['customer_id'], axis=1)

        # Make prediction
        churn_proba = model.predict_proba(X)[0, 1]  # Probability of churn (class 1)
        churn_pred = churn_proba > 0.5

        # Determine risk level
        if churn_proba < 0.3:
            risk_level = "Low"
        elif churn_proba < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Calculate confidence (distance from decision boundary)
        confidence = abs(churn_proba - 0.5) * 2

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(churn_proba, 4),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(customers: List[CustomerData]):
    """Predict churn probability for multiple customers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for customer in customers:
            # Reuse single prediction logic
            result = await predict_churn(customer)
            predictions.append(result)

        return {
            "predictions": predictions,
            "total_customers": len(customers),
            "high_risk_count": sum(1 for p in predictions if p.risk_level == "High")
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        model_type = type(model).__name__

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            # This would need actual feature names from training
            feature_importance = "Available (use /feature_importance endpoint)"

        return {
            "model_type": model_type,
            "feature_importance": feature_importance,
            "model_parameters": str(model.get_params()) if hasattr(model, 'get_params') else "Not available"
        }

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Example usage data for testing
example_customer = {
    "customer_id": 12345,
    "age": 45,
    "gender": "Female",
    "tenure_months": 24,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "monthly_charges": 85.50,
    "total_charges": 2052.0,
    "total_transaction_amount": 2500.0,
    "avg_transaction_amount": 85.50,
    "transaction_count": 24,
    "avg_resolution_time": 0.0,
    "support_ticket_count": 0,
    "avg_monthly_usage_gb": 150.5,
    "streaming_services_used": 3
}

@app.get("/example")
async def get_example():
    """Get example customer data for testing"""
    return {
        "example_customer": example_customer,
        "usage": "POST this data to /predict endpoint"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
