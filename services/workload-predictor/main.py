# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Dict
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import joblib
# import logging
# from prometheus_client import Counter, Histogram, generate_latest
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# DEMO PROJECT - Most code commented out for presentation
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import logging
from prometheus_client import Counter, Histogram, generate_latest
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
# PREDICTION_REQUESTS = Counter('workload_prediction_requests_total', 'Total prediction requests')
# PREDICTION_LATENCY = Histogram('workload_prediction_duration_seconds', 'Prediction request duration')

# app = FastAPI(
#     title="HPC Workload Predictor",
#     description="ML-based service for predicting HPC workload patterns and resource requirements",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# DEMO PROJECT - Simplified for presentation
PREDICTION_REQUESTS = Counter('workload_prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('workload_prediction_duration_seconds', 'Prediction request duration')

app = FastAPI(
    title="HPC Workload Predictor",
    description="ML-based service for predicting HPC workload patterns and resource requirements",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class WorkloadData(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    job_queue_length: int
    active_jobs: int
    cluster_size: int

class WorkloadPredictionRequest(BaseModel):
    historical_data: List[WorkloadData]
    prediction_horizon: int = 24  # hours
    confidence_level: float = 0.95

class WorkloadPrediction(BaseModel):
    timestamp: datetime
    predicted_cpu_usage: float
    predicted_memory_usage: float
    predicted_gpu_usage: Optional[float]
    predicted_job_queue_length: int
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_version: str

class PredictionResponse(BaseModel):
    predictions: List[WorkloadPrediction]
    model_accuracy: float
    generated_at: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_retrain: Optional[datetime]
    uptime_seconds: float

# Global variables for model state
model = None
model_version = "1.0.0"
model_last_retrain = None
start_time = datetime.now()

def load_model():
    """Load the pre-trained workload prediction model"""
    # global model, model_last_retrain
    # try:
    #     # In a real implementation, this would load from a model registry
    #     # For demo purposes, we'll create a simple synthetic model
    #     model = create_synthetic_model()
    #     model_last_retrain = datetime.now()
    #     logger.info(f"Model loaded successfully. Version: {model_version}")
    #     return True
    # except Exception as e:
    #     logger.error(f"Failed to load model: {e}")
    #     return False
    
    # DEMO: Simplified model loading
    global model, model_last_retrain
    model = create_synthetic_model()
    model_last_retrain = datetime.now()
    logger.info(f"Demo model loaded. Version: {model_version}")
    return True

def create_synthetic_model():
    """Create a synthetic model for demonstration purposes"""
    # # This would be replaced with actual trained models
    # class SyntheticModel:
    #     def predict(self, X):
    #         # Simple synthetic predictions based on historical patterns
    #         n_samples = len(X)
    #         return {
    #             'cpu_predictions': np.random.normal(0.7, 0.1, n_samples),
    #             'memory_predictions': np.random.normal(0.6, 0.15, n_samples),
    #             'gpu_predictions': np.random.normal(0.5, 0.2, n_samples),
    #             'queue_predictions': np.random.poisson(5, n_samples),
    #             'confidence_lower': np.random.normal(0.5, 0.1, n_samples),
    #             'confidence_upper': np.random.normal(0.9, 0.1, n_samples)
    #         }
    # return SyntheticModel()
    
    # DEMO: Simplified synthetic model
    class SyntheticModel:
        def predict(self, X):
            n_samples = len(X)
            return {
                'cpu_predictions': np.random.normal(0.7, 0.1, n_samples),
                'memory_predictions': np.random.normal(0.6, 0.15, n_samples),
                'gpu_predictions': np.random.normal(0.5, 0.2, n_samples),
                'queue_predictions': np.random.poisson(5, n_samples),
                'confidence_lower': np.random.normal(0.5, 0.1, n_samples),
                'confidence_upper': np.random.normal(0.9, 0.1, n_samples)
            }
    return SyntheticModel()

@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup"""
    logger.info("Starting HPC Workload Predictor service...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes readiness/liveness probes"""
    uptime = (datetime.now() - start_time).total_seconds()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        last_retrain=model_last_retrain,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_workload(request: WorkloadPredictionRequest):
    """
    Generate workload predictions based on historical data
    """
    PREDICTION_REQUESTS.inc()
    
    with PREDICTION_LATENCY.time():
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Convert request data to DataFrame for processing
            df = pd.DataFrame([data.dict() for data in request.historical_data])
            
            # Feature engineering (simplified for demo)
            features = extract_features(df)
            
            # Make predictions
            predictions_data = model.predict(features)
            
            # Generate prediction timestamps
            start_time = request.historical_data[-1].timestamp
            prediction_times = [
                start_time + timedelta(hours=i) 
                for i in range(1, request.prediction_horizon + 1)
            ]
            
            # Create prediction objects
            predictions = []
            for i, timestamp in enumerate(prediction_times):
                if i < len(predictions_data['cpu_predictions']):
                    prediction = WorkloadPrediction(
                        timestamp=timestamp,
                        predicted_cpu_usage=max(0, min(1, predictions_data['cpu_predictions'][i])),
                        predicted_memory_usage=max(0, min(1, predictions_data['memory_predictions'][i])),
                        predicted_gpu_usage=max(0, min(1, predictions_data['gpu_predictions'][i])),
                        predicted_job_queue_length=max(0, int(predictions_data['queue_predictions'][i])),
                        confidence_interval_lower=max(0, predictions_data['confidence_lower'][i]),
                        confidence_interval_upper=min(1, predictions_data['confidence_upper'][i]),
                        model_version=model_version
                    )
                    predictions.append(prediction)
            
            return PredictionResponse(
                predictions=predictions,
                model_accuracy=0.95,  # This would come from model evaluation
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract features from historical workload data"""
    # Simple feature extraction for demo
    # In practice, this would include sophisticated time series features
    features = []
    
    # Basic statistical features
    features.extend([
        df['cpu_usage'].mean(),
        df['cpu_usage'].std(),
        df['memory_usage'].mean(),
        df['memory_usage'].std(),
        df['job_queue_length'].mean(),
        df['active_jobs'].mean(),
    ])
    
    # Trend features
    if len(df) > 1:
        cpu_trend = (df['cpu_usage'].iloc[-1] - df['cpu_usage'].iloc[0]) / len(df)
        memory_trend = (df['memory_usage'].iloc[-1] - df['memory_usage'].iloc[0]) / len(df)
        features.extend([cpu_trend, memory_trend])
    else:
        features.extend([0, 0])
    
    # Time-based features
    if len(df) > 0:
        last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
        features.extend([
            last_timestamp.hour,
            last_timestamp.weekday(),
            last_timestamp.month
        ])
    else:
        features.extend([0, 0, 0])
    
    return np.array([features])

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)"""
    background_tasks.add_task(perform_model_retrain)
    return {"message": "Model retraining initiated"}

def perform_model_retrain():
    """Perform model retraining in the background"""
    global model, model_last_retrain
    try:
        logger.info("Starting model retraining...")
        # In practice, this would:
        # 1. Fetch new training data
        # 2. Retrain the model
        # 3. Validate performance
        # 4. Deploy new model if better
        
        # For demo, just reload the model
        model = create_synthetic_model()
        model_last_retrain = datetime.now()
        logger.info("Model retraining completed successfully")
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "last_retrain": model_last_retrain,
        "features_used": [
            "cpu_usage_mean", "cpu_usage_std",
            "memory_usage_mean", "memory_usage_std", 
            "job_queue_length_mean", "active_jobs_mean",
            "cpu_trend", "memory_trend",
            "hour_of_day", "day_of_week", "month"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
