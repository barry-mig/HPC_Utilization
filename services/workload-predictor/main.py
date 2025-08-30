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
    """
    Initialize and load the machine learning model for workload prediction.
    
    This function handles the complete model loading process including:
    - Accessing the global model state variables
    - Creating a new model instance through the synthetic model factory
    - Recording the model initialization timestamp for tracking purposes
    - Logging the successful loading event for monitoring and debugging
    
    In a production environment, this would connect to a model registry
    service, download the latest trained model artifacts, verify model
    integrity, and handle version compatibility checks.
    
    Returns:
        bool: True if model loading succeeded, False otherwise
    """
    # DEMO: Simplified model loading
    global model, model_last_retrain
    model = create_synthetic_model()
    model_last_retrain = datetime.now()
    logger.info(f"Demo model loaded. Version: {model_version}")
    return True

def create_synthetic_model():
    """
    Factory function that creates a synthetic machine learning model for demonstration.
    
    This function constructs a mock model object that simulates the behavior of a
    real time-series forecasting model. The synthetic model generates realistic
    predictions for HPC workload metrics including CPU utilization, memory usage,
    GPU consumption, and job queue lengths.
    
    The model uses statistical distributions to create believable predictions:
    - Normal distributions for resource utilization percentages
    - Poisson distribution for discrete job queue counts
    - Confidence intervals to represent prediction uncertainty
    
    In a real implementation, this would be replaced with actual trained models
    such as LSTM networks, ARIMA models, or ensemble methods that have been
    trained on historical HPC cluster data.
    
    Returns:
        SyntheticModel: An instance of the mock model class with predict method
    """
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
            """
            Generate mock predictions for workload forecasting based on input features.
            
            This method simulates the prediction process of a trained machine learning
            model by generating statistically realistic values for various HPC metrics.
            The predictions are based on normal and Poisson distributions that mimic
            real-world patterns observed in HPC environments.
            
            Args:
                X: Input feature matrix (length determines number of predictions)
                
            Returns:
                dict: Dictionary containing prediction arrays for different metrics:
                    - cpu_predictions: CPU utilization percentages (0-1)
                    - memory_predictions: Memory usage percentages (0-1) 
                    - gpu_predictions: GPU utilization percentages (0-1)
                    - queue_predictions: Number of queued jobs (integer)
                    - confidence_lower: Lower confidence interval bounds
                    - confidence_upper: Upper confidence interval bounds
            """
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
    """
    Service initialization handler that runs when the FastAPI application starts.
    
    This event handler is automatically triggered during the application startup
    process and handles critical initialization tasks. It attempts to load the
    machine learning model and logs the result for monitoring purposes.
    
    The function performs essential bootstrap operations:
    - Loads the prediction model into memory
    - Validates model loading success
    - Logs startup events for operational visibility
    - Prepares the service to handle incoming prediction requests
    
    If model loading fails, the service will log an error but continue running
    to allow health check endpoints to respond and indicate the service state.
    """
    logger.info("Starting HPC Workload Predictor service...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Kubernetes health check endpoint for monitoring service availability and readiness.
    
    This endpoint provides detailed health information that Kubernetes uses for:
    - Liveness probes: Determining if the service is running and responsive
    - Readiness probes: Checking if the service is ready to handle traffic
    - Load balancer decisions: Routing traffic only to healthy instances
    
    The health check reports multiple status indicators:
    - Overall service status (healthy/unhealthy)
    - Model loading state (critical for prediction functionality)
    - Last model retraining timestamp (for model freshness tracking)
    - Service uptime (for operational monitoring)
    
    Returns:
        HealthResponse: Structured health information including status and metrics
    """
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
    Main prediction endpoint that generates HPC workload forecasts based on historical data.
    
    This endpoint serves as the core functionality of the workload prediction service.
    It processes incoming requests containing historical workload metrics and returns
    detailed predictions for future time periods. The function handles the complete
    prediction pipeline including data validation, feature engineering, model inference,
    and response formatting.
    
    The prediction process involves several key steps:
    - Input validation and data structure conversion
    - Feature extraction from time-series historical data  
    - Model inference using the loaded machine learning model
    - Timestamp generation for future prediction periods
    - Response formatting with confidence intervals and metadata
    
    Metrics collection is integrated throughout the process to monitor:
    - Request counts for load tracking
    - Prediction latency for performance monitoring
    - Error rates for service health assessment
    
    Args:
        request (WorkloadPredictionRequest): Contains historical workload data,
                                           prediction horizon, and confidence level
    
    Returns:
        PredictionResponse: Structured predictions with timestamps, values,
                          confidence intervals, and metadata
    
    Raises:
        HTTPException: 503 if model is not loaded, 500 for prediction failures
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
    """
    Transform historical workload data into feature vectors for machine learning prediction.
    
    This function performs feature engineering on time-series workload data to create
    meaningful input features for the prediction model. The feature extraction process
    converts raw historical metrics into statistical summaries and patterns that
    capture important trends and behaviors in the HPC workload data.
    
    The feature engineering includes several categories:
    - Statistical measures: Mean, standard deviation, min/max values
    - Trend indicators: Rate of change, moving averages
    - Temporal patterns: Periodic behavior, seasonality indicators
    - Resource correlations: Relationships between CPU, memory, and GPU usage
    
    In a production environment, this would implement sophisticated time-series
    feature extraction techniques such as:
    - Fourier transforms for frequency domain analysis
    - Autocorrelation functions for pattern detection
    - Lag features for temporal dependencies
    - Rolling window statistics for trend analysis
    
    Args:
        df (pd.DataFrame): Historical workload data with columns for timestamps,
                          CPU usage, memory usage, GPU usage, job queues, etc.
    
    Returns:
        np.ndarray: Feature vector suitable for model input, containing extracted
                   statistical and temporal features from the historical data
    """
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
