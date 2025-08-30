"""
Main pipeline orchestrator for Customer Churn Prediction
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from model_training import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_data_pipeline():
    """Run the data ingestion and processing pipeline"""
    logger.info("Starting data pipeline...")

    # Data ingestion
    ingestion = DataIngestion()
    raw_data = ingestion.load_data()
    merged_data = ingestion.merge_data(raw_data)
    ingestion.save_processed_data(merged_data)

    logger.info("Data pipeline completed successfully")
    return merged_data

def run_training_pipeline(data=None):
    """Run the model training pipeline"""
    logger.info("Starting training pipeline...")

    # Load config
    with open("ml_pipeline/config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Load data if not provided
    if data is None:
        ingestion = DataIngestion()
        processed_data_path = Path("ml_pipeline/data/processed/processed_data.csv")
        if processed_data_path.exists():
            import pandas as pd
            data = pd.read_csv(processed_data_path)
        else:
            logger.info("No processed data found, running data pipeline first...")
            data = run_data_pipeline()

    # Feature engineering
    engineer = FeatureEngineer(config)
    X, y = engineer.engineer_features(data)

    # Model training
    predictor = ChurnPredictor()
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X, y)

    # Train and compare models
    scores = predictor.train_models(X_train, y_train, X_val, y_val)
    logger.info(f"Model comparison: {scores}")

    # Hyperparameter tuning
    predictor.hyperparameter_tuning(X_train, y_train)

    # Final evaluation
    results = predictor.evaluate_model(X_test, y_test)
    predictor.generate_report(results, X_test, y_test)

    logger.info(f"Training pipeline completed. Final test AUC: {results['auc_score']:.4f}")
    return results

def run_full_pipeline():
    """Run the complete pipeline from data to trained model"""
    logger.info("Starting full ML pipeline...")

    # Run data pipeline
    data = run_data_pipeline()

    # Run training pipeline
    results = run_training_pipeline(data)

    logger.info("Full pipeline completed successfully!")
    return results

def start_api():
    """Start the prediction API server"""
    logger.info("Starting API server...")

    import uvicorn
    import sys
    sys.path.append("ml_pipeline/api")

    # Check if model exists
    model_path = Path("ml_pipeline/models/best_model.pkl")
    if not model_path.exists():
        logger.warning("No trained model found. Running training pipeline first...")
        run_full_pipeline()

    # Start API
    uvicorn.run("ml_pipeline.api.main:app", host="0.0.0.0", port=8000, reload=True)

def create_sample_prediction():
    """Create a sample prediction for testing"""
    import pandas as pd
    import joblib

    # Load model
    model_path = Path("ml_pipeline/models/best_model.pkl")
    if not model_path.exists():
        logger.error("No trained model found. Run training first.")
        return

    model = joblib.load(model_path)

    # Create sample customer data
    sample_customer = pd.DataFrame({
        'customer_id': [99999],
        'age': [35],
        'gender_encoded': [1],  # This would need proper encoding
        'tenure_months': [12],
        'contract_type_encoded': [0],
        'monthly_charges': [75.0],
        'total_charges': [900.0],
        # Add other required features...
    })

    # This is simplified - in practice you'd use the feature engineering pipeline
    logger.info("Sample prediction functionality - implement with proper feature engineering")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("command", choices=[
        "data", "train", "full", "api", "predict"
    ], help="Command to run")

    args = parser.parse_args()

    try:
        if args.command == "data":
            run_data_pipeline()
        elif args.command == "train":
            run_training_pipeline()
        elif args.command == "full":
            run_full_pipeline()
        elif args.command == "api":
            start_api()
        elif args.command == "predict":
            create_sample_prediction()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # If no command line args, run full pipeline
    if len(sys.argv) == 1:
        run_full_pipeline()
    else:
        main()
