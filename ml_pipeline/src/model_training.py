"""
Model Training Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self, config_path: str = "ml_pipeline/config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.models = {}
        self.best_model = None
        self.model_scores = {}

        # Create models directory
        self.models_path = Path("ml_pipeline/models")
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Create reports directory
        self.reports_path = Path("ml_pipeline/reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with configuration"""
        models = {}

        # Random Forest
        rf_config = self.config['models']['random_forest']
        models['random_forest'] = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state'],
            n_jobs=-1
        )

        # XGBoost
        xgb_config = self.config['models']['xgboost']
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            random_state=xgb_config['random_state'],
            eval_metric='logloss'
        )

        # Logistic Regression
        models['logistic_regression'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        self.models = models
        return models

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train, validation, and test sets"""
        config = self.config['training']

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y
        )

        # Second split: train vs validation
        val_size = config['validation_size'] / (1 - config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=config['random_state'],
            stratify=y_temp
        )

        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train all models and return validation scores"""
        if not self.models:
            self.initialize_models()

        scores = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Validate
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            auc_score = roc_auc_score(y_val, val_pred_proba)
            scores[name] = auc_score

            logger.info(f"{name} - Validation AUC: {auc_score:.4f}")

            # Save model
            model_path = self.models_path / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")

        self.model_scores = scores
        return scores

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Perform hyperparameter tuning for the best performing model"""
        if not self.model_scores:
            raise ValueError("Train models first to identify best performer")

        # Get best model name
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        logger.info(f"Tuning hyperparameters for {best_model_name}")

        if best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)

        elif best_model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

        else:  # logistic_regression
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000)

        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config['training']['cross_validation_folds'],
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Save best model
        self.best_model = grid_search.best_estimator_
        best_model_path = self.models_path / "best_model.pkl"
        joblib.dump(self.best_model, best_model_path)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"Saved best model to {best_model_path}")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the best model on test set"""
        if self.best_model is None:
            # Load best model if not in memory
            best_model_path = self.models_path / "best_model.pkl"
            if best_model_path.exists():
                self.best_model = joblib.load(best_model_path)
            else:
                raise ValueError("No best model found. Train and tune model first.")

        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'auc_score': auc_score,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }

        logger.info(f"Test AUC Score: {auc_score:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return results

    def generate_report(self, results: Dict[str, Any], X_test: pd.DataFrame,
                       y_test: pd.Series) -> None:
        """Generate comprehensive model report"""
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            sns.barplot(data=feature_importance, y='feature', x='importance', ax=axes[0, 1])
            axes[0, 1].set_title('Top 10 Feature Importances')

        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, results['prediction_probabilities'])
        axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc_score"]:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()

        # Prediction Distribution
        axes[1, 1].hist(results['prediction_probabilities'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Prediction Probabilities')

        plt.tight_layout()

        # Save report
        report_path = self.reports_path / "model_evaluation_report.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics to file
        metrics_path = self.reports_path / "model_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"Model Evaluation Report\n")
            f.write(f"=====================\n\n")
            f.write(f"Test AUC Score: {results['auc_score']:.4f}\n\n")
            f.write(f"Classification Report:\n")
            f.write(classification_report(y_test, results['predictions']))
            f.write(f"\n\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']))

        logger.info(f"Saved evaluation report to {report_path}")
        logger.info(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from feature_engineering import FeatureEngineer
    import yaml

    # Load config
    with open("ml_pipeline/config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Load and process data
    ingestion = DataIngestion()
    raw_data = ingestion.load_data()
    merged_data = ingestion.merge_data(raw_data)

    # Engineer features
    engineer = FeatureEngineer(config)
    X, y = engineer.engineer_features(merged_data)

    # Train models
    predictor = ChurnPredictor()
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X, y)

    # Train and compare models
    scores = predictor.train_models(X_train, y_train, X_val, y_val)
    print("Model comparison:", scores)

    # Hyperparameter tuning
    predictor.hyperparameter_tuning(X_train, y_train)

    # Final evaluation
    results = predictor.evaluate_model(X_test, y_test)
    predictor.generate_report(results, X_test, y_test)

    print(f"Final test AUC: {results['auc_score']:.4f}")
