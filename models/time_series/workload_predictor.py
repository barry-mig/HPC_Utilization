# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# import logging
# from datetime import datetime, timedelta
# from typing import List, Dict, Tuple, Optional
# import warnings
# warnings.filterwarnings('ignore')

"""
DEMO PROJECT: HPC Workload Time Series Predictor
Advanced ML models for HPC workload forecasting
Most implementation commented out for presentation purposes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPredictor:
    """
    Advanced time series predictor for HPC workload forecasting
    Supports multiple algorithms and automatic model selection
    """
    
    def __init__(self, model_type: str = 'auto'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_metrics = {}
        
        # Available models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features from raw data"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Lag features
        for col in ['cpu_usage', 'memory_usage', 'job_queue_length', 'active_jobs']:
            if col in df.columns:
                for lag in [1, 2, 3, 6, 12, 24]:  # hours
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling statistics
        for col in ['cpu_usage', 'memory_usage', 'job_queue_length']:
            if col in df.columns:
                for window in [3, 6, 12, 24]:  # hours
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        
        # Trend features
        for col in ['cpu_usage', 'memory_usage']:
            if col in df.columns:
                for window in [6, 12, 24]:
                    df[f'{col}_trend_{window}'] = df[col].rolling(window=window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
                    )
        
        # Interaction features
        if 'cpu_usage' in df.columns and 'memory_usage' in df.columns:
            df['cpu_memory_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-8)
            df['cpu_memory_product'] = df['cpu_usage'] * df['memory_usage']
        
        if 'job_queue_length' in df.columns and 'active_jobs' in df.columns:
            df['queue_active_ratio'] = df['job_queue_length'] / (df['active_jobs'] + 1)
            df['total_jobs'] = df['job_queue_length'] + df['active_jobs']
        
        # Remove rows with NaN values (due to lag and rolling features)
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_cols: List[str], 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        
        # Create features
        df_features = self.create_features(df)
        
        # Define feature columns (exclude timestamp and targets)
        exclude_cols = ['timestamp'] + target_cols
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        X = df_features[feature_cols].values
        y = df_features[target_cols].values
        
        # Handle multi-target case
        if len(target_cols) == 1:
            y = y.ravel()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, df: pd.DataFrame, target_cols: List[str] = None) -> Dict:
        """Train the time series model"""
        
        if target_cols is None:
            target_cols = ['cpu_usage', 'memory_usage', 'job_queue_length']
        
        self.logger.info(f"Training model with targets: {target_cols}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_cols)
        
        if self.model_type == 'auto':
            # Automatic model selection
            best_model, best_score = self._select_best_model(X_train, y_train)
            self.model = best_model
            self.model_type = best_score['model_name']
        else:
            self.model = self.models[self.model_type]
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        self.model_metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'model_type': self.model_type,
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'trained_at': datetime.now().isoformat()
        }
        
        self.is_trained = True
        
        self.logger.info(f"Model trained successfully. Test R²: {test_metrics['r2']:.4f}")
        
        return self.model_metrics
    
    def _select_best_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """Automatically select the best model using cross-validation"""
        
        best_score = float('-inf')
        best_model = None
        best_name = None
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, scoring='r2', n_jobs=-1
                )
                avg_score = cv_scores.mean()
                
                self.logger.info(f"{name}: CV R² = {avg_score:.4f} (±{cv_scores.std():.4f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
                    
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {name}: {e}")
        
        return best_model, {'model_name': best_name, 'cv_score': best_score}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        
        # Handle multi-target case
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        metrics = {}
        
        for i in range(y_true.shape[1]):
            target_name = f'target_{i}'
            metrics[target_name] = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i])
            }
        
        # Overall metrics
        metrics['overall'] = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred) if y_true.shape[1] > 1 else metrics['target_0']['r2']
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Make predictions for future time points"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features for the input data
        df_features = self.create_features(df)
        
        # Use the last available data point as starting point
        last_row = df_features.iloc[-1:].copy()
        predictions = []
        
        # Iterative prediction for the specified horizon
        for step in range(horizon):
            # Prepare features
            X = last_row[self.feature_names].values
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            pred = self.model.predict(X_scaled)
            
            # Store prediction
            pred_dict = {
                'step': step + 1,
                'timestamp': last_row['timestamp'].iloc[0] + timedelta(hours=step + 1)
            }
            
            # Handle multi-target predictions
            if isinstance(pred, np.ndarray) and len(pred.shape) > 1:
                pred_dict.update({
                    'cpu_usage': max(0, min(1, pred[0][0])),
                    'memory_usage': max(0, min(1, pred[0][1])),
                    'job_queue_length': max(0, int(pred[0][2]))
                })
            else:
                pred_dict['prediction'] = pred[0]
            
            predictions.append(pred_dict)
            
            # Update last_row for next iteration (simplified approach)
            # In a real implementation, you'd update lag features properly
            if 'cpu_usage' in pred_dict:
                last_row['cpu_usage'] = pred_dict['cpu_usage']
                last_row['memory_usage'] = pred_dict['memory_usage']
                last_row['job_queue_length'] = pred_dict['job_queue_length']
        
        return pd.DataFrame(predictions)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance if supported by the model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.model_metrics,
            'version': '1.0.0'
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.model_metrics = model_data['metrics']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic HPC workload data for testing"""
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=n_samples)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate realistic workload patterns
    np.random.seed(42)
    
    data = []
    for i, ts in enumerate(timestamps):
        # Daily and weekly patterns
        hour_factor = 0.5 + 0.5 * np.sin(2 * np.pi * ts.hour / 24)
        day_factor = 0.8 + 0.2 * np.sin(2 * np.pi * ts.weekday() / 7)
        
        # Base patterns with noise
        cpu_usage = np.clip(
            0.3 + 0.4 * hour_factor * day_factor + 0.1 * np.random.randn(),
            0, 1
        )
        
        memory_usage = np.clip(
            0.4 + 0.3 * hour_factor * day_factor + 0.1 * np.random.randn(),
            0, 1
        )
        
        # Job patterns
        base_jobs = 20 * hour_factor * day_factor
        job_queue_length = max(0, int(base_jobs + 5 * np.random.randn()))
        active_jobs = max(0, int(base_jobs * 0.7 + 3 * np.random.randn()))
        
        cluster_size = 100  # Fixed cluster size
        gpu_usage = np.clip(
            0.2 + 0.3 * hour_factor + 0.15 * np.random.randn(),
            0, 1
        ) if np.random.random() > 0.3 else 0  # Not all jobs use GPU
        
        data.append({
            'timestamp': ts,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'job_queue_length': job_queue_length,
            'active_jobs': active_jobs,
            'cluster_size': cluster_size
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(2000)
    
    # Initialize and train model
    print("Training time series predictor...")
    predictor = TimeSeriesPredictor(model_type='auto')
    
    # Train the model
    metrics = predictor.train(df, target_cols=['cpu_usage', 'memory_usage', 'job_queue_length'])
    
    print(f"\nTraining completed!")
    print(f"Model type: {metrics['model_type']}")
    print(f"Test R²: {metrics['test']['overall']['r2']:.4f}")
    
    # Feature importance
    importance = predictor.feature_importance()
    if not importance.empty:
        print(f"\nTop 10 most important features:")
        print(importance.head(10))
    
    # Make predictions
    print(f"\nMaking 24-hour predictions...")
    recent_data = df.tail(100)  # Use last 100 hours for context
    predictions = predictor.predict(recent_data, horizon=24)
    print(predictions.head())
    
    # Save model
    predictor.save_model('workload_predictor_model.pkl')
    print(f"\nModel saved successfully!")
