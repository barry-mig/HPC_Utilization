"""
Feature Engineering Module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.numerical_features = config['features']['numerical']
        self.categorical_features = config['features']['categorical']
        self.derived_features = config['features']['derived']

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones"""
        df = df.copy()

        logger.info("Creating derived features...")

        # Charges per month ratio
        df['charges_per_month_ratio'] = df['total_charges'] / (df['tenure_months'] + 1)

        # Support tickets per month
        df['support_tickets_per_month'] = df.get('support_ticket_count', 0) / (df['tenure_months'] + 1)

        # Usage trend (high/medium/low usage)
        if 'avg_monthly_usage_gb' in df.columns:
            usage_quantiles = df['avg_monthly_usage_gb'].quantile([0.33, 0.67])
            df['usage_trend'] = pd.cut(df['avg_monthly_usage_gb'],
                                     bins=[-np.inf, usage_quantiles[0.33], usage_quantiles[0.67], np.inf],
                                     labels=['Low', 'Medium', 'High'])
            df['usage_trend'] = df['usage_trend'].astype(str)
        else:
            df['usage_trend'] = 'Unknown'

        # Customer lifetime value estimate
        df['estimated_clv'] = df['monthly_charges'] * df['tenure_months']

        # Contract risk score (higher for month-to-month)
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df['contract_risk_score'] = df['contract_type'].map(contract_risk)

        # Age group
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100],
                               labels=['Young', 'Middle', 'Senior', 'Elder'])
        df['age_group'] = df['age_group'].astype(str)

        # High value customer flag
        df['high_value_customer'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)

        logger.info(f"Created {len(self.derived_features)} derived features")
        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()

        logger.info("Encoding categorical features...")

        # Get all categorical columns (including derived ones)
        categorical_cols = [col for col in df.columns
                          if df[col].dtype == 'object' and col != 'churn']

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values

                    if unknown_values:
                        logger.warning(f"Unknown categories in {col}: {unknown_values}")
                        # Map unknown values to a default category
                        df[col] = df[col].astype(str).replace(list(unknown_values), 'Unknown')

                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))

        # Drop original categorical columns
        df = df.drop(columns=categorical_cols)

        return df

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()

        logger.info("Scaling numerical features...")

        # Get numerical columns (excluding target)
        numerical_cols = [col for col in df.columns
                         if df[col].dtype in ['int64', 'float64'] and col != 'churn']

        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """Select top k features using statistical tests"""
        logger.info(f"Selecting top {k} features...")

        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()

        logger.info(f"Selected features: {selected_features}")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def transform_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Apply feature selection to new data"""
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted. Call select_features first.")

        X_selected = self.feature_selector.transform(X)
        selected_feature_names = X.columns[self.feature_selector.get_support()].tolist()

        return pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)

    def engineer_features(self, df: pd.DataFrame, fit: bool = True,
                         select_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")

        # Separate features and target
        target_col = 'churn'
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df.copy()

        # Create derived features
        X = self.create_derived_features(X)

        # Encode categorical features
        X = self.encode_categorical_features(X, fit=fit)

        # Scale numerical features
        X = self.scale_numerical_features(X, fit=fit)

        # Feature selection (only during training)
        if select_features and fit and y is not None:
            X = self.select_features(X, y)
        elif select_features and not fit:
            X = self.transform_features(X)

        logger.info(f"Feature engineering complete. Final shape: {X.shape}")

        return X, y

if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
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

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    print(f"Feature columns: {list(X.columns)}")
