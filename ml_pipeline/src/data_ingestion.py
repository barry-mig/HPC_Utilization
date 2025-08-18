"""
Data Ingestion Module for Customer Churn Prediction
Handles data collection from multiple sources and initial preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config_path: str = "ml_pipeline/config/config.yaml"):
        """Initialize data ingestion with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.data_sources = self.config['data_sources']
        self.raw_data_path = Path("ml_pipeline/data/raw")
        self.processed_data_path = Path("ml_pipeline/data/processed")

        # Create directories if they don't exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_data(self) -> None:
        """Generate synthetic customer data for demonstration"""
        logger.info("Generating synthetic customer data...")

        np.random.seed(42)
        n_customers = 10000

        # Customer demographics
        demographics = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'tenure_months': np.random.exponential(24, n_customers).astype(int),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                            n_customers, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                             'Bank transfer', 'Credit card'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'],
                                               n_customers, p=[0.4, 0.4, 0.2])
        })

        # Generate charges based on service type and tenure
        base_charges = np.where(demographics['internet_service'] == 'Fiber optic', 80,
                               np.where(demographics['internet_service'] == 'DSL', 50, 20))
        demographics['monthly_charges'] = base_charges + np.random.normal(0, 10, n_customers)
        demographics['total_charges'] = (demographics['monthly_charges'] *
                                       demographics['tenure_months'] +
                                       np.random.normal(0, 100, n_customers))

        # Generate churn target (higher probability for month-to-month, lower tenure)
        churn_prob = (0.3 * (demographics['contract_type'] == 'Month-to-month') +
                     0.1 * (demographics['contract_type'] == 'One year') +
                     0.05 * (demographics['contract_type'] == 'Two year') +
                     0.1 * (demographics['tenure_months'] < 12) +
                     0.1 * (demographics['monthly_charges'] > 80))

        demographics['churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_customers)

        demographics.to_csv(self.raw_data_path / "customer_demographics.csv", index=False)

        # Transaction history (sample)
        transactions = []
        for customer_id in range(1, min(1001, n_customers + 1)):  # Sample for first 1000 customers
            n_transactions = np.random.poisson(12)  # ~12 transactions per year
            for _ in range(n_transactions):
                transactions.append({
                    'customer_id': customer_id,
                    'transaction_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                    'amount': np.random.lognormal(3, 0.5),
                    'transaction_type': np.random.choice(['Payment', 'Refund', 'Adjustment'], p=[0.8, 0.1, 0.1])
                })

        pd.DataFrame(transactions).to_csv(self.raw_data_path / "transactions.csv", index=False)

        # Support tickets
        support_tickets = []
        for customer_id in range(1, min(2001, n_customers + 1)):
            if np.random.random() < 0.3:  # 30% of customers have support tickets
                n_tickets = np.random.poisson(2)
                for _ in range(n_tickets):
                    support_tickets.append({
                        'customer_id': customer_id,
                        'ticket_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                        'issue_type': np.random.choice(['Technical', 'Billing', 'Service'], p=[0.5, 0.3, 0.2]),
                        'resolution_time_hours': np.random.exponential(24)
                    })

        pd.DataFrame(support_tickets).to_csv(self.raw_data_path / "support_tickets.csv", index=False)

        # Product usage
        usage_data = []
        for customer_id in range(1, min(5001, n_customers + 1)):
            if demographics.loc[demographics['customer_id'] == customer_id, 'internet_service'].iloc[0] != 'No':
                usage_data.append({
                    'customer_id': customer_id,
                    'avg_monthly_usage_gb': np.random.lognormal(4, 1),
                    'peak_usage_hours': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night']),
                    'streaming_services_used': np.random.poisson(2)
                })

        pd.DataFrame(usage_data).to_csv(self.raw_data_path / "product_usage.csv", index=False)

        logger.info(f"Generated synthetic data for {n_customers} customers")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources"""
        data = {}

        # Check if data exists, if not generate synthetic data
        if not (self.raw_data_path / "customer_demographics.csv").exists():
            self.generate_synthetic_data()

        for source_name, file_path in self.data_sources.items():
            try:
                data[source_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {source_name}: {data[source_name].shape}")
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}")
                continue

        return data

    def merge_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data sources on customer_id"""
        if 'customer_demographics' not in data:
            raise ValueError("Customer demographics data is required")

        merged_df = data['customer_demographics'].copy()

        # Merge transaction aggregations
        if 'transaction_history' in data:
            trans_agg = data['transaction_history'].groupby('customer_id').agg({
                'amount': ['sum', 'mean', 'count'],
                'transaction_date': 'max'
            }).reset_index()
            trans_agg.columns = ['customer_id', 'total_transaction_amount',
                               'avg_transaction_amount', 'transaction_count', 'last_transaction_date']
            merged_df = merged_df.merge(trans_agg, on='customer_id', how='left')

        # Merge support ticket aggregations
        if 'support_tickets' in data:
            support_agg = data['support_tickets'].groupby('customer_id').agg({
                'resolution_time_hours': ['mean', 'count']
            }).reset_index()
            support_agg.columns = ['customer_id', 'avg_resolution_time', 'support_ticket_count']
            merged_df = merged_df.merge(support_agg, on='customer_id', how='left')

        # Merge product usage
        if 'product_usage' in data:
            merged_df = merged_df.merge(data['product_usage'], on='customer_id', how='left')

        # Fill missing values
        merged_df = merged_df.fillna(0)

        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv") -> None:
        """Save processed data"""
        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    raw_data = ingestion.load_data()
    merged_data = ingestion.merge_data(raw_data)
    ingestion.save_processed_data(merged_data)
    print(f"Processed {len(merged_data)} customer records")
    print(f"Churn rate: {merged_data['churn'].mean():.2%}")
