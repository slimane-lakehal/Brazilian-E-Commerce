"""
Data loading and preprocessing module for the Brazilian E-commerce dataset.
"""
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from pandas import DataFrame


class OlistDataLoader:
    """Data loader for the Olist E-commerce dataset."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, Optional[DataFrame]] = {
            'customers': None,
            'orders': None,
            'order_items': None,
            'products': None,
            'payments': None,
            'reviews': None,
            'sellers': None,
            'geolocation': None,
            'category_translation': None
        }

    def load_all_datasets(self) -> Dict[str, DataFrame]:
        """Load all available datasets."""
        dataset_files = {
            'customers': 'olist_customers_dataset.csv',
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'payments': 'olist_order_payments_dataset.csv',
            'reviews': 'olist_order_reviews_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }

        for key, filename in dataset_files.items():
            file_path = self.data_dir / filename
            self.datasets[key] = pd.read_csv(file_path)

        return self.datasets

    def preprocess_orders(self) -> DataFrame:
        """Preprocess orders dataset with datetime conversions."""
        if self.datasets['orders'] is None:
            raise ValueError("Orders dataset not loaded")

        date_columns = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]

        orders = self.datasets['orders'].copy()
        for col in date_columns:
            orders[col] = pd.to_datetime(orders[col])

        return orders

    def create_customer_features(self, analysis_date: Optional[pd.Timestamp] = None) -> DataFrame:
        """Create customer features for analysis."""
        if any(self.datasets[key] is None for key in ['orders', 'payments', 'reviews']):
            raise ValueError("Required datasets not loaded")

        orders = self.preprocess_orders()
        
        if analysis_date is None:
            analysis_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=30)

        # Customer order history
        customer_orders = orders.groupby('customer_id').agg({
            'order_id': 'count',
            'order_purchase_timestamp': [
                'min',
                'max',
                lambda x: (analysis_date - x.max()).days
            ]
        }).reset_index()

        customer_orders.columns = [
            'customer_id', 'order_count', 'first_purchase_date',
            'last_purchase_date', 'days_since_last_purchase'
        ]

        # Add monetary value features
        order_values = self.datasets['payments'].groupby('order_id')['payment_value'].sum()
        customer_values = (orders.merge(order_values.reset_index(), on='order_id')
                         .groupby('customer_id')
                         .agg({
                             'payment_value': ['sum', 'mean', 'std', 'min', 'max']
                         }).reset_index())

        customer_values.columns = [
            'customer_id', 'total_spend', 'avg_order_value',
            'std_order_value', 'min_order_value', 'max_order_value'
        ]

        # Merge features
        customer_features = customer_orders.merge(
            customer_values, on='customer_id', how='left'
        )

        return customer_features

    def get_preprocessed_data(self) -> Dict[str, DataFrame]:
        """Get all preprocessed datasets ready for analysis."""
        if not all(self.datasets.values()):
            self.load_all_datasets()

        processed_data = {
            'orders': self.preprocess_orders(),
            'customer_features': self.create_customer_features(),
            'products': self.datasets['products'],
            'reviews': self.datasets['reviews'],
            'sellers': self.datasets['sellers'],
        }

        return processed_data