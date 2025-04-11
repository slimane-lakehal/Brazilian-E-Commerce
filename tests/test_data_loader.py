"""
Unit tests for the data loader module.
"""
import pytest
import pandas as pd
from pathlib import Path

from src.data_processing.data_loader import OlistDataLoader


def test_data_loader_initialization(sample_project_paths):
    """Test data loader initialization."""
    loader = OlistDataLoader(sample_project_paths['raw'])
    assert isinstance(loader.data_dir, Path)
    assert all(dataset is None for dataset in loader.datasets.values())


def test_load_all_datasets(sample_project_paths, sample_orders_data):
    """Test loading all datasets."""
    loader = OlistDataLoader(sample_project_paths['raw'])
    datasets = loader.load_all_datasets()
    
    assert isinstance(datasets, dict)
    assert 'orders' in datasets
    assert 'customers' in datasets
    assert 'products' in datasets


def test_preprocess_orders(sample_orders_data):
    """Test order preprocessing."""
    loader = OlistDataLoader(Path('.'))
    loader.datasets['orders'] = sample_orders_data
    
    processed_orders = loader.preprocess_orders()
    
    # Check datetime conversions
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    for col in date_columns:
        assert pd.api.types.is_datetime64_any_dtype(processed_orders[col])


def test_create_customer_features(
    sample_orders_data,
    sample_customers_data,
    sample_reviews_data
):
    """Test customer feature creation."""
    loader = OlistDataLoader(Path('.'))
    loader.datasets['orders'] = sample_orders_data
    loader.datasets['customers'] = sample_customers_data
    loader.datasets['reviews'] = sample_reviews_data
    
    features = loader.create_customer_features()
    
    # Check required columns
    required_columns = [
        'customer_id',
        'order_count',
        'total_spend',
        'avg_order_value',
        'days_since_last_purchase'
    ]
    
    assert all(col in features.columns for col in required_columns)
    assert not features.empty


def test_get_preprocessed_data(
    sample_orders_data,
    sample_products_data,
    sample_customers_data,
    sample_reviews_data
):
    """Test getting all preprocessed data."""
    loader = OlistDataLoader(Path('.'))
    
    # Set sample data
    loader.datasets['orders'] = sample_orders_data
    loader.datasets['products'] = sample_products_data
    loader.datasets['customers'] = sample_customers_data
    loader.datasets['reviews'] = sample_reviews_data
    
    processed_data = loader.get_preprocessed_data()
    
    assert isinstance(processed_data, dict)
    assert 'orders' in processed_data
    assert 'customer_features' in processed_data
    assert 'products' in processed_data
    assert 'reviews' in processed_data


def test_data_loader_with_invalid_path():
    """Test data loader with invalid path."""
    with pytest.raises(Exception):
        loader = OlistDataLoader(Path('nonexistent_path'))
        loader.load_all_datasets()


def test_customer_features_with_missing_data():
    """Test customer feature creation with missing data."""
    loader = OlistDataLoader(Path('.'))
    
    # Test with missing datasets
    with pytest.raises(ValueError):
        loader.create_customer_features()
    
    # Test with empty datasets
    loader.datasets['orders'] = pd.DataFrame()
    loader.datasets['customers'] = pd.DataFrame()
    loader.datasets['reviews'] = pd.DataFrame()
    
    with pytest.raises(ValueError):
        loader.create_customer_features()