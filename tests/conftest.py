"""
Shared test fixtures for the Brazilian E-commerce analysis project.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def sample_orders_data():
    """Create sample orders data for testing."""
    return pd.DataFrame({
        'order_id': ['1', '2', '3', '4', '5'],
        'customer_id': ['A', 'B', 'A', 'C', 'B'],
        'order_purchase_timestamp': pd.date_range('2022-01-01', periods=5),
        'order_approved_at': pd.date_range('2022-01-01', periods=5),
        'order_delivered_carrier_date': pd.date_range('2022-01-02', periods=5),
        'order_delivered_customer_date': pd.date_range('2022-01-03', periods=5),
        'order_estimated_delivery_date': pd.date_range('2022-01-05', periods=5)
    })


@pytest.fixture(scope="session")
def sample_products_data():
    """Create sample products data for testing."""
    return pd.DataFrame({
        'product_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'product_category_name': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2'],
        'price': [100.0, 200.0, 150.0, 300.0, 250.0],
        'freight_value': [10.0, 20.0, 15.0, 30.0, 25.0]
    })


@pytest.fixture(scope="session")
def sample_customers_data():
    """Create sample customers data for testing."""
    return pd.DataFrame({
        'customer_id': ['A', 'B', 'C', 'D', 'E'],
        'customer_city': ['City1', 'City2', 'City1', 'City3', 'City2'],
        'customer_state': ['State1', 'State2', 'State1', 'State3', 'State2']
    })


@pytest.fixture(scope="session")
def sample_reviews_data():
    """Create sample reviews data for testing."""
    return pd.DataFrame({
        'review_id': ['r1', 'r2', 'r3', 'r4', 'r5'],
        'order_id': ['1', '2', '3', '4', '5'],
        'review_score': [5, 4, 3, 4, 5],
        'review_comment_title': ['Great', 'Good', 'OK', 'Good', 'Excellent'],
        'review_comment_message': ['Very satisfied', 'Nice product', 
                                 'Average', 'Would buy again', 'Perfect!']
    })


@pytest.fixture(scope="session")
def sample_time_series_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Generate seasonal data with trend
    t = np.arange(len(dates))
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
    noise = np.random.normal(0, 2, len(dates))
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })


@pytest.fixture(scope="session")
def sample_project_paths():
    """Create sample project paths for testing."""
    return {
        'root': Path(__file__).parent.parent,
        'data': Path(__file__).parent.parent / 'data',
        'raw': Path(__file__).parent.parent / 'data' / 'raw',
        'processed': Path(__file__).parent.parent / 'data' / 'processed',
        'notebooks': Path(__file__).parent.parent / 'notebooks',
        'src': Path(__file__).parent.parent / 'src'
    }