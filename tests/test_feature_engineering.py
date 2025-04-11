"""
Unit tests for feature engineering utilities.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from utils.feature_engineering import (
    create_time_features,
    create_categorical_features,
    create_numeric_features,
    create_interaction_features,
    create_lag_features,
    create_window_features
)


@pytest.fixture
def sample_time_data():
    """Create sample time series data."""
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=5),
        'value': [1, 2, 3, 4, 5]
    })


@pytest.fixture
def sample_categorical_data():
    """Create sample categorical data."""
    return pd.DataFrame({
        'category1': ['A', 'B', 'A', 'C', 'B'],
        'category2': ['X', 'Y', 'X', 'Z', 'Y']
    })


@pytest.fixture
def sample_numeric_data():
    """Create sample numeric data."""
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50]
    })


def test_create_time_features(sample_time_data):
    """Test time-based feature creation."""
    result = create_time_features(sample_time_data, 'date')
    
    expected_columns = [
        'date_year',
        'date_month',
        'date_day',
        'date_dayofweek',
        'date_quarter',
        'date_is_weekend'
    ]
    
    assert all(col in result.columns for col in expected_columns)
    assert result['date_year'].dtype == np.int32
    assert result['date_is_weekend'].dtype == np.int32


def test_create_categorical_features(sample_categorical_data):
    """Test categorical feature creation."""
    # Test label encoding
    result_label, encoders = create_categorical_features(
        sample_categorical_data,
        ['category1'],
        encoding='label'
    )
    
    assert 'category1_encoded' in result_label.columns
    assert isinstance(encoders['category1'], LabelEncoder)
    
    # Test one-hot encoding
    result_onehot, mappings = create_categorical_features(
        sample_categorical_data,
        ['category1'],
        encoding='onehot'
    )
    
    expected_columns = [
        'category1_A',
        'category1_B',
        'category1_C'
    ]
    assert all(col in result_onehot.columns for col in expected_columns)


def test_create_numeric_features(sample_numeric_data):
    """Test numeric feature creation."""
    operations = ['log', 'sqrt', 'square']
    result = create_numeric_features(
        sample_numeric_data,
        ['numeric1'],
        operations=operations
    )
    
    expected_columns = [
        'numeric1_log',
        'numeric1_sqrt',
        'numeric1_squared'
    ]
    assert all(col in result.columns for col in expected_columns)
    
    # Test mathematical relationships
    np.testing.assert_array_almost_equal(
        result['numeric1_squared'],
        sample_numeric_data['numeric1'] ** 2
    )


def test_create_interaction_features(sample_numeric_data):
    """Test interaction feature creation."""
    feature_pairs = [('numeric1', 'numeric2')]
    result = create_interaction_features(sample_numeric_data, feature_pairs)
    
    expected_columns = [
        'numeric1_numeric2_product',
        'numeric1_numeric2_ratio'
    ]
    assert all(col in result.columns for col in expected_columns)
    
    # Test mathematical relationships
    np.testing.assert_array_almost_equal(
        result['numeric1_numeric2_product'],
        sample_numeric_data['numeric1'] * sample_numeric_data['numeric2']
    )


def test_create_lag_features():
    """Test lag feature creation."""
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4, 5]
    })
    
    result = create_lag_features(df, 'group', 'value', lags=[1, 2])
    
    expected_columns = ['value_lag_1', 'value_lag_2']
    assert all(col in result.columns for col in expected_columns)
    
    # Check lag values for group A
    assert result.loc[1, 'value_lag_1'] == 1
    assert result.loc[2, 'value_lag_1'] == 2


def test_create_window_features():
    """Test window feature creation."""
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4, 5]
    })
    
    result = create_window_features(
        df,
        'group',
        'value',
        windows=[2],
        functions=['mean']
    )
    
    expected_columns = ['value_mean_2']
    assert all(col in result.columns for col in expected_columns)
    
    # Check rolling mean values for group A
    assert result.loc[1, 'value_mean_2'] == 1.5  # mean of [1, 2]
    assert result.loc[2, 'value_mean_2'] == 2.5  # mean of [2, 3]


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    # Test non-existent column
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = create_numeric_features(df, ['nonexistent'])
    assert result.equals(df)  # Should return original DataFrame
    
    # Test invalid encoding method
    with pytest.raises(ValueError):
        create_categorical_features(
            df,
            ['A'],
            encoding='invalid'
        )
    
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    result = create_time_features(empty_df, 'date')
    assert result.empty