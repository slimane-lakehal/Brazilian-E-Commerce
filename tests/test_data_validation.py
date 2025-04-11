"""
Unit tests for data validation utilities.
"""
import pandas as pd
import pytest
import numpy as np

from utils.data_validation import (
    validate_date_columns,
    validate_numeric_columns,
    clean_string_columns,
    handle_outliers,
    validate_dataframe_schema
)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'date_col': pd.date_range('2022-01-01', periods=5),
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['A', 'B', 'C', 'D', 'E'],
        'mixed_col': ['A', 'B', '1', '2', np.nan]
    })


def test_validate_date_columns(sample_dataframe):
    """Test date column validation."""
    # Valid case
    assert validate_date_columns(sample_dataframe, ['date_col'])
    
    # Invalid column
    assert not validate_date_columns(sample_dataframe, ['nonexistent'])
    
    # Non-date column
    assert not validate_date_columns(sample_dataframe, ['numeric_col'])


def test_validate_numeric_columns(sample_dataframe):
    """Test numeric column validation."""
    # Valid case
    assert validate_numeric_columns(sample_dataframe, ['numeric_col'])
    
    # Invalid column
    assert not validate_numeric_columns(sample_dataframe, ['nonexistent'])
    
    # Non-numeric column
    assert not validate_numeric_columns(sample_dataframe, ['string_col'])
    
    # Test negative values
    df = sample_dataframe.copy()
    df['numeric_col'] = [-1, -2, 3, 4, 5]
    assert not validate_numeric_columns(df, ['numeric_col'], allow_negative=False)
    assert validate_numeric_columns(df, ['numeric_col'], allow_negative=True)


def test_clean_string_columns(sample_dataframe):
    """Test string column cleaning."""
    df = sample_dataframe.copy()
    df['string_col'] = ['A!', 'B@', 'C#', 'D$', 'E%']
    
    # Test with default parameters
    cleaned_df = clean_string_columns(df, ['string_col'])
    assert all(cleaned_df['string_col'] == ['a', 'b', 'c', 'd', 'e'])
    
    # Test without lowercase conversion
    cleaned_df = clean_string_columns(df, ['string_col'], lower=False)
    assert all(cleaned_df['string_col'] == ['A', 'B', 'C', 'D', 'E'])
    
    # Test with NA filling
    cleaned_df = clean_string_columns(df, ['mixed_col'], fill_na='missing')
    assert cleaned_df['mixed_col'].iloc[-1] == 'missing'


def test_handle_outliers():
    """Test outlier handling."""
    df = pd.DataFrame({
        'values': [1, 2, 3, 100, 200, 3, 4, 5]
    })
    
    # Test IQR method with clipping
    cleaned_df = handle_outliers(
        df,
        ['values'],
        method='iqr',
        strategy='clip'
    )
    assert cleaned_df['values'].max() < 200
    
    # Test z-score method with removal
    cleaned_df = handle_outliers(
        df,
        ['values'],
        method='zscore',
        strategy='remove'
    )
    assert len(cleaned_df) < len(df)


def test_validate_dataframe_schema(sample_dataframe):
    """Test DataFrame schema validation."""
    schema = {
        'numeric_col': {
            'type': 'numeric',
            'required': True,
            'min_value': 0
        },
        'string_col': {
            'type': 'string',
            'required': True,
            'allowed_values': ['A', 'B', 'C', 'D', 'E']
        },
        'date_col': {
            'type': 'datetime',
            'required': True
        }
    }
    
    # Valid case
    assert validate_dataframe_schema(sample_dataframe, schema)
    
    # Test with invalid data
    df = sample_dataframe.copy()
    df['numeric_col'] = [-1, -2, -3, -4, -5]  # Negative values
    assert not validate_dataframe_schema(df, schema)
    
    # Test with missing required column
    df = sample_dataframe.drop('numeric_col', axis=1)
    assert not validate_dataframe_schema(df, schema)
    
    # Test with invalid categorical values
    df = sample_dataframe.copy()
    df['string_col'] = ['X', 'Y', 'Z', 'W', 'V']
    assert not validate_dataframe_schema(df, schema)