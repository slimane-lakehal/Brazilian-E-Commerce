"""
Data validation and cleaning utilities for the Brazilian E-commerce analysis project.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from config.settings import DATE_COLUMNS
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def validate_date_columns(df: DataFrame, date_columns: Optional[List[str]] = None) -> bool:
    """
    Validate date columns in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to validate
    date_columns : Optional[List[str]]
        List of column names to validate (default from settings)
        
    Returns
    -------
    bool
        True if all validations pass
    """
    date_columns = date_columns or DATE_COLUMNS
    
    for col in date_columns:
        if col not in df.columns:
            logger.error(f"Missing required date column: {col}")
            return False
        
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.error(f"Column {col} is not in datetime format")
            return False
        
        if df[col].isnull().any():
            logger.warning(f"Column {col} contains null values")
    
    return True


def validate_numeric_columns(
    df: DataFrame,
    numeric_columns: List[str],
    allow_negative: bool = False
) -> bool:
    """
    Validate numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to validate
    numeric_columns : List[str]
        List of column names to validate
    allow_negative : bool
        Whether to allow negative values
        
    Returns
    -------
    bool
        True if all validations pass
    """
    for col in numeric_columns:
        if col not in df.columns:
            logger.error(f"Missing required numeric column: {col}")
            return False
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Column {col} is not numeric")
            return False
        
        if not allow_negative and (df[col] < 0).any():
            logger.error(f"Column {col} contains negative values")
            return False
        
        if df[col].isnull().any():
            logger.warning(f"Column {col} contains null values")
    
    return True


def clean_string_columns(
    df: DataFrame,
    string_columns: List[str],
    lower: bool = True,
    remove_special_chars: bool = True,
    fill_na: Optional[str] = None
) -> DataFrame:
    """
    Clean string columns in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to clean
    string_columns : List[str]
        List of column names to clean
    lower : bool
        Convert to lowercase
    remove_special_chars : bool
        Remove special characters
    fill_na : Optional[str]
        Value to fill NaN values with
        
    Returns
    -------
    DataFrame
        Cleaned DataFrame
    """
    df = df.copy()
    
    for col in string_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        if not pd.api.types.is_string_dtype(df[col]):
            logger.warning(f"Column {col} is not string type")
            continue
        
        # Convert to lowercase if requested
        if lower:
            df[col] = df[col].str.lower()
        
        # Remove special characters if requested
        if remove_special_chars:
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        # Fill NA values if specified
        if fill_na is not None:
            df[col] = df[col].fillna(fill_na)
    
    return df


def handle_outliers(
    df: DataFrame,
    numeric_columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5,
    strategy: str = 'clip'
) -> DataFrame:
    """
    Handle outliers in numeric columns.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to process
    numeric_columns : List[str]
        List of column names to process
    method : str
        Method to detect outliers ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
    strategy : str
        Strategy to handle outliers ('clip' or 'remove')
        
    Returns
    -------
    DataFrame
        Processed DataFrame
    """
    df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric type")
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores > threshold
            if strategy == 'remove':
                df = df[~mask]
                continue
            lower_bound = df[col].mean() - threshold * df[col].std()
            upper_bound = df[col].mean() + threshold * df[col].std()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if strategy == 'clip':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif strategy == 'remove':
            df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def validate_dataframe_schema(
    df: DataFrame,
    schema: Dict[str, Dict[str, Union[str, bool, List[str]]]]
) -> bool:
    """
    Validate DataFrame against a schema.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to validate
    schema : Dict
        Schema definition with column names, types, and constraints
        
    Returns
    -------
    bool
        True if all validations pass
    
    Example
    -------
    schema = {
        'customer_id': {
            'type': 'string',
            'required': True,
            'unique': True
        },
        'order_amount': {
            'type': 'numeric',
            'required': True,
            'min_value': 0
        }
    }
    """
    for column, rules in schema.items():
        # Check if required column exists
        if rules.get('required', False) and column not in df.columns:
            logger.error(f"Required column {column} not found")
            return False
        
        if column not in df.columns:
            continue
        
        # Check column type
        col_type = rules.get('type', 'any')
        if col_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(df[column]):
                logger.error(f"Column {column} should be numeric")
                return False
        elif col_type == 'string':
            if not pd.api.types.is_string_dtype(df[column]):
                logger.error(f"Column {column} should be string")
                return False
        elif col_type == 'datetime':
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                logger.error(f"Column {column} should be datetime")
                return False
        
        # Check uniqueness
        if rules.get('unique', False) and not df[column].is_unique:
            logger.error(f"Column {column} should be unique")
            return False
        
        # Check numeric constraints
        if col_type == 'numeric':
            if 'min_value' in rules and (df[column] < rules['min_value']).any():
                logger.error(f"Column {column} contains values below {rules['min_value']}")
                return False
            if 'max_value' in rules and (df[column] > rules['max_value']).any():
                logger.error(f"Column {column} contains values above {rules['max_value']}")
                return False
        
        # Check categorical constraints
        if 'allowed_values' in rules:
            invalid_values = ~df[column].isin(rules['allowed_values'])
            if invalid_values.any():
                logger.error(f"Column {column} contains invalid values")
                return False
    
    return True