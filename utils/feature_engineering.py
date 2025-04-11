"""
Feature engineering utilities for the Brazilian E-commerce analysis project.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import holidays

from config.settings import FEATURE_GROUPS
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)
br_holidays = holidays.BR()  # Brazilian holidays


def create_time_features(
    df: DataFrame,
    date_column: str,
    drop_original: bool = False
) -> DataFrame:
    """
    Create comprehensive time-based features from a date column.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    date_column : str
        Name of the date column
    drop_original : bool
        Whether to drop the original date column
        
    Returns
    -------
    DataFrame
        DataFrame with new time features
    """
    df = df.copy()
    
    # Basic datetime components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Enhanced seasonality features
    df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
    df[f'{date_column}_weekofyear'] = df[date_column].dt.isocalendar().week
    df[f'{date_column}_days_in_month'] = df[date_column].dt.days_in_month
    
    # Cyclical features (converting cyclical variables to continuous using sine and cosine)
    df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df[date_column].dt.day / 31)
    df[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df[date_column].dt.day / 31)
    df[f'{date_column}_dayofweek_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
    df[f'{date_column}_dayofweek_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
    
    # Holiday features
    df[f'{date_column}_is_holiday'] = df[date_column].map(lambda x: x in br_holidays).astype(int)
    df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    # Special events (Black Friday, Christmas season, etc.)
    df[f'{date_column}_is_christmas_season'] = (
        ((df[date_column].dt.month == 12) & (df[date_column].dt.day >= 1) & (df[date_column].dt.day <= 25))
    ).astype(int)
    
    df[f'{date_column}_is_black_friday_season'] = (
        ((df[date_column].dt.month == 11) & (df[date_column].dt.day >= 20) & (df[date_column].dt.day <= 30))
    ).astype(int)
    
    if drop_original:
        df = df.drop(columns=[date_column])
    
    return df


def create_categorical_features(
    df: DataFrame,
    categorical_columns: List[str],
    encoding: str = 'label',
    handle_unknown: str = 'ignore'
) -> tuple[DataFrame, Dict[str, Union[LabelEncoder, Dict[str, int]]]]:
    """
    Create features from categorical columns.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    categorical_columns : List[str]
        List of categorical column names
    encoding : str
        Encoding method ('label' or 'onehot')
    handle_unknown : str
        How to handle unknown categories ('ignore' or 'error')
        
    Returns
    -------
    tuple[DataFrame, Dict]
        Transformed DataFrame and dictionary of encoders/mappings
    """
    df = df.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        if encoding == 'label':
            encoder = LabelEncoder()
            df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            
        elif encoding == 'onehot':
            dummies = pd.get_dummies(
                df[col],
                prefix=col,
                dummy_na=handle_unknown == 'ignore'
            )
            df = pd.concat([df, dummies], axis=1)
            encoders[col] = {
                category: 1 for category in dummies.columns
            }
            
        else:
            raise ValueError(f"Unknown encoding method: {encoding}")
    
    return df, encoders


def create_numeric_features(
    df: DataFrame,
    numeric_columns: List[str],
    operations: Optional[List[str]] = None
) -> DataFrame:
    """
    Create features from numeric columns.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    numeric_columns : List[str]
        List of numeric column names
    operations : Optional[List[str]]
        List of operations to perform ('log', 'sqrt', 'square', 'standardize')
        
    Returns
    -------
    DataFrame
        DataFrame with new numeric features
    """
    df = df.copy()
    operations = operations or ['log', 'sqrt']
    
    for col in numeric_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric")
            continue
            
        if 'log' in operations and (df[col] > 0).all():
            df[f'{col}_log'] = np.log1p(df[col])
            
        if 'sqrt' in operations and (df[col] >= 0).all():
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            
        if 'square' in operations:
            df[f'{col}_squared'] = df[col] ** 2
            
        if 'standardize' in operations:
            scaler = StandardScaler()
            df[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
    
    return df


def create_interaction_features(
    df: DataFrame,
    feature_pairs: List[tuple[str, str]]
) -> DataFrame:
    """
    Create interaction features between pairs of numeric columns.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    feature_pairs : List[tuple[str, str]]
        List of column name pairs to create interactions for
        
    Returns
    -------
    DataFrame
        DataFrame with new interaction features
    """
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Columns {col1} and/or {col2} not found in DataFrame")
            continue
            
        if not (pd.api.types.is_numeric_dtype(df[col1]) and 
                pd.api.types.is_numeric_dtype(df[col2])):
            logger.warning(f"Columns {col1} and/or {col2} are not numeric")
            continue
            
        # Product interaction
        df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
        
        # Ratio interaction (avoiding division by zero)
        if (df[col2] != 0).all():
            df[f'{col1}_{col2}_ratio'] = df[col1] / df[col2]
    
    return df


def create_lag_features(
    df: DataFrame,
    group_column: str,
    target_column: str,
    lags: List[int]
) -> DataFrame:
    """
    Create lagged features for time series data.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    group_column : str
        Column to group by (e.g., customer_id)
    target_column : str
        Column to create lags for
    lags : List[int]
        List of lag periods
        
    Returns
    -------
    DataFrame
        DataFrame with new lag features
    """
    df = df.copy().sort_index()
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df.groupby(group_column)[target_column].shift(lag)
    
    return df


def create_window_features(
    df: DataFrame,
    group_column: str,
    target_column: str,
    windows: List[int],
    functions: List[str] = ['mean', 'std', 'min', 'max']
) -> DataFrame:
    """
    Create rolling window features for time series data.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    group_column : str
        Column to group by (e.g., customer_id)
    target_column : str
        Column to create windows for
    windows : List[int]
        List of window sizes
    functions : List[str]
        List of aggregation functions to apply
        
    Returns
    -------
    DataFrame
        DataFrame with new window features
    """
    df = df.copy().sort_index()
    
    for window in windows:
        for func in functions:
            df[f'{target_column}_{func}_{window}'] = (
                df.groupby(group_column)[target_column]
                .rolling(window=window, min_periods=1)
                .agg(func)
                .reset_index(level=0, drop=True)
            )
    
    return df


def create_advanced_window_features(
    df: DataFrame,
    group_column: str,
    target_column: str,
    windows: List[int],
    functions: List[str] = ['mean', 'std', 'min', 'max', 'sum', 'skew']
) -> DataFrame:
    """
    Create advanced rolling window features with trend indicators.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    group_column : str
        Column to group by (e.g., customer_id)
    target_column : str
        Column to create windows for
    windows : List[int]
        List of window sizes
    functions : List[str]
        List of aggregation functions to apply
        
    Returns
    -------
    DataFrame
        DataFrame with new window features
    """
    df = df.copy().sort_index()
    
    for window in windows:
        # Basic window statistics
        for func in functions:
            df[f'{target_column}_{func}_{window}'] = (
                df.groupby(group_column)[target_column]
                .rolling(window=window, min_periods=1)
                .agg(func)
                .reset_index(level=0, drop=True)
            )
        
        # Trend indicators
        rolling = df.groupby(group_column)[target_column].rolling(window=window, min_periods=1)
        
        # Momentum (rate of change)
        df[f'{target_column}_momentum_{window}'] = (
            df[target_column] - rolling.mean().reset_index(level=0, drop=True)
        )
        
        # Acceleration (change in momentum)
        df[f'{target_column}_acceleration_{window}'] = df[f'{target_column}_momentum_{window}'].diff()
        
        # Relative strength (current value compared to moving average)
        df[f'{target_column}_relative_strength_{window}'] = (
            df[target_column] / rolling.mean().reset_index(level=0, drop=True)
        )
        
        # Volatility (rolling standard deviation normalized by mean)
        df[f'{target_column}_volatility_{window}'] = (
            rolling.std().reset_index(level=0, drop=True) / 
            rolling.mean().reset_index(level=0, drop=True)
        )
    
    return df