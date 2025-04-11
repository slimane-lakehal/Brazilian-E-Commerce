"""
Model evaluation utilities for the Brazilian E-commerce analysis.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, str]]:
    """
    Evaluate a classification model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : Optional[List[str]]
        Names of the classes
        
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary containing evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=3
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a regression model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def evaluate_time_series(
    y_true: pd.Series,
    y_pred: pd.Series,
    freq: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a time series forecast.
    
    Parameters
    ----------
    y_true : pd.Series
        True values with datetime index
    y_pred : pd.Series
        Predicted values with datetime index
    freq : Optional[str]
        Frequency for resampling (e.g., 'D' for daily, 'W' for weekly)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    if freq:
        y_true = y_true.resample(freq).mean()
        y_pred = y_pred.resample(freq).mean()
    
    # Align the series
    y_true, y_pred = y_true.align(y_pred, join='inner')
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics


def evaluate_recommendation(
    true_items: List[str],
    recommended_items: List[str],
    k: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate a recommendation system.
    
    Parameters
    ----------
    true_items : List[str]
        List of actual items
    recommended_items : List[str]
        List of recommended items
    k : Optional[int]
        Number of recommendations to consider
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    if k is not None:
        recommended_items = recommended_items[:k]
    
    # Calculate precision
    precision = len(set(true_items) & set(recommended_items)) / len(recommended_items)
    
    # Calculate recall
    recall = len(set(true_items) & set(recommended_items)) / len(true_items)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics