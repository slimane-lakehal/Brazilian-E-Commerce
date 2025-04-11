"""
Unit tests for the model evaluation utilities.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.model_evaluation import (
    evaluate_classifier,
    evaluate_regression,
    evaluate_time_series,
    evaluate_recommendation
)


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    class_names = ['Class_0', 'Class_1']
    return y_true, y_pred, class_names


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    return y_true, y_pred


@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    dates = pd.date_range(start='2022-01-01', end='2022-01-10', freq='D')
    y_true = pd.Series(np.random.randn(len(dates)) + 10, index=dates)
    y_pred = pd.Series(np.random.randn(len(dates)) + 10, index=dates)
    return y_true, y_pred


@pytest.fixture
def recommendation_data():
    """Create sample recommendation data."""
    true_items = ['item1', 'item2', 'item3']
    recommended_items = ['item1', 'item3', 'item4', 'item5']
    return true_items, recommended_items


def test_evaluate_classifier(classification_data):
    """Test classification evaluation metrics."""
    y_true, y_pred, class_names = classification_data
    metrics = evaluate_classifier(y_true, y_pred, class_names)
    
    assert 'accuracy' in metrics
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics
    assert isinstance(metrics['accuracy'], float)
    assert 0 <= metrics['accuracy'] <= 1


def test_evaluate_regression(regression_data):
    """Test regression evaluation metrics."""
    y_true, y_pred = regression_data
    metrics = evaluate_regression(y_true, y_pred)
    
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics['rmse'] >= 0


def test_evaluate_time_series(time_series_data):
    """Test time series evaluation metrics."""
    y_true, y_pred = time_series_data
    metrics = evaluate_time_series(y_true, y_pred)
    
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics['rmse'] >= 0


def test_evaluate_time_series_with_resampling(time_series_data):
    """Test time series evaluation with resampling."""
    y_true, y_pred = time_series_data
    metrics = evaluate_time_series(y_true, y_pred, freq='2D')
    
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_recommendation(recommendation_data):
    """Test recommendation system evaluation metrics."""
    true_items, recommended_items = recommendation_data
    metrics = evaluate_recommendation(true_items, recommended_items)
    
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())


def test_evaluate_recommendation_with_k(recommendation_data):
    """Test recommendation system evaluation with k limit."""
    true_items, recommended_items = recommendation_data
    metrics = evaluate_recommendation(true_items, recommended_items, k=2)
    
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())