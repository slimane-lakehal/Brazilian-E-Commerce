"""
Unit tests for the visualization utilities.
"""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.visualization.plot_utils import (
    plot_time_series,
    plot_category_distribution,
    plot_correlation_matrix
)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2022-01-01', end='2022-01-10', freq='D')
    values = np.random.randn(len(dates))
    return pd.DataFrame({
        'date': dates,
        'value': values
    })


@pytest.fixture
def sample_category_data():
    """Create sample categorical data for testing."""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [10, 8, 6, 4, 2]
    return pd.DataFrame({
        'category': categories * 2,
        'value': values * 2
    })


@pytest.fixture
def sample_correlation_data():
    """Create sample numeric data for correlation testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })


def test_plot_time_series(sample_time_series_data):
    """Test time series plotting function."""
    fig = plot_time_series(
        data=sample_time_series_data,
        date_column='date',
        value_column='value',
        title='Test Time Series'
    )
    
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == 'Test Time Series'
    assert fig.axes[0].get_xlabel() == 'Date'
    assert fig.axes[0].get_ylabel() == 'Value'


def test_plot_category_distribution(sample_category_data):
    """Test category distribution plotting function."""
    fig = plot_category_distribution(
        data=sample_category_data,
        category_column='category',
        value_column='value',
        title='Test Categories'
    )
    
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == 'Test Categories'


def test_plot_correlation_matrix(sample_correlation_data):
    """Test correlation matrix plotting function."""
    fig = plot_correlation_matrix(
        data=sample_correlation_data,
        title='Test Correlation'
    )
    
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == 'Test Correlation'


def test_plot_time_series_with_forecast(sample_time_series_data):
    """Test time series plotting with forecast data."""
    # Create forecast data
    forecast_dates = pd.date_range(
        start='2022-01-11',
        end='2022-01-15',
        freq='D'
    )
    forecast_values = pd.Series(
        np.random.randn(len(forecast_dates)),
        index=forecast_dates
    )
    
    fig = plot_time_series(
        data=sample_time_series_data,
        date_column='date',
        value_column='value',
        title='Test Time Series with Forecast',
        forecast=forecast_values
    )
    
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2  # One line for data, one for forecast