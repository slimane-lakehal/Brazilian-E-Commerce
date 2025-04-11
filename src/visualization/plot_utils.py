"""
Visualization utilities for the Brazilian E-commerce analysis project.
"""
from typing import Optional, Union, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame, Series

from config.settings import COLORS, FIGURE_SIZES, PLOT_STYLE


def set_plotting_style(style: str = PLOT_STYLE):
    """Set consistent plotting style."""
    plt.style.use(style)
    sns.set_palette(list(COLORS.values()))


def plot_time_series(
    data: DataFrame,
    date_column: str,
    value_column: str,
    title: str = '',
    rolling_window: Optional[int] = None,
    figsize: tuple = FIGURE_SIZES['medium']
) -> plt.Figure:
    """Plot time series data with optional rolling average.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    date_column : str
        Name of date column
    value_column : str
        Name of value column
    title : str
        Plot title
    rolling_window : Optional[int]
        Window size for rolling average
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    ax.plot(
        data[date_column],
        data[value_column],
        alpha=0.5,
        color=COLORS['primary'],
        label='Raw Data'
    )
    
    # Add rolling average if specified
    if rolling_window:
        rolling_avg = data[value_column].rolling(
            window=rolling_window,
            min_periods=1
        ).mean()
        
        ax.plot(
            data[date_column],
            rolling_avg,
            color=COLORS['secondary'],
            label=f'{rolling_window}-period Moving Average'
        )
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_distribution(
    data: Union[Series, np.ndarray],
    title: str = '',
    kde: bool = True,
    figsize: tuple = FIGURE_SIZES['small']
) -> plt.Figure:
    """Plot distribution of data.
    
    Parameters
    ----------
    data : Union[Series, np.ndarray]
        Data to plot
    title : str
        Plot title
    kde : bool
        Whether to include KDE plot
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(
        data=data,
        kde=kde,
        color=COLORS['primary'],
        ax=ax
    )
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_correlation_matrix(
    data: DataFrame,
    figsize: tuple = FIGURE_SIZES['medium']
) -> plt.Figure:
    """Plot correlation matrix heatmap.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    corr_matrix = data.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        ax=ax
    )
    
    ax.set_title('Correlation Matrix')
    
    return fig


def plot_category_comparison(
    data: DataFrame,
    category_column: str,
    value_column: str,
    title: str = '',
    top_n: Optional[int] = None,
    sort_values: bool = True,
    figsize: tuple = FIGURE_SIZES['medium']
) -> plt.Figure:
    """Plot comparison across categories.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    category_column : str
        Name of category column
    value_column : str
        Name of value column
    title : str
        Plot title
    top_n : Optional[int]
        Number of top categories to show
    sort_values : bool
        Whether to sort by values
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Prepare data
    plot_data = data.groupby(category_column)[value_column].mean()
    
    if sort_values:
        plot_data = plot_data.sort_values(ascending=False)
    
    if top_n:
        plot_data = plot_data.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_data.plot(
        kind='bar',
        color=COLORS['primary'],
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel(category_column)
    ax.set_ylabel(f'Average {value_column}')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def create_interactive_map(
    data: DataFrame,
    lat_column: str,
    lon_column: str,
    hover_data: List[str],
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    title: str = ''
) -> go.Figure:
    """Create interactive map visualization.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    lat_column : str
        Name of latitude column
    lon_column : str
        Name of longitude column
    hover_data : List[str]
        Columns to show in hover tooltip
    color_column : Optional[str]
        Column to use for point colors
    size_column : Optional[str]
        Column to use for point sizes
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = px.scatter_mapbox(
        data,
        lat=lat_column,
        lon=lon_column,
        hover_data=hover_data,
        color=color_column,
        size=size_column,
        title=title,
        mapbox_style='carto-positron'
    )
    
    fig.update_layout(
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
        mapbox=dict(zoom=4)
    )
    
    return fig


def plot_time_of_day_analysis(
    data: DataFrame,
    datetime_column: str,
    value_column: str,
    figsize: tuple = FIGURE_SIZES['medium']
) -> plt.Figure:
    """Plot time of day analysis.
    
    Parameters
    ----------
    data : DataFrame
        Input data
    datetime_column : str
        Name of datetime column
    value_column : str
        Name of value column
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract hour from datetime
    hours = pd.to_datetime(data[datetime_column]).dt.hour
    
    # Calculate average values by hour
    hourly_avg = data.groupby(hours)[value_column].mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        hourly_avg.index,
        hourly_avg.values,
        marker='o',
        color=COLORS['primary']
    )
    
    ax.set_title('Time of Day Analysis')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'Average {value_column}')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks for each hour
    ax.set_xticks(range(24))
    
    return fig


def plot_radar_chart(
    categories: List[str],
    values: List[float],
    title: str = '',
    figsize: tuple = FIGURE_SIZES['small']
) -> plt.Figure:
    """Create a radar chart.
    
    Parameters
    ----------
    categories : List[str]
        Category names
    values : List[float]
        Values for each category
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Plot data
    values += values[:1]
    ax.plot(angles, values, color=COLORS['primary'])
    ax.fill(angles, values, color=COLORS['primary'], alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    ax.set_title(title)
    
    return fig


def plot_funnel(
    stages: List[str],
    values: List[float],
    percentage_drop: bool = True,
    figsize: tuple = FIGURE_SIZES['medium']
) -> plt.Figure:
    """Create a funnel chart.
    
    Parameters
    ----------
    stages : List[str]
        Names of funnel stages
    values : List[float]
        Values for each stage
    percentage_drop : bool
        Whether to show percentage drop between stages
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Calculate stage widths
    width = np.array(values) / max(values) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    y_positions = range(len(stages))
    ax.barh(y_positions, width, align='center', color=COLORS['primary'])
    
    # Add stage names and values
    for i, (stage, value) in enumerate(zip(stages, values)):
        # Stage name on the left
        ax.text(-5, i, stage, ha='right', va='center')
        
        # Value on the bar
        ax.text(
            width[i] + 2, i,
            f'{value:,.0f}',
            ha='left',
            va='center'
        )
        
        # Add percentage drop
        if percentage_drop and i > 0:
            pct_drop = (values[i-1] - value) / values[i-1] * 100
            ax.text(
                width[i] + 25, i,
                f'-{pct_drop:.1f}%',
                ha='left',
                va='center',
                color=COLORS['negative']
            )
    
    # Customize the chart
    ax.set_xlim(-40, 120)
    ax.set_ylim(-0.5, len(stages) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig