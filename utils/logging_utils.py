"""
Logging utilities for the Brazilian E-commerce analysis project.
"""
import logging
from pathlib import Path
from typing import Optional

from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Parameters
    ----------
    name : str
        Name of the logger
    level : Optional[str]
        Logging level (default from settings)
    log_format : Optional[str]
        Log message format (default from settings)
    log_file : Optional[Path]
        Path to log file (default from settings)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters and handlers
    formatter = logging.Formatter(log_format or LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file or LOG_FILE:
        log_path = log_file or LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Decorator to log function calls with parameters.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    func_name : str
        Name of the function being logged
    **kwargs : dict
        Function parameters to log
    """
    logger.debug(
        f"Calling {func_name} with parameters: "
        f"{', '.join(f'{k}={v}' for k, v in kwargs.items())}"
    )


def log_dataframe_info(logger: logging.Logger, df_name: str, df):
    """
    Log information about a DataFrame.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    df_name : str
        Name of the DataFrame
    df : pd.DataFrame
        DataFrame to log information about
    """
    logger.info(f"\nDataFrame: {df_name}")
    logger.info(f"Shape: {df.shape}")
    logger.info("\nColumns:")
    for col in df.columns:
        logger.info(f"  - {col}: {df[col].dtype}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")


def log_model_metrics(logger: logging.Logger, model_name: str, metrics: dict):
    """
    Log model evaluation metrics.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    model_name : str
        Name of the model
    metrics : dict
        Dictionary of metric names and values
    """
    logger.info(f"\nModel: {model_name}")
    logger.info("Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  - {metric}: {value:.4f}")


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[str] = None
):
    """
    Log an error with context.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    error : Exception
        The error to log
    context : Optional[str]
        Additional context about where/why the error occurred
    """
    error_message = f"Error: {str(error)}"
    if context:
        error_message = f"{context}\n{error_message}"
    
    logger.error(error_message, exc_info=True)