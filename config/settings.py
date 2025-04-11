"""
Configuration settings for the Brazilian E-commerce analysis project.
"""
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
MODEL_ARTIFACTS_DIR = 'models/artifacts'

# Data processing settings
DATE_COLUMNS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

NUMERIC_COLUMNS = [
    'payment_value',
    'price',
    'freight_value'
]

CATEGORICAL_COLUMNS = [
    'product_category_name',
    'payment_type',
    'customer_city',
    'customer_state'
]

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering settings
FEATURE_GROUPS = {
    'customer': [
        'customer_id',
        'customer_unique_id',
        'customer_zip_code_prefix',
        'customer_city',
        'customer_state'
    ],
    'order': [
        'order_id',
        'customer_id',
        'order_status',
        'order_purchase_timestamp'
    ],
    'product': [
        'product_id',
        'product_category_name',
        'product_name_length',
        'product_description_length',
        'product_photos_qty',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]
}

DATE_FEATURES = [
    'year',
    'month',
    'day',
    'day_of_week',
    'day_of_year',
    'week_of_year',
    'is_weekend',
]

# Time series analysis settings
TIME_WINDOWS = [7, 14, 30, 90]  # Days
FORECAST_HORIZONS = [7, 30, 90]  # Days
FORECAST_HORIZON = 30  # days
SEASONAL_PERIODS = {
    'weekly': 7,
    'monthly': 30,
    'yearly': 365,
}

# Customer segmentation settings
RFM_QUANTILES = 4  # Number of quantiles for RFM analysis
CUSTOMER_SEGMENTS = {
    'best': {
        'recency': (4, 5),
        'frequency': (4, 5),
        'monetary': (4, 5)
    },
    'loyal': {
        'recency': (3, 5),
        'frequency': (3, 5),
        'monetary': (3, 5)
    },
    'lost': {
        'recency': (1, 2),
        'frequency': (1, 2),
        'monetary': (1, 3)
    }
}

# Model training settings
DEFAULT_CV_FOLDS = 5

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8
    }
}

# Visualization settings
PLOT_STYLE = 'seaborn'
COLORS = {
    'primary': '#1f77b4',    # Blue
    'secondary': '#ff7f0e',  # Orange
    'positive': '#2ca02c',   # Green
    'negative': '#d62728',   # Red
    'neutral': '#7f7f7f',    # Gray
}

FIGURE_SIZES = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 10),
}

# A/B testing settings
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_LEVEL = 0.95
STATISTICAL_POWER = 0.8
MINIMUM_EFFECT_SIZE = 0.1
MINIMUM_SAMPLE_SIZE = 100

# Text analysis settings
NLTK_PACKAGES = [
    'punkt',
    'stopwords',
    'wordnet'
]

TEXT_PREPROCESSING = {
    'min_word_length': 3,
    'max_features': 10000,
    'min_df': 5,
    'max_df': 0.95
}

STOPWORDS_LANGUAGE = 'portuguese'

# Logging settings
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
LOG_FILE = PROJECT_ROOT / 'logs' / 'analysis.log'