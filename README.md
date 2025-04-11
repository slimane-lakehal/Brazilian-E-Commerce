# Brazilian E-Commerce Data Analysis

## Project Overview
This project provides comprehensive analysis of the Brazilian E-commerce public dataset by Olist Store, available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The analysis includes customer behavior, sales patterns, product performance, and market trends.

## Features
- Customer Churn Prediction
- Sales Forecasting
- Product Recommendation System
- Customer Segmentation
- Geospatial Analysis
- Sentiment Analysis of Reviews
- A/B Testing Framework

## Project Structure
```
brazilian-ecommerce/
├── data/
│   ├── processed/      # Cleaned and processed data
│   └── raw/           # Original data files
├── notebooks/
│   ├── 1_exploratory_data_analysis/
│   ├── 2_customer_analysis/
│   ├── 3_product_analysis/
│   ├── 4_sales_analysis/
│   ├── 5_sentiment_analysis/
│   └── 6_experiments/
├── src/
│   ├── data_processing/  # Data processing scripts
│   ├── models/          # Model training and evaluation
│   └── visualization/   # Visualization utilities
├── tests/              # Unit tests
└── utils/              # Utility functions
```

## Requirements
- Python 3.12+
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- seaborn
- pmdarima
- scipy

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brazilian-ecommerce.git
cd brazilian-ecommerce
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Data Preparation:
   - Place the Olist dataset files in `data/raw/`
   - Run data processing scripts:
   ```bash
   python src/data_processing/prepare_data.py
   ```

2. Analysis:
   - Navigate to the `notebooks/` directory
   - Run Jupyter notebooks in numerical order:
   ```bash
   jupyter lab
   ```

## Key Findings
1. Customer Behavior:
   - Average customer lifetime value
   - Purchase frequency patterns
   - Churn prediction accuracy

2. Sales Patterns:
   - Seasonal trends
   - Geographic distribution
   - Category performance

3. Product Performance:
   - Best-selling categories
   - Price sensitivity analysis
   - Review sentiment correlation

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Olist for providing the dataset
- Kaggle community for insights and inspiration
- Contributors and maintainers of the used libraries