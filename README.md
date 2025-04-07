# Brazilian E-Commerce Data Analysis

This project analyzes the Brazilian e-commerce public dataset of orders made at Olist Store. The dataset contains information on 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil.

## Dataset Overview

The dataset includes information on:
- Orders: status, price, payment, freight, delivery time, etc.
- Products: category, measurements, descriptions, etc.
- Customers: location, etc.
- Sellers: location, etc.
- Order Reviews: score, comments, etc.

## Project Structure

- `brazilian_ecommerce_analysis.ipynb`: Jupyter notebook containing the complete analysis
- `data/`: Directory containing the dataset CSV files
- `requirements.txt`: List of Python packages required for this project

## Key Analysis Areas

1. **Sales Analysis**: Trends, total sales, average order value
2. **Product Category Analysis**: Popular categories, pricing by category
3. **Customer Analysis**: Geographic distribution, purchasing patterns
4. **Delivery Performance**: Delivery times, on-time delivery rates
5. **Customer Satisfaction**: Review scores and their correlation with other metrics
6. **Seller Analysis**: Geographic distribution, performance metrics
7. **Payment Analysis**: Payment methods, installment patterns
8. **Geographical Analysis**: Impact of location on sales and shipping

## Setup and Execution

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Open and run the Jupyter notebook:
```
jupyter notebook brazilian_ecommerce_analysis.ipynb
```

## Data Source

The dataset was provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. The dataset is available on Kaggle. 