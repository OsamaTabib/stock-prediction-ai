# Stock Price Prediction using LSTM

This repository provides a script for predicting future stock prices using an LSTM (Long Short-Term Memory) neural network model. The script utilizes historical stock price data, scales it for optimal LSTM training, builds or loads a cached LSTM model, and predicts future prices. It is designed to handle multiple stock tickers and can process predictions in parallel.

## Features

- **Stock Data Download & Caching**: Downloads up to 2 years of stock data from Yahoo Finance (`yfinance`) and caches it locally for quicker access.
- **Data Preprocessing**: Uses `MinMaxScaler` to normalize stock prices for training.
- **LSTM Model**: Builds an LSTM model using `TensorFlow/Keras`, or loads a pre-trained model from cache.
- **Price Prediction**: Predicts future stock prices over different time intervals (e.g., 1 hour, 6 hours, 24 hours, etc.).
- **Parallel Processing**: Handles multiple stock tickers in parallel using `ProcessPoolExecutor` for efficient computation.

## Requirements

Ensure you have the following dependencies installed:

- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

To install the required packages, run:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
