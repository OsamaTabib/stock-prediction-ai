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
```


## How to Use

### Download or Load Stock Data:
The function `load_stock_data(ticker, period)` downloads stock data or loads it from a local cache if available.

### Preprocess Data:
The function `preprocess_data(data)` scales the 'Close' prices to a range between 0 and 1 for training the LSTM model.

### Create LSTM Datasets:
The function `create_lstm_datasets(data, sequence_length)` generates sequences of stock price data to feed into the LSTM model for training.

### Build or Load LSTM Model:
The function `build_or_load_lstm_model(input_shape, ticker)` either builds a new LSTM model or loads a cached version if it exists.

### Predict Future Prices:
The function `predict_future_prices(model, last_sequence, scaler, steps)` predicts future prices based on the trained model.

### Process Multiple Tickers:
Use the function `process_tickers(tickers)` to predict prices for multiple stock tickers in parallel.

### Run the Script:
Run the script and input the stock tickers when prompted. The script will display predicted prices for each ticker across different time intervals.

```bash
python stock_prediction.py

