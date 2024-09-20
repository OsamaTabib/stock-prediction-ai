import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# download or load cached stock data
def load_stock_data(ticker, period='2y', cache_dir='stock_cache'): # 2 years of data because any futher back will make the model more inaccurate due to changing market conditions and trends over time.
 
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{ticker}_{period}.csv')

    if os.path.exists(cache_file):
        try:
            print(f"Loading cached stock data for {ticker}...")
            stock_data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            if 'Close' not in stock_data.columns:
                raise ValueError("Invalid cached file format, downloading fresh data.")
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            print(f"Cache file is corrupted or invalid for {ticker}, downloading fresh data...")
            stock_data = download_and_cache_stock_data(ticker, period, cache_file)
    else:
        print(f"Downloading stock data for {ticker}...")
        stock_data = download_and_cache_stock_data(ticker, period, cache_file)

    return stock_data

def download_and_cache_stock_data(ticker, period, cache_file):
    stock_data = yf.download(ticker, period=period, interval='1h')
    if not stock_data.empty:
        stock_data.to_csv(cache_file)
    else:
        raise ValueError(f"No data found for {ticker}. It might be delisted or unavailable.")
    return stock_data

# preprocess data (Scaling and preparing for LSTM)
def preprocess_data(data):
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, scaler

# create datasets for LSTM (sequence-based)
def create_lstm_datasets(data, sequence_length=1000):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# build or load cached LSTM model
def build_or_load_lstm_model(input_shape, ticker, model_dir='model_cache'):
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f'{ticker}_lstm_model.keras')

    if os.path.exists(model_file):
        print(f"Loading cached model for {ticker}...")
        model = load_model(model_file)
    else:
        print(f"Building new model for {ticker}...")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

    return model, model_file

def predict_future_prices(model, last_sequence, scaler, steps=1):
    predicted_prices = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
        predicted_prices.append(prediction[0][0])
        current_sequence = np.append(current_sequence[1:], prediction)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

# Main function, load data, build the model, and predict
def stock_price_prediction(ticker):
    data = load_stock_data(ticker, period='2y')

    if data.shape[0] < 1000:
        raise ValueError("Not enough data points for a sequence length of 1000.")
    
    model_file = f'model_cache/{ticker}_lstm_model.keras'
    scaled_data, scaler = preprocess_data(data)

    sequence_length = 1000
    X, y = create_lstm_datasets(scaled_data, sequence_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model, model_file = build_or_load_lstm_model((X.shape[1], 1), ticker)

    if not os.path.exists(model_file):
        checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='loss', mode='min')
        model.fit(X, y, batch_size=32, epochs=10, callbacks=[checkpoint])

    last_sequence = scaled_data[-sequence_length:]
    time_steps = [1, 6, 12, 24, 168, 720, 8760]
    predictions = {f'Next {step} hours': predict_future_prices(model, last_sequence, scaler, steps=step)[-1][0] for step in time_steps}

    return predictions

# function to process multiple tickers in parallel using ProcessPoolExecutor
def process_tickers(tickers):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(stock_price_prediction, tickers))
    return results

# get stock ticker input from user and predict
tickers = input("Enter stock tickers separated by commas (e.g., AAPL,TSLA,AMZN): ").split(',')
tickers = [ticker.strip() for ticker in tickers]
predictions = process_tickers(tickers)

# display the predicted prices
for ticker, prediction in zip(tickers, predictions):
    print(f"Predictions for {ticker}:")
    for key, price in prediction.items():
        print(f"  {key}: ${price:.2f}")