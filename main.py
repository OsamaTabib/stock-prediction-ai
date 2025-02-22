import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib  # For saving and loading the scaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from concurrent.futures import ProcessPoolExecutor

# ======================
# DATA PIPELINE
# ======================

def load_gold_data(interval='5m', cache_dir='gold_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    ticker = 'GC=F'
    cache_file = os.path.join(cache_dir, f'gold_{interval}_full.csv')
    
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Convert columns to numeric to avoid issues with string types
        data = data.apply(pd.to_numeric, errors='coerce')
        if not data.empty:
            return data
    
    print(f"Downloading fresh gold data ({interval} intervals)...")
    data = yf.download(ticker, period='60d', interval=interval)
    
    # Add technical indicators
    data = calculate_technical_indicators(data)
    
    # Remove rows with missing values
    data = data.dropna()
    
    data.to_csv(cache_file, index_label='Date')
    return data

def calculate_technical_indicators(data):
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    data.loc[:, 'MACD'] = exp12 - exp26
    data.loc[:, 'Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

# ======================
# PREPROCESSING
# ======================

def preprocess_data(data, lookback=168, forecast_horizon=12):
    features = data[['Close', 'RSI', 'MACD', 'Signal']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data) - forecast_horizon):
        X.append(scaled_data[i - lookback:i])
        
        current_price = data['Close'].values[i]
        future_price = data['Close'].values[i + forecast_horizon]
        target = 1 if future_price > current_price else 0
        y.append(target)
        
    return np.array(X), np.array(y), scaler

# ======================
# MODEL ARCHITECTURE
# ======================

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model

# ======================
# TRAINING & PREDICTION
# ======================

def train_model():
    data = load_gold_data(interval='15m')
    
    # Verify data length
    min_required = 168 + 12 + 1  # lookback + horizon + buffer
    if len(data) < min_required:
        raise ValueError(f"Need at least {min_required} data points. Got {len(data)}")
            
    # Prepare dataset
    lookback = 168  # For example, 168 periods lookback
    X, y, scaler = preprocess_data(data, lookback=lookback)
    
    # Time-series split
    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    # Build model
    model = build_lstm_model((lookback, X.shape[2]))
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_gold_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    early_stop = EarlyStopping(patience=7, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    return model, scaler, history

def predict_next_12h(model, scaler):
    # Load latest data
    fresh_data = load_gold_data(interval='15m')
    fresh_data = calculate_technical_indicators(fresh_data[-168:])  # Last 168 periods
    
    # Prepare input
    features = fresh_data[['Close', 'RSI', 'MACD', 'Signal']].values
    scaled_input = scaler.transform(features)[-168:]
    scaled_input = scaled_input.reshape(1, 168, 4)
    
    # Predict
    prob_up = model.predict(scaled_input)[0][0]
    return prob_up

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    if not os.path.exists('best_gold_model.keras'):
        print("Training new model...")
        model, scaler, _ = train_model()
        # Save the fitted scaler for later use
        joblib.dump(scaler, 'scaler.pkl')
    else:
        print("Loading existing model...")
        model = load_model('best_gold_model.keras')
        # Load the fitted scaler
        scaler = joblib.load('scaler.pkl')
    
    probability = predict_next_12h(model, scaler)
    direction = "UP" if probability > 0.6 else "DOWN"
    confidence = max(probability, 1 - probability)
    
    print(f"\nGold Price Prediction (Next 12 hours):")
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Probability: {probability*100:.1f}%")
