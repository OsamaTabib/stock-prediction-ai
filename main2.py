import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from joblib import dump, load

# ======================
# IMPROVED DATA PIPELINE
# ======================

def load_gold_data(interval='2m', cache_dir='gold_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    ticker = 'GC=F'
    cache_file = os.path.join(cache_dir, f'gold_{interval}.csv')
    
    if os.path.exists(cache_file):
        try:
            # Explicit date format specification
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Validate numeric data
            data = data.apply(pd.to_numeric, errors='coerce').dropna()
            if not data.empty and 'Close' in data.columns:
                return data
            os.remove(cache_file)
        except Exception as e:
            print(f"Cache error: {str(e)}, re-downloading...")
            os.remove(cache_file)
    
    print(f"Downloading fresh gold data ({interval} intervals)...")
    try:
        data = yf.download(ticker, period='60d', interval=interval)
        data = data[['Close']].dropna()
        # Ensure proper datetime formatting
        data.index = pd.to_datetime(data.index, utc=True)
        data.to_csv(cache_file, index=True, index_label='Date')
        return data
    except Exception as e:
        raise ValueError(f"Data download failed: {str(e)}")

# ======================
# FIXED SCALER IMPLEMENTATION
# ======================

def create_scaler(data):
    scaler = MinMaxScaler()
    # Fit on DataFrame to preserve feature names
    scaler.fit(data[['Close']])
    return scaler

# ======================
# MODEL ARCHITECTURE
# ======================

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ======================
# TRAINING WORKFLOW
# ======================

def train_model():
    data = load_gold_data()
    
    # Validate data quality
    if data.empty or 'Close' not in data.columns:
        raise ValueError("Invalid training data - missing 'Close' prices")
    
    # Create scaler with proper feature handling
    scaler = create_scaler(data)
    scaled_data = scaler.transform(data[['Close']])
    
    lookback = 168  # 7 days of 15m intervals
    X, y = [], []
    
    for i in range(lookback, len(scaled_data)-1):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i+1])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Build and train model
    model = build_lstm_model((lookback, 1))
    
    checkpoint = ModelCheckpoint(
        'gold_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop = EarlyStopping(patience=10)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    dump(scaler, 'gold_scaler.joblib')
    return model, scaler

# ======================
# PREDICTION FUNCTION
# ======================

def predict_future_price(model, scaler, hours=12):
    data = load_gold_data()
    
    # Validate data
    if data.empty or 'Close' not in data.columns:
        raise ValueError("Invalid data for prediction")
    
    scaled_data = scaler.transform(data[['Close']])
    last_sequence = scaled_data[-168:]
    
    steps = hours * 4  # Convert hours to 15m intervals
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        pred = model.predict(current_sequence.reshape(1, 168, 1), verbose=0)[0][0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    # Fixed line with proper parentheses
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    current_price = data['Close'].iloc[-1].item()
    predicted_price = float(predicted_prices[-1][0])
    
    return {
        'current': current_price,
        'predicted': predicted_price,
        'change_pct': ((predicted_price - current_price) / current_price) * 100
    }

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    try:
        # Clean environment
        if not os.path.exists('gold_cache'):
            os.makedirs('gold_cache', exist_ok=True)
        
        # Get user input
        hours = int(input("Enter prediction hours (1-24): ").strip() or 12)
        hours = max(1, min(24, hours))
        
        # Model management
        if not os.path.exists('gold_model.keras'):
            print("Training new model...")
            model, scaler = train_model()
        else:
            model = load_model('gold_model.keras')
            scaler = load('gold_scaler.joblib')
        
        # Get prediction
        result = predict_future_price(model, scaler, hours)
        
        print(f"\nGold Price Prediction (Next {hours} hours)")
        print(f"Current Price: ${result['current']:.2f}")
        print(f"Predicted Price: ${result['predicted']:.2f}")
        print(f"Predicted Change: {result['change_pct']:.2f}%")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("Process completed.")