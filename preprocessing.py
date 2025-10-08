# preprocessing.py
"""
Preprocessing and feature engineering for stock data without pandas-ta.
Fixes numeric type issues.
"""

import pandas as pd
import numpy as np

def preprocess_stock_data(csv_file):
    # Load CSV and parse date column
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    # Convert numeric columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs in numeric columns
    df = df.dropna()

    # 1. Returns
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # 3. RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # 5. Bollinger Bands (20-day SMA + 2 std)
    df['BB_middle'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

    # 6. Lag features
    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['SMA_20_lag1'] = df['SMA_20'].shift(1)
    df['EMA_20_lag1'] = df['EMA_20'].shift(1)

    # Drop rows with NaNs from rolling/lag calculations
    df = df.dropna()

    return df

# Example usage
if __name__ == "__main__":
    df_features = preprocess_stock_data("AAPL_data.csv")
    print(df_features.tail())
    df_features.to_csv("AAPL_features.csv")
    print("\n✅ Preprocessed features saved as 'AAPL_features.csv'")
