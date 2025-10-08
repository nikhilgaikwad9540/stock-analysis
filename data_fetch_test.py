import yfinance as yf
import pandas as pd

def download(ticker='AAPL', start='2018-01-01', end=None, interval='1d'):
    # auto_adjust=True adjusts for splits/dividends
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    return df

# Example: Apple stock from 2019
df = download('AAPL', '2019-01-01')
print(df.head())

# Save to CSV for reference
df.to_csv('AAPL_data.csv')
print("\n✅ Data saved as 'AAPL_data.csv' in your project folder!")
