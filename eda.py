# eda.py
"""
Exploratory Data Analysis for stock data.
Reads AAPL_features.csv and shows:
- Candlestick + volume chart
- Rolling volatility
- Correlation heatmap
- Summary statistics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def load_features(csv_file="AAPL_features.csv"):
    """Load preprocessed stock features"""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df

def candlestick_chart(df):
    """Interactive candlestick chart with volume"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    )])
    fig.add_bar(x=df.index, y=df['Volume'], name="Volume", yaxis='y2')
    # Add secondary y-axis for volume
    fig.update_layout(
        title="Candlestick chart with Volume",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False, position=0.15),
        xaxis_rangeslider_visible=False
    )
    fig.show()

def rolling_volatility(df, window=20):
    """Plot rolling volatility (std of returns)"""
    df['rolling_vol'] = df['return'].rolling(window).std()
    fig = px.line(df, x=df.index, y='rolling_vol', title=f'{window}-day Rolling Volatility')
    fig.show()

def correlation_heatmap(df):
    """Correlation heatmap of numeric features"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def summary_statistics(df):
    """Print summary statistics of numeric features"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    print("\nSummary Statistics:")
    print(df[numeric_cols].describe().T)

if __name__ == "__main__":
    df = load_features("AAPL_features.csv")
    summary_statistics(df)
    candlestick_chart(df)
    rolling_volatility(df)
    correlation_heatmap(df)
