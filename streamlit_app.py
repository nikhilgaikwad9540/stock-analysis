# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------- Helper Functions ---------

@st.cache_data
def load_features(csv_file="AAPL_features.csv"):
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df

def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
    )])
    fig.update_layout(title='Candlestick Chart', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_rolling_vol(df, window=20):
    df['rolling_vol'] = df['return'].rolling(window).std()
    fig = px.line(df, x=df.index, y='rolling_vol', title=f'{window}-day Rolling Volatility')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def train_rf_predict(df):
    """Train RandomForest to predict next-day Close"""
    df['target'] = df['Close'].shift(-1)
    features = ['Close','return','SMA_20','EMA_20','RSI_14','MACD','MACD_signal']
    df_model = df.dropna()
    X = df_model[features]
    y = df_model['target']
    
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, pred)
    st.write(f"**RandomForest MAE:** {mae:.2f}")
    
    # Show next-day prediction
    next_day_pred = model.predict(X.iloc[[-1]])[0]
    st.write(f"**Next-day Close Prediction:** {next_day_pred:.2f}")
    
def backtest_sma(df):
    """Simple SMA Crossover Backtest"""
    df['SMA_short'] = df['Close'].rolling(50).mean()
    df['SMA_long']  = df['Close'].rolling(200).mean()
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1
    df['strategy_return'] = df['signal'].shift(1) * df['return']
    df['cum_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['cum_buyhold'] = (1 + df['return'].fillna(0)).cumprod()
    
    fig = px.line(df, x=df.index, y=['cum_strategy','cum_buyhold'],
                  labels={'value':'Cumulative Returns', 'variable':'Strategy'},
                  title='Backtest: SMA Crossover vs Buy & Hold')
    st.plotly_chart(fig, use_container_width=True)

# --------- Streamlit Layout ---------

st.set_page_config(page_title="Stock Analysis Dashboard", page_icon="💹", layout="wide")
st.title("💹 Stock Analysis Dashboard")
st.markdown("A professional demo showing **EDA, Technical Indicators, Prediction, and Backtesting**.")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if st.sidebar.button("Load Data"):
    df = load_features("AAPL_features.csv")  # Using preprocessed CSV
    
    # Filter by dates
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    
    st.subheader(f"Data Preview: {ticker}")
    st.dataframe(df.tail(10))
    
    st.subheader("Candlestick Chart")
    plot_candlestick(df)
    
    st.subheader("Rolling Volatility")
    plot_rolling_vol(df)
    
    st.subheader("Correlation Heatmap")
    plot_correlation_heatmap(df)
    
    st.subheader("Next-day Prediction (RandomForest)")
    train_rf_predict(df)
    
    st.subheader("Backtesting SMA Crossover")
    backtest_sma(df)
    
    st.success("✅ Dashboard Loaded Successfully!")
