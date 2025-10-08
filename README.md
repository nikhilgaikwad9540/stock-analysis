# 💹 Stock Analysis Dashboard

An interactive **Stock Market Analysis Dashboard** built with **Python**, **Streamlit**, and **Machine Learning**.  
This project demonstrates data fetching, preprocessing, exploratory data analysis (EDA), predictive modeling, and trading strategy backtesting — all in one beautiful, professional web interface.

---

## 🚀 Features

### 🧩 Data Preprocessing
- Fetch stock data using `yfinance`
- Compute key technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Handle missing values and create lag features for modeling

### 📊 Exploratory Data Analysis
- Interactive candlestick charts (Plotly)
- Volatility visualization
- Correlation heatmaps
- Summary statistics and feature overview

### 🤖 Machine Learning Prediction
- Uses **RandomForestRegressor** to predict the next-day closing price
- Displays MAE and predicted price

### 📈 Backtesting
- Simple **SMA crossover** trading strategy vs. buy-and-hold comparison
- Visualizes cumulative returns interactively

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Data Fetching | yfinance |
| Data Processing | pandas, numpy |
| Visualization | Plotly, Seaborn, Matplotlib |
| Machine Learning | scikit-learn |
| Web App | Streamlit |

---

## 🏗️ Project Structure


How to run this code 

---

## ⚙️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/<YOUR_USERNAME>/stock_analysis.git
cd stock_analysis

# 2. Create a virtual environment
python -m venv venv
source venv/Scripts/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt


run the streamlit app
streamlit run streamlit_app.py

