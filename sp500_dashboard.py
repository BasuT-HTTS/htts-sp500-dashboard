import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from hmmlearn.hmm import GaussianHMM
from io import StringIO

st.set_page_config(layout="wide")
st.title("📊 S&P 500 + Multi-Index Dashboard")

# Load data
def load_data(start_date, ticker='^GSPC'):
    df = yf.download(ticker, start=start_date)
    df['Return'] = df['Close'].pct_change()
    return df.dropna()

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=True)
    df.columns = [col.strip() for col in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    if 'Close' in df.columns and 'Open' in df.columns:
        df['Return'] = df['Close'].pct_change()
    return df.dropna()

# Compute rolling volatility
def compute_volatility(df, window=20):
    df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
    return df

# Fit HMM
def fit_hmm(df, n_states=3):
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    returns = df[['Return']].dropna().values
    model.fit(returns)
    df['HMM_State'] = model.predict(returns)
    return df, model

# Prophet forecast
def forecast_prophet(df, days=60):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# Aggregate signals
def aggregate_signals(df):
    df['Signal'] = 'Neutral'
    df.loc[(df['Return'] > 0) & (df['HMM_State'] == 2), 'Signal'] = 'Bullish'
    df.loc[(df['Return'] < 0) & (df['HMM_State'] == 0), 'Signal'] = 'Bearish'
    return df

# Sidebar options
st.sidebar.title("⚙️ Options")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

source = st.sidebar.radio("Data Source", ["Yahoo Finance", "Upload CSV"])
uploaded_file = None
symbol = None

if source == "Yahoo Finance":
    symbol = st.sidebar.selectbox("Select Index", ["^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN"])
    df = load_data(start_date, symbol)
else:
    uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

df = compute_volatility(df)
df, hmm_model = fit_hmm(df)
df = aggregate_signals(df)

st.subheader("Cumulative Returns")
st.line_chart((1 + df['Return']).cumprod())

st.subheader("Volatility")
vol_cols = [col for col in df.columns if "Volatility" in col]
st.line_chart(df[vol_cols])

st.subheader("HMM Market Regimes")
fig_hmm = px.scatter(df, x=df.index, y='Close', color='HMM_State', title="Market States")
st.plotly_chart(fig_hmm, use_container_width=True)

st.subheader("60-Day Forecast using Prophet")
forecast = forecast_prophet(df)
fig_forecast = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title="Forecast")
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Aggregated Signal Snapshot")
st.dataframe(df[['Close', 'Return', 'HMM_State', 'Signal']].tail(10))
