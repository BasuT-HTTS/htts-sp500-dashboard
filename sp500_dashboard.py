import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from hmmlearn.hmm import GaussianHMM

st.set_page_config(layout="wide")
st.title("📊 S&P 500 + Multi-Index Dashboard")

# Load from CSV with full cleaning
def load_csv_from_file(filename):
    df = pd.read_csv(os.path.join(data_dir, filename))
    df.columns = [col.strip() for col in df.columns]

    rename_map = {
        "Price": "Close",
        "Vol.": "Volume",
        "Change %": "Change_Pct"
    }
    df.rename(columns=rename_map, inplace=True)

    # Date parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        df = df[df['Date'].notna()]  # Drop rows with bad dates
        df.set_index('Date', inplace=True)

    # Clean numeric columns
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # Convert percentage string to float
    if 'Change_Pct' in df.columns:
        df['Return'] = df['Change_Pct'].str.replace('%', '').astype(float) / 100

    # Fallback: compute return from Close
    if 'Return' not in df.columns and 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()

    df = df[df.index >= pd.to_datetime(start_date)]

    return df
    # return df.dropna()

# Rolling volatility
def compute_volatility(df, window=20):
    if 'Return' not in df.columns:
        st.error("Missing 'Return' column.")
        st.stop()
    df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
    return df

# HMM
def fit_hmm(df, n_states=3):
    returns = df[['Return']].dropna().values
    if returns.shape[0] < n_states * 10:
        st.warning(f"Not enough data for HMM (found {returns.shape[0]} rows).")
        df['HMM_State'] = 1
        return df, None
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(returns)
    df['HMM_State'] = model.predict(returns)
    return df, model

# Prophet
def forecast_prophet(df, days=60):
    df_prophet = df.reset_index()[['Date', 'Close']].dropna()
    if df_prophet.shape[0] < 2:
        st.warning("Not enough data for forecasting.")
        return None
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# Signal logic
def aggregate_signals(df):
    df['Signal'] = 'Neutral'
    if 'HMM_State' in df.columns:
        df.loc[(df['Return'] > 0) & (df['HMM_State'] == 2), 'Signal'] = 'Bullish'
        df.loc[(df['Return'] < 0) & (df['HMM_State'] == 0), 'Signal'] = 'Bearish'
    return df

# Sidebar
st.sidebar.title("⚙️ Options")
data_dir = "data"
if not os.path.exists(data_dir):
    st.error(f"Missing folder: {data_dir}")
    st.stop()

csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    st.error("No CSV files found in /data folder.")
    st.stop()

index_choice = st.sidebar.selectbox("Select Dataset", csv_files)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

# Main logic
df = load_csv_from_file(index_choice)
df = compute_volatility(df)
df, hmm_model = fit_hmm(df)
df = aggregate_signals(df)

st.markdown(f"**Dataset Loaded:** `{index_choice}`")

st.subheader("Cumulative Returns")
st.line_chart((1 + df['Return']).cumprod())

st.subheader("Volatility")
vol_cols = [col for col in df.columns if "Volatility" in col]
st.line_chart(df[vol_cols])

st.subheader("HMM Market Regimes")
df_plot = df.reset_index()
if 'Date' not in df_plot.columns:
    df_plot.rename(columns={df_plot.columns[0]: 'Date'}, inplace=True)
df_plot.columns = [col.strip() for col in df_plot.columns]
required_cols = ['Date', 'Close', 'HMM_State']
missing_cols = [col for col in required_cols if col not in df_plot.columns]
if missing_cols:
    st.error(f"Missing columns for HMM plot: {missing_cols}")
else:
    fig_hmm = px.scatter(df_plot, x='Date', y='Close', color='HMM_State', title="Market States")
    st.plotly_chart(fig_hmm, use_container_width=True)

st.subheader("60-Day Forecast using Prophet")
forecast = forecast_prophet(df)
if forecast is not None:
    fig_forecast = px.line(forecast, x='ds', y='yhat', title="Forecast")
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='yhat_lower')
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='yhat_upper')
    st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Aggregated Signal Snapshot")
cols = ['Close', 'Return', 'HMM_State', 'Signal']
st.dataframe(df[[c for c in cols if c in df.columns]].tail(10))
