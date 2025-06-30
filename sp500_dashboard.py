import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from datetime import datetime, date

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
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce').dt.date
        df = df[df['Date'].notna()]
        df.set_index('Date', inplace=True)

    # Clean numeric columns
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    if 'Change_Pct' in df.columns:
        df['Return'] = df['Change_Pct'].str.replace('%', '').astype(float) / 100

    if 'Return' not in df.columns and 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change()

    return df

# Volatility calculation
def compute_volatility(df, window=20):
    if 'Return' not in df.columns:
        st.error("Missing 'Return' column.")
        st.stop()
    df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
    return df

# HMM fit
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

# Prophet forecast
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

# Signal generation
def aggregate_signals(df):
    df['Signal'] = 'Neutral'
    if 'HMM_State' in df.columns:
        df.loc[(df['Return'] > 0) & (df['HMM_State'] == 2), 'Signal'] = 'Bullish'
        df.loc[(df['Return'] < 0) & (df['HMM_State'] == 0), 'Signal'] = 'Bearish'
    return df

# Sidebar setup
st.sidebar.title("⚙️ Settings")
data_dir = "data"

if not os.path.exists(data_dir):
    st.error(f"Missing folder: {data_dir}")
    st.stop()

csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    st.error("No CSV files found in /data folder.")
    st.stop()

display_names = [f.replace(".csv", "") for f in csv_files]
selected_display = st.sidebar.selectbox("Select Index", display_names)
index_choice = csv_files[display_names.index(selected_display)]

df = load_csv_from_file(index_choice)

# Start date limited to actual range
start_date = st.sidebar.date_input("Start Date", datetime(2025, 1, 2).date(), min(df.index), min_value=min(df.index), max_value=max(df.index))
df = df[df.index >= start_date]

df = compute_volatility(df)
df, hmm_model = fit_hmm(df)
df = aggregate_signals(df)

st.markdown(f"**Dataset Loaded:** `{selected_display}`")

# Section: Cumulative Returns
st.subheader("Cumulative Returns")
st.markdown("Cumulative return shows how much the index value has grown over time since the selected start date.")
st.line_chart((1 + df['Return']).cumprod())

# Section: Volatility
st.subheader("Volatility (Rolling)")
st.markdown("Rolling volatility helps track how volatile the index has been across a moving window.")
vol_cols = [col for col in df.columns if "Volatility" in col]
st.line_chart(df[vol_cols])

# Section: HMM States
st.subheader("Market Regimes via HMM")
st.markdown("Hidden Markov Model (HMM) detects different market states based on historical return patterns.")
df_plot = df.reset_index()
if 'Date' not in df_plot.columns:
    df_plot.rename(columns={df_plot.columns[0]: 'Date'}, inplace=True)
df_plot.columns = [col.strip() for col in df_plot.columns]
required_cols = ['Date', 'Close', 'HMM_State']
missing_cols = [col for col in required_cols if col not in df_plot.columns]
if not missing_cols:
    fig_hmm = px.scatter(df_plot, x='Date', y='Close', color='HMM_State', title="Market States")
    st.plotly_chart(fig_hmm, use_container_width=True)

# Section: Prophet Forecast
st.subheader("60-Day Forecast using Prophet")
st.markdown("Forecast using Prophet model. `yhat` is the predicted value; `yhat_lower` and `yhat_upper` form the confidence interval.")
forecast = forecast_prophet(df)
if forecast is not None:
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='yhat_lower', line=dict(color='lightblue')))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='yhat_upper', line=dict(color='lightblue')))
    fig_forecast.update_layout(title="60-Day Forecast using Prophet")
    st.plotly_chart(fig_forecast, use_container_width=True)
    with st.expander("ℹ️ Forecast Legend"):
        st.markdown("""
        - **yhat**: Predicted future value.
        - **yhat_lower**/**yhat_upper**: Confidence bounds (95%) for uncertainty in the forecast.
        """)

# Section: Final Signal
st.subheader("Latest Signal Snapshot")
st.markdown("This shows the most recent market signal derived from return, volatility and model states.")
cols = ['Close', 'Return', 'HMM_State', 'Signal']
latest_row = df[[c for c in cols if c in df.columns]].iloc[:-5]
st.dataframe(latest_row)
