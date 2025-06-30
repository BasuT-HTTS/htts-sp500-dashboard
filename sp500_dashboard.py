import os
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

def load_csv_from_file(filename):
    df = pd.read_csv(os.path.join(data_dir, filename))
    df.columns = [col.strip() for col in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    if 'Close' in df.columns:
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

# # Sidebar options
# st.sidebar.title("⚙️ Options")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

st.sidebar.title("⚙️ Options")
data_dir = "data"  # Place all CSVs in this subfolder

# Automatically list all CSVs in /data
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
index_choice = st.sidebar.selectbox("Select Dataset", csv_files)

df = load_csv_from_file(index_choice)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))


# source = st.sidebar.radio("Data Source", ["Yahoo Finance", "Upload CSV"])
# uploaded_file = None
# symbol = None

# if source == "Yahoo Finance":
#     symbol = st.sidebar.selectbox("Select Index", ["^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN"])
#     df = load_data(start_date, symbol)
# else:
#     uploaded_file = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
#     if uploaded_file:
#         df = load_csv(uploaded_file)
#     else:
#         st.warning("Please upload a CSV file to proceed.")
#         st.stop()

df = compute_volatility(df)
df, hmm_model = fit_hmm(df)
df = aggregate_signals(df)

st.subheader("Cumulative Returns")
st.line_chart((1 + df['Return']).cumprod())

st.subheader("Volatility")
vol_cols = [col for col in df.columns if "Volatility" in col]
st.line_chart(df[vol_cols])

st.subheader("HMM Market Regimes")
# fig_hmm = px.scatter(df, x=df.index, y='Close', color='HMM_State', title="Market States")
# df_plot = df.reset_index()  # Flatten index

# df_plot = df.reset_index()
# # Rename index column to 'Date' if not already
# if 'Date' not in df_plot.columns:
#     df_plot.rename(columns={df_plot.columns[0]: 'Date'}, inplace=True)

df_plot = df.reset_index()

# Ensure 'Date' column exists
if 'Date' not in df_plot.columns:
    # Rename the first column to Date if unnamed
    first_col = df_plot.columns[0]
    if pd.api.types.is_datetime64_any_dtype(df_plot[first_col]):
        df_plot.rename(columns={first_col: 'Date'}, inplace=True)
    else:
        df_plot['Date'] = pd.date_range(start='2000-01-01', periods=len(df_plot))

# Clean any extra spaces from column names
df_plot.columns = [col.strip() for col in df_plot.columns]

# Ensure required columns are present
required_cols = ['Date', 'Close', 'HMM_State']
missing_cols = [col for col in required_cols if col not in df_plot.columns]
if missing_cols:
    st.error(f"Missing columns for HMM plot: {missing_cols}")
    st.stop()


fig_hmm = px.scatter(df_plot, x='Date', y='Close', color='HMM_State', title="Market States")

st.plotly_chart(fig_hmm, use_container_width=True)

st.subheader("60-Day Forecast using Prophet")
forecast = forecast_prophet(df)
# fig_forecast = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title="Forecast")
fig_forecast = px.line(forecast, x='ds', y='yhat', title="Forecast")
fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='yhat_lower')
fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='yhat_upper')

st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Aggregated Signal Snapshot")
st.dataframe(df[['Close', 'Return', 'HMM_State', 'Signal']].tail(10))
