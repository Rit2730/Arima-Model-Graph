import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Asian Paints ARIMA Forecasting", layout="wide")

# ---------------------------
# FUNCTIONS
# ---------------------------

def download_data(start, end):
    return yf.download("ASIANPAINT.NS", start=start, end=end, interval="1mo")["Close"]

def build_arima(series):
    model = ARIMA(series, order=(1,1,1))
    fitted = model.fit()
    return fitted

def forecast(fitted, steps):
    f = fitted.forecast(steps=steps)
    return f

def plot_line(title, series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def plot_forecast_vs_actual(title, series, forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="Actual", linewidth=2)
    ax.plot(forecast, label="Forecast", linestyle="--")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def compute_metrics(actual, forecast):
    mape = mean_absolute_percentage_error(actual, forecast)
    rmse = mean_squared_error(actual, forecast, squared=False)
    return mape, rmse

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("📈 Asian Paints ARIMA Forecasting App")
st.markdown("Created for **Project 1 & Project 2** with full forecasting comparison.")

project = st.selectbox("Select Project:", ["Project 1 (2010–2018 → Forecast 2019)", 
                                           "Project 2 (2021–2025 → Forecast 2026)"])

# ---------------------------
# PROJECT 1
# ---------------------------
if project == "Project 1 (2010–2018 → Forecast 2019)":

    st.header("📌 Project 1: 2010–2018 ARIMA Model")

    # Download data
    data = download_data("2010-01-01", "2019-01-01")
    actual_2019 = download_data("2019-01-01", "2020-01-01")

    st.subheader("Graph 1: Change in Price (2010–2018)")
    plot_line("Price Trend (2010–2018)", data)

    # Build ARIMA
    fitted = build_arima(data)

    # Forecast 12 months (2019)
    forecast_2019 = forecast(fitted, 12)
    forecast_2019.index = actual_2019.index  # align dates

    st.subheader("Graph 2: ARIMA Forecast Overlapped on Actual")
    plot_forecast_vs_actual("Forecast vs Actual (2010–2019)", 
                            pd.concat([data, actual_2019]), forecast_2019)

    # Future Graph
    st.subheader("Graph 3: Forecast for 2019")
    plot_line("Forecast Only (2019)", forecast_2019)

    # Comparison Table
    st.subheader("Comparison Table (Forecast vs Actual)")
    df = pd.DataFrame({
        "Forecast": forecast_2019.values,
        "Actual": actual_2019.values
    }, index=actual_2019.index)
    st.dataframe(df)

    # Accuracy Metrics
    mape, rmse = compute_metrics(actual_2019.values, forecast_2019.values)
    st.metric("MAPE", f"{mape:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")

# ---------------------------
# PROJECT 2
# ---------------------------
if project == "Project 2 (2021–2025 → Forecast 2026)":

    st.header("📌 Project 2: 2021–2025 ARIMA Model")

    # Download data
    data = download_data("2021-01-01", "2025-01-01")
    actual_2025 = download_data("2025-01-01", "2026-01-01")

    st.subheader("Graph 1: Change in Price (2021–2025)")
    plot_line("Price Trend (2021–2025)", data)

    # Build ARIMA
    fitted = build_arima(data)

    # Forecast 12 months (2025–2026)
    forecast_2026 = forecast(fitted, 12)
    forecast_2026.index = actual_2025.index

    st.subheader("Graph 2: ARIMA Forecast Overlapped on Actual")
    plot_forecast_vs_actual("Forecast vs Actual (2021–2026)", 
                            pd.concat([data, actual_2025]), forecast_2026)

    # Future Graph
    st.subheader("Graph 3: Forecast for 2025–2026")
    plot_line("Forecast Only (2025–2026)", forecast_2026)

    # Comparison Table
    st.subheader("Comparison Table (Forecast vs Actual)")
    df = pd.DataFrame({
        "Forecast": forecast_2026.values,
        "Actual": actual_2025.values
    }, index=actual_2025.index)
    st.dataframe(df)

    # Accuracy Metrics
    mape, rmse = compute_metrics(actual_2025.values, forecast_2026.values)
    st.metric("MAPE", f"{mape:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
