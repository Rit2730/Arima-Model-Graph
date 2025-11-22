import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Asian Paints ARIMA Forecast", layout="wide")

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------

def get_data(start, end):
    df = yf.download("ASIANPAINT.NS", start=start, end=end, interval="1mo")
    df = df["Close"].dropna()
    return df

def build_model(series):
    model = ARIMA(series, order=(1,1,1))
    fitted = model.fit()
    return fitted

def forecast(fitted, steps, index_ref):
    f = fitted.forecast(steps=steps)
    f.index = index_ref.index
    return f

def plot_series(title, series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def plot_overlay(title, actual, forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label="Actual", linewidth=2)
    ax.plot(forecast, label="Forecast", linestyle="--")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.title("📈 Asian Paints ARIMA Forecasting App")

project = st.selectbox(
    "Choose Project",
    ["Project 1 (2010–2018 → Forecast 2019)",
     "Project 2 (2021–2025 → Forecast 2026)"]
)

# ------------------------------------------------
# PROJECT 1
# ------------------------------------------------

if project.startswith("Project 1"):

    st.header("📌 Project 1: 2010–2018 ARIMA Model")

    train = get_data("2010-01-01", "2019-01-01")
    actual_2019 = get_data("2019-01-01", "2020-01-01")

    st.subheader("Graph 1: Change in Price (2010–2018)")
    plot_series("Price Trend (2010–2018)", train)

    fitted = build_model(train)

    # 12-month forecast
    fcast_2019 = forecast(fitted, 12, actual_2019)

    st.subheader("Graph 2: ARIMA Forecast vs Actual")
    combined = pd.concat([train, actual_2019])
    plot_overlay("Forecast vs Actual (2010–2019)", combined, fcast_2019)

    st.subheader("Graph 3: Forecast Only (2019)")
    plot_series("Forecast (2019)", fcast_2019)

    # Table
    st.subheader("📘 Forecast vs Actual Table")
    df = pd.DataFrame({
        "Forecast": fcast_2019.values,
        "Actual": actual_2019.values
    }, index=actual_2019.index)
    st.dataframe(df)

    # Metrics
    mape = mean_absolute_percentage_error(actual_2019.values, fcast_2019.values)
    rmse = mean_squared_error(actual_2019.values, fcast_2019.values, squared=False)

    st.metric("MAPE", f"{mape:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")

# ------------------------------------------------
# PROJECT 2
# ------------------------------------------------

if project.startswith("Project 2"):

    st.header("📌 Project 2: 2021–2025 ARIMA Model")

    train = get_data("2021-01-01", "2025-01-01")
    actual_2025 = get_data("2025-01-01", "2026-01-01")

    st.subheader("Graph 1: Change in Price (2021–2025)")
    plot_series("Price Trend (2021–2025)", train)

    fitted = build_model(train)

    # 12-month forecast
    fcast_2026 = forecast(fitted, 12, actual_2025)

    st.subheader("Graph 2: ARIMA Forecast vs Actual")
    combined = pd.concat([train, actual_2025])
    plot_overlay("Forecast vs Actual (2021–2026)", combined, fcast_2026)

    st.subheader("Graph 3: Forecast Only (2025–2026)")
    plot_series("Forecast (2026)", fcast_2026)

    # Table
    st.subheader("📘 Forecast vs Actual Table")
    df = pd.DataFrame({
        "Forecast": fcast_2026.values,
        "Actual": actual_2025.values
    }, index=actual_2025.index)
    st.dataframe(df)

    # Metrics
    mape = mean_absolute_percentage_error(actual_2025.values, fcast_2026.values)
    rmse = mean_squared_error(actual_2025.values, fcast_2026.values, squared=False)

    st.metric("MAPE", f"{mape:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
