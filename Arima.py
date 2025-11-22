import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

st.set_page_config(page_title="Asian Paints ARIMA Forecast", layout="wide")

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def get_data(start, end):
    df = yf.download("ASIANPAINT.NS", start=start, end=end, interval="1mo")
    df = df["Close"].dropna()
    return df

def build_model(series):
    model = ARIMA(series, order=(1, 1, 1))
    fitted = model.fit()
    return fitted

def forecast_with_match(fitted, actual):
    steps = len(actual)  # match number of actual data points
    fc = fitted.forecast(steps=steps)
    fc.index = actual.index
    return fc

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
    ax.plot(forecast, label="Forecast", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def observation_text(mape, rmse):
    text = f"""
### 📘 Observation  
The ARIMA model was evaluated by comparing forecasted values with actual market prices.

- **MAPE: {mape:.4f}** indicates the average forecasting error percentage.  
- **RMSE: {rmse:.4f}** reflects how far predictions deviate from true values.  

**Interpretation:**
- A lower MAPE (<0.10) means the model fits the market trend very well.
- RMSE closer to zero means high forecasting accuracy.
- If the forecast line closely follows the actual price in the graph, the model is reliable.  
- Any sharp deviation suggests volatility or sudden market shocks not captured by ARIMA.

This analysis helps investors understand how accurately statistical models can predict Asian Paints’ price movement.
"""
    return text


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.title("📈 Asian Paints ARIMA Forecasting & Model Evaluation")

project = st.selectbox(
    "Choose Model",
    ["Project 1 (2010–2018 → Forecast 2019)",
     "Project 2 (2021–2025 → Forecast 2026)"]
)

# -------------------------------------------------------
# PROJECT 1
# -------------------------------------------------------

if project.startswith("Project 1"):

    st.header("📌 Project 1: ARIMA Based Forecasting (2010–2018 → 2019)")

    train = get_data("2010-01-01", "2019-01-01")
    actual_2019 = get_data("2019-01-01", "2020-01-01")

    st.subheader("📊 Price Trend (2010–2018)")
    plot_series("Price Trend", train)

    fitted = build_model(train)
    forecast_2019 = forecast_with_match(fitted, actual_2019)

    st.subheader("📉 Forecast vs Actual (2019)")
    total = pd.concat([train, actual_2019])
    plot_overlay("Forecast vs Actual (2010–2019)", total, forecast_2019)

    # table
    comparison = pd.DataFrame({
        "Forecast": forecast_2019.values,
        "Actual": actual_2019.values
    }, index=actual_2019.index)

    st.subheader("📘 Forecast vs Actual Data (2019)")
    st.dataframe(comparison)

    mape = mean_absolute_percentage_error(actual_2019.values, forecast_2019.values)
    rmse = mean_squared_error(actual_2019.values, forecast_2019.values, squared=False)

    col1, col2 = st.columns(2)
    col1.metric("MAPE", f"{mape:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")

    st.markdown(observation_text(mape, rmse))


# -------------------------------------------------------
# PROJECT 2
# -------------------------------------------------------

if project.startswith("Project 2"):

    st.header("📌 Project 2: ARIMA Based Forecasting (2021–2025 → 2026)")

    train = get_data("2021-01-01", "2025-01-01")
    actual_2025 = get_data("2025-01-01", "2026-01-01")

    st.subheader("📊 Price Trend (2021–2025)")
    plot_series("Price Trend", train)

    fitted = build_model(train)
    forecast_2026 = forecast_with_match(fitted, actual_2025)

    st.subheader("📉 Forecast vs Actual (2025)")
    total = pd.concat([train, actual_2025])
    plot_overlay("Forecast vs Actual (2021–2026)", total, forecast_2026)

    comparison = pd.DataFrame({
        "Forecast": forecast_2026.values,
        "Actual": actual_2025.values
    }, index=actual_2025.index)

    st.subheader("📘 Forecast vs Actual Data (2025)")
    st.dataframe(comparison)

    mape = mean_absolute_percentage_error(actual_2025.values, forecast_2026.values)
    rmse = mean_squared_error(actual_2025.values, forecast_2026.values, squared=False)

    col1, col2 = st.columns(2)
    col1.metric("MAPE", f"{mape:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")

    st.markdown(observation_text(mape, rmse))
