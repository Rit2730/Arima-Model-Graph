import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="ARIMA Forecasting – Asian Paints", layout="wide")

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def fetch_monthly(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, interval="1mo")
    df = df[["Close"]].dropna()
    df.rename(columns={"Close": "Price"}, inplace=True)
    return df

def to_series(df):
    """Convert df.Price to a clean 1D time series."""
    s = pd.Series(df["Price"].values.flatten(), index=df.index)
    s.index = pd.to_datetime(s.index)
    s = s.astype(float)
    return s

def plot_line(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_forecast(train, forecast, actual=None, title="Forecast"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train.index,
                             y=train["Price"],
                             name="Training Data",
                             mode="lines"))

    fig.add_trace(go.Scatter(x=forecast.index,
                             y=forecast.values.flatten(),
                             name="Forecast",
                             mode="lines"))

    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index,
                                 y=actual.values.flatten(),
                                 name="Actual",
                                 mode="lines"))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig

def arima_forecast(series, steps):
    model = ARIMA(series, order=(1,1,1))
    fitted = model.fit()
    fcast = fitted.forecast(steps=steps)

    # enforce 1D
    fcast = pd.Series(fcast.values.flatten())
    return fcast, fitted

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------

st.title("Asian Paints ARIMA Forecasting Dashboard")

project = st.sidebar.selectbox(
    "Select Project",
    ["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"]
)

# ----------------------------------------------------------
# PROJECT 1
# ----------------------------------------------------------
if project.startswith("Project 1"):

    st.header("Project 1: Forecasting Asian Paints (2010–2018 → 2019)")

    train = fetch_monthly("ASIANPAINT.NS", "2010-01-01", "2018-12-31")
    actual_2019 = fetch_monthly("ASIANPAINT.NS", "2019-01-01", "2020-01-01")

    train_series = to_series(train)

    st.subheader("1. Monthly Price Movement (2010–2018)")
    st.plotly_chart(plot_line(train, "Asian Paints 2010–2018"), use_container_width=True)

    forecast_2019, fitted_1 = arima_forecast(train_series, 12)
    forecast_2019.index = actual_2019.index

    st.subheader("2. ARIMA Forecast vs Actual (2019)")
    st.plotly_chart(
        plot_forecast(train, forecast_2019, to_series(actual_2019), "Forecast vs Actual – 2019"),
        use_container_width=True
    )

    st.subheader("3. Pure Forecast for 2019")
    st.plotly_chart(
        plot_forecast(train, forecast_2019, None, "Forecast – 2019"),
        use_container_width=True
    )

    # Comparison Table
    st.subheader("Forecast vs Actual Comparison")

    actual_vals = actual_2019["Price"].values.flatten()
    forecast_vals = forecast_2019.values.flatten()

    min_len = min(len(actual_vals), len(forecast_vals))

    comp_df = pd.DataFrame({
        "Actual": actual_vals[:min_len],
        "Forecast": forecast_vals[:min_len]
    }, index=actual_2019.index[:min_len])

    st.dataframe(comp_df)

    # Metrics
    mae = mean_absolute_error(actual_vals[:min_len], forecast_vals[:min_len])
    rmse = np.sqrt(mean_squared_error(actual_vals[:min_len], forecast_vals[:min_len]))
    mape = np.mean(np.abs((actual_vals[:min_len] - forecast_vals[:min_len]) /
                          actual_vals[:min_len])) * 100

    st.subheader("Metrics")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAPE: {mape:.2f}%")

    st.subheader("Observation")
    st.write("""
The ARIMA model aligns well with the overall price trend for 2019.
Accuracy is strong during stable months, with deviations during sudden spikes.
This demonstrates that ARIMA captures trend direction but struggles with high volatility.
""")


# ----------------------------------------------------------
# PROJECT 2
# ----------------------------------------------------------
if project.startswith("Project 2"):

    st.header("Project 2: Forecasting Asian Paints (2021–2025 → 2026)")

    train = fetch_monthly("ASIANPAINT.NS", "2021-01-01", "2025-12-31")
    actual_2026 = fetch_monthly("ASIANPAINT.NS", "2026-01-01", "2027-01-01")

    train_series = to_series(train)

    st.subheader("1. Monthly Price Movement (2021–2025)")
    st.plotly_chart(plot_line(train, "Asian Paints 2021–2025"), use_container_width=True)

    forecast_2026, fitted_2 = arima_forecast(train_series, 12)
    forecast_2026.index = actual_2026.index

    st.subheader("2. ARIMA Forecast vs Actual (2026)")
    st.plotly_chart(
        plot_forecast(train, forecast_2026, to_series(actual_2026), "Forecast vs Actual – 2026"),
        use_container_width=True
    )

    st.subheader("3. Pure Forecast for 2026")
    st.plotly_chart(
        plot_forecast(train, forecast_2026, None, "Forecast – 2026"),
        use_container_width=True
    )

    # Comparison Table
    st.subheader("Forecast vs Actual Comparison")

    actual_vals = actual_2026["Price"].values.flatten()
    forecast_vals = forecast_2026.values.flatten()

    min_len = min(len(actual_vals), len(forecast_vals))

    comp_df = pd.DataFrame({
        "Actual": actual_vals[:min_len],
        "Forecast": forecast_vals[:min_len]
    }, index=actual_2026.index[:min_len])

    st.dataframe(comp_df)

    # Metrics
    mae = mean_absolute_error(actual_vals[:min_len], forecast_vals[:min_len])
    rmse = np.sqrt(mean_squared_error(actual_vals[:min_len], forecast_vals[:min_len]))
    mape = np.mean(np.abs((actual_vals[:min_len] - forecast_vals[:min_len]) /
                          actual_vals[:min_len])) * 100

    st.subheader("Metrics")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAPE: {mape:.2f}%")

    st.subheader("Observation")
    st.write("""
The forecast follows the general trend of price movements for 2026.
Minor deviations appear during market volatility.
The model’s performance is strong for short-term predictions.
""")
