import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Page config
st.set_page_config(page_title="Asian Paints - ARIMA Forecasting Dashboard", layout="wide")

# Helper functions
@st.cache_data
def load_monthly_close(ticker="ASIANPAINT.NS", period="25y"):
    df = yf.download(ticker, interval="1mo", period=period)
    if "Close" not in df.columns:
        raise RuntimeError("Could not retrieve Close price from yfinance.")
    series = df["Close"].dropna()
    # ensure monthly period start alignment to month start for consistent indexing
    series.index = pd.to_datetime(series.index).to_period('M').to_timestamp()
    return series

def fit_arima(series, order=(1,1,1)):
    # Ensure 1D series
    s = pd.Series(series).astype(float).dropna()
    model = ARIMA(s, order=order)
    fitted = model.fit()
    return fitted

def make_forecast(fitted, start_date_index, periods):
    # produce forecast and index it at monthly period starts
    fc = fitted.forecast(steps=periods)
    fc = pd.Series(np.array(fc).flatten())
    fc_index = pd.date_range(start=start_date_index, periods=periods, freq="MS")
    fc.index = fc_index
    return fc

def safe_series(x):
    # convert DataFrame/ndarray/Series into a 1-D Series
    s = pd.Series(x).squeeze()
    s = pd.Series(np.array(s).flatten(), index=pd.Index(s.index) if hasattr(s, 'index') else None)
    return s

def metrics(actual, forecast):
    actual = np.array(actual).astype(float)
    forecast = np.array(forecast).astype(float)
    # align lengths
    n = min(len(actual), len(forecast))
    actual = actual[:n]
    forecast = forecast[:n]
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if np.any(actual != 0) else np.nan
    return mae, rmse, mape

def df_to_csv_bytes(df):
    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    return buffer

# Load base dataset once
series_full = load_monthly_close()

# Sidebar and project selection
st.sidebar.header("Project selector")
project = st.sidebar.selectbox("Select project view", ["Project 1: 2010–2018 → Forecast 2019", "Project 2: 2021–2025 → Forecast 2026 (with backtest)"])

st.markdown("## Asian Paints - ARIMA Forecasting Dashboard")
st.markdown("This dashboard contains two independent project views. Each view shows three graphs, a forecast vs actual comparison, residual diagnostics, statistical summary, downloadable data and a concise professional observation. The ARIMA model is used exclusively as requested.")

# Shared ARIMA order controls
st.sidebar.header("ARIMA settings")
p = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1, step=1)
order = (int(p), int(d), int(q))

# ------------------------------------------------------------------------
# PROJECT 1: 2010–2018 train, forecast 2019 (compare with actual 2019)
# ------------------------------------------------------------------------
if project.startswith("Project 1"):
    st.header("Project 1: Training 2010–2018, Forecast 2019")

    # Prepare series
    train_start = "2010-01-01"
    train_end = "2018-12-31"
    forecast_start = "2019-01-01"
    train_series = series_full.loc[train_start:train_end]
    train_series = safe_series(train_series)

    # Validate enough training points (should be fine given 25y period)
    st.subheader("Graph 1: Monthly Price Movement (2010–2018)")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=train_series.index, y=train_series.values, mode="lines", name="Historical (2010-2018)"))
    fig1.update_layout(xaxis_title="Date", yaxis_title="Price", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # Fit model
    with st.spinner("Fitting ARIMA model for Project 1"):
        fitted1 = fit_arima(train_series, order=order)

    # Forecast 12 months (2019)
    forecast_periods = 12
    fcast1 = make_forecast(fitted1, forecast_start, forecast_periods)

    # Graph 2: Forecast overlaid on original (we overlay forecast with history + actual 2019 when available)
    st.subheader("Graph 2: ARIMA Forecast Overlaid on Actual")
    combined_index = train_series.index.union(fcast1.index)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_series.index, y=train_series.values, mode="lines", name="Train (2010-2018)"))
    fig2.add_trace(go.Scatter(x=fcast1.index, y=fcast1.values, mode="lines", name="Forecast (2019)"))
    # attempt to fetch actual 2019 for comparison and overlay it if present
    actual_2019 = series_full.loc[forecast_start:"2019-12-31"]
    if len(actual_2019) > 0:
        actual_2019 = safe_series(actual_2019)
        fig2.add_trace(go.Scatter(x=actual_2019.index, y=actual_2019.values, mode="lines", name="Actual (2019)"))
    fig2.update_layout(xaxis_title="Date", yaxis_title="Price", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Graph 3: Future forecast (out-of-sample)
    st.subheader("Graph 3: Forecasted Prices (2019)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fcast1.index, y=fcast1.values, mode="lines", name="Forecast 2019"))
    fig3.update_layout(xaxis_title="Date", yaxis_title="Price", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # Forecast vs Actual comparison table for 2019
    st.subheader("Forecast vs Actual Comparison (2019)")
    actual_2019 = series_full.loc[forecast_start:"2019-12-31"]
    actual_2019 = safe_series(actual_2019)
    # align lengths
    min_len = min(len(actual_2019), len(fcast1))
    comp1 = pd.DataFrame({
        "Forecast": fcast1.values[:min_len],
        "Actual": actual_2019.values[:min_len]
    }, index=actual_2019.index[:min_len])
    st.dataframe(comp1.style.format("{:.2f}"))

    # Metrics
    if len(comp1) > 0:
        mae1, rmse1, mape1 = metrics(comp1["Actual"].values, comp1["Forecast"].values)
        st.markdown("Model performance for 2019 (comparison of forecast vs actual):")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae1:.3f}")
        col2.metric("RMSE", f"{rmse1:.3f}")
        col3.metric("MAPE", f"{mape1:.2f} %")
    else:
        st.info("Actual data for 2019 is not available to compare; the forecast is still shown above.")

    # Residual diagnostics
    st.subheader("Residual Diagnostics")
    resid1 = fitted1.resid
    fig_res1 = px.line(x=resid1.index, y=resid1.values, labels={"x":"Date","y":"Residual"}, title="Residuals (Training Period)")
    st.plotly_chart(fig_res1, use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary (Training Data)")
    st.table(train_series.describe().to_frame(name="Value"))

    # Downloads and observation
    st.subheader("Downloads")
    st.download_button("Download Training Series CSV", df_to_csv_bytes(train_series.to_frame(name="Close")), file_name="project1_train_2010_2018.csv")
    st.download_button("Download Forecast 2019 CSV", df_to_csv_bytes(fcast1.to_frame(name="Forecast")), file_name="project1_forecast_2019.csv")
    if len(comp1) > 0:
        st.download_button("Download Comparison (2019) CSV", df_to_csv_bytes(comp1), file_name="project1_comparison_2019.csv")

    st.subheader("Observation")
    obs1 = (
        "The ARIMA( p,d,q ) model was fitted to monthly closing prices between 2010 and 2018. "
        "The out-of-sample forecast for 2019 is displayed and directly compared with actual monthly prices from 2019. "
        "Error metrics are provided for the overlapping months. Residuals over the training period are shown for model diagnostics. "
        "Interpretation: a low MAPE (closer to zero) together with small residual variance indicates the model captured the principal trend. "
        "Where deviations occur, these reflect unmodeled shocks or volatility in the market."
    )
    st.write(obs1)

# ------------------------------------------------------------------------
# PROJECT 2: 2021–2025 train, forecast 2026; backtest on 2025 for comparison
# ------------------------------------------------------------------------
else:
    st.header("Project 2: Training 2021–2025, Forecast 2026 (Backtest on 2025 for comparison)")

    # Training as requested is 2021-2025, but to obtain an actual comparison we will backtest:
    train_period_start = "2021-01-01"
    train_period_end = "2025-12-31"
    backtest_train_end = "2024-12-31"  # train for backtest
    backtest_test_start = "2025-01-01"
    backtest_test_end = "2025-12-31"

    # Prepare series
    train_full_p2 = series_full.loc[train_period_start:train_period_end]
    train_full_p2 = safe_series(train_full_p2)

    # Show Graph 1: monthly price movement (2021-2025)
    st.subheader("Graph 1: Monthly Price Movement (2021–2025)")
    fig_p2_g1 = go.Figure()
    fig_p2_g1.add_trace(go.Scatter(x=train_full_p2.index, y=train_full_p2.values, mode="lines", name="Historical (2021-2025)"))
    fig_p2_g1.update_layout(xaxis_title="Date", yaxis_title="Price", height=350)
    st.plotly_chart(fig_p2_g1, use_container_width=True)

    # Fit model on 2021-2025 for final forecast
    with st.spinner("Fitting ARIMA model on full 2021–2025 for final forecast"):
        fitted_p2_full = fit_arima(train_full_p2, order=order)

    # Forecast 12 months for 2026 (out-of-sample)
    forecast_start_p2 = "2026-01-01"
    fcast_p2 = make_forecast(fitted_p2_full, forecast_start_p2, 12)

    # Graph 2: ARIMA Forecast Overlaid on Actual (overlay historical + forecast)
    st.subheader("Graph 2: ARIMA Forecast Overlaid on Actual")
    fig_p2_g2 = go.Figure()
    fig_p2_g2.add_trace(go.Scatter(x=train_full_p2.index, y=train_full_p2.values, mode="lines", name="Historical (2021-2025)"))
    fig_p2_g2.add_trace(go.Scatter(x=fcast_p2.index, y=fcast_p2.values, mode="lines", name="Forecast (2026)"))
    # overlay available actual 2026 if any (unlikely)
    actual_2026 = series_full.loc["2026-01-01":"2026-12-31"]
    if len(actual_2026) > 0:
        actual_2026 = safe_series(actual_2026)
        fig_p2_g2.add_trace(go.Scatter(x=actual_2026.index, y=actual_2026.values, mode="lines", name="Actual (2026)"))
    fig_p2_g2.update_layout(xaxis_title="Date", yaxis_title="Price", height=400)
    st.plotly_chart(fig_p2_g2, use_container_width=True)

    # Graph 3: Forecasted Prices (2026)
    st.subheader("Graph 3: Forecasted Prices (2026)")
    fig_p2_g3 = go.Figure()
    fig_p2_g3.add_trace(go.Scatter(x=fcast_p2.index, y=fcast_p2.values, mode="lines", name="Forecast 2026"))
    fig_p2_g3.update_layout(xaxis_title="Date", yaxis_title="Price", height=350)
    st.plotly_chart(fig_p2_g3, use_container_width=True)

    # Backtest: train on 2021-2024, test on 2025 to produce a real forecast vs actual comparison
    st.subheader("Forecast vs Actual Comparison via Backtest (Train 2021–2024, Test 2025)")
    backtest_train = series_full.loc[train_period_start:backtest_train_end]
    backtest_test = series_full.loc[backtest_test_start:backtest_test_end]
    backtest_train = safe_series(backtest_train)
    backtest_test = safe_series(backtest_test)

    # Fit on backtest training data and forecast length of test
    fitted_backtest = fit_arima(backtest_train, order=order)
    fcast_backtest = make_forecast(fitted_backtest, backtest_test.index[0], len(backtest_test))

    # Align and present comparison
    min_len2 = min(len(backtest_test), len(fcast_backtest))
    comp2 = pd.DataFrame({
        "Actual": backtest_test.values[:min_len2],
        "Forecast": fcast_backtest.values[:min_len2]
    }, index=backtest_test.index[:min_len2])
    st.dataframe(comp2.style.format("{:.2f}"))

    # Metrics for backtest
    mae2, rmse2, mape2 = metrics(comp2["Actual"].values, comp2["Forecast"].values)
    col1b, col2b, col3b = st.columns(3)
    col1b.metric("MAE (backtest 2025)", f"{mae2:.3f}")
    col2b.metric("RMSE (backtest 2025)", f"{rmse2:.3f}")
    col3b.metric("MAPE (backtest 2025)", f"{mape2:.2f} %")

    # Residual diagnostics for full-fit model (2021-2025)
    st.subheader("Residual Diagnostics (Full fit on 2021–2025)")
    resid_p2 = fitted_p2_full.resid
    fig_res_p2 = px.line(x=resid_p2.index, y=resid_p2.values, labels={"x":"Date","y":"Residual"}, title="Residuals (Full fit)")
    st.plotly_chart(fig_res_p2, use_container_width=True)

    # Statistical summary for 2021-2025
    st.subheader("Statistical Summary (2021–2025)")
    st.table(train_full_p2.describe().to_frame(name="Value"))

    # Downloads and observation
    st.subheader("Downloads")
    st.download_button("Download Training Series CSV (2021-2025)", df_to_csv_bytes(train_full_p2.to_frame(name="Close")), file_name="project2_train_2021_2025.csv")
    st.download_button("Download Forecast 2026 CSV", df_to_csv_bytes(fcast_p2.to_frame(name="Forecast")), file_name="project2_forecast_2026.csv")
    st.download_button("Download Backtest Comparison 2025 CSV", df_to_csv_bytes(comp2), file_name="project2_backtest_comparison_2025.csv")

    st.subheader("Observation")
    obs2 = (
        "The primary output is an out-of-sample monthly forecast for 2026 generated from the model fitted on the full 2021–2025 series. "
        "Since actual 2026 data is not available at present, a backtest has been performed: the model was fitted on 2021–2024 and evaluated on 2025 monthly actuals. "
        "The backtest metrics provide an empirical measure of the model's likely performance for the unseen 2026 months. "
        "Interpretation: small error metrics on the backtest indicate the ARIMA approach captures monthly dynamics well within the given horizon; larger errors highlight periods of volatility not captured by a simple ARIMA."
    )
    st.write(obs2)
