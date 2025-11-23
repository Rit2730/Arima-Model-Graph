import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from io import BytesIO

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Asian Paints — ARIMA Dashboard", layout="wide")

st.title("Asian Paints — ARIMA Forecasting Dashboard")
st.write("Two independent projects. Monthly data used. ARIMA only. Professional plots, diagnostics, metrics and downloads.")

# ----------------------
# Helper utilities
# ----------------------
@st.cache_data
def load_monthly_close(ticker="ASIANPAINT.NS", period="25y"):
    df = yf.download(ticker, interval="1mo", period=period)
    if df is None or "Close" not in df.columns:
        raise RuntimeError("Failed to fetch Close prices for ticker.")
    s = df["Close"].dropna().copy()
    # normalize index to month start timestamps
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()

def ensure_series_1d(x):
    """Return a clean 1-D pandas Series indexed by timestamp."""
    if isinstance(x, pd.DataFrame):
        # if dataframe has a 'price' or single column, select it
        if "price" in x.columns:
            s = x["price"].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    # flatten and ensure numeric
    s = pd.Series(np.array(s).flatten(), index=s.index if hasattr(s, "index") else None)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s

def fit_arima_safe(series, order=(1,1,1)):
    s = ensure_series_1d(series)
    model = ARIMA(s, order=order)
    fitted = model.fit()
    return fitted

def forecast_with_ci(fitted, steps, start_month):
    """Return forecast series, lower and upper CI. start_month is a Timestamp or 'YYYY-MM-01'"""
    # Get forecast result with conf_int
    res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(res.predicted_mean).flatten())
    ci = res.conf_int(alpha=0.05)
    lower = pd.Series(np.array(ci.iloc[:, 0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:, 1]).flatten())

    index = pd.date_range(start=pd.to_datetime(start_month), periods=steps, freq="MS")
    mean.index = index
    lower.index = index
    upper.index = index
    mean.name = "forecast"
    lower.name = "lower"
    upper.name = "upper"
    return mean, lower, upper

def compute_metrics(actual, forecast):
    # align lengths
    actual = ensure_series_1d(actual)
    forecast = ensure_series_1d(forecast)
    n = min(len(actual), len(forecast))
    if n == 0:
        return (np.nan, np.nan, np.nan)
    a = actual.values[:n].astype(float)
    f = forecast.values[:n].astype(float)
    mae = mean_absolute_error(a, f)
    rmse = np.sqrt(mean_squared_error(a, f))
    mape = np.mean(np.abs((a - f) / a)) * 100 if np.any(a != 0) else np.nan
    return mae, rmse, mape

def df_to_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=True)
    buf.seek(0)
    return buf

# ----------------------
# Load base series
# ----------------------
with st.spinner("Loading monthly price series from Yahoo Finance..."):
    series_full = load_monthly_close()  # monthly series of 'price'

# Sidebar: ARIMA order and UI controls
st.sidebar.header("Model & View Options")
p = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1, step=1)
order = (int(p), int(d), int(q))

project_choice = st.sidebar.radio("Select Project", ("Project 1: 2010–2018 → Forecast 2019", "Project 2: 2021–2025 → Forecast 2026 (with backtest)"))

# Common UI function to render interactive overlay chart
def plot_actual_and_forecast(actual_series, forecast_mean, lower=None, upper=None, title="Actual vs Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_series.index, y=actual_series.values, mode="lines", name="Actual", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean.values, mode="lines", name="Forecast", line=dict(width=2, dash="dash")))
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(x=lower.index, y=upper.values, fill='tonexty', fillcolor='rgba(180,180,255,0.2)',
                                  name="95% CI", line=dict(width=0), hoverinfo="skip"))
        # draw lower bound as invisible trace so fill works
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Project 1
# ----------------------
if project_choice.startswith("Project 1"):
    st.header("Project 1 — Train: 2010–2018 | Forecast: 2019")
    train_start = "2010-01-01"
    train_end = "2018-12-31"
    forecast_start = "2019-01-01"
    # Prepare training series
    train_series = series_full.loc[train_start:train_end]
    train_series = ensure_series_1d(train_series)

    # Graph 1: Monthly Price Movement (train)
    st.subheader("1) Monthly Price Movement (2010–2018)")
    fig_a = px.line(x=train_series.index, y=train_series.values, labels={"x":"Date","y":"Price"}, title="Price (2010–2018)")
    st.plotly_chart(fig_a, use_container_width=True)

    # Fit ARIMA safely
    with st.spinner("Fitting ARIMA model for Project 1..."):
        fitted1 = fit_arima_safe(train_series, order=order)

    # Forecast 12 months for 2019 with CI
    fmean1, lower1, upper1 = forecast_with_ci(fitted1, steps=12, start_month=forecast_start)

    # Graph 2: Forecast overlaid on actual (if actual exists)
    st.subheader("2) ARIMA Forecast Overlaid on Actual")
    # overlay train + forecast; add actual 2019 if available
    actual_2019 = series_full.loc["2019-01-01":"2019-12-31"]
    actual_2019 = ensure_series_1d(actual_2019)
    plot_actual_and_forecast(pd.concat([train_series, actual_2019]) if len(actual_2019)>0 else train_series, fmean1, lower1, upper1, title="ARIMA Forecast vs Actual (with forecasted 2019)")

    # Graph 3: Future Forecast (2019)
    st.subheader("3) Future Forecast (2019 — Out of sample)")
    fig_fore = px.line(x=fmean1.index, y=fmean1.values, labels={"x":"Date","y":"Forecast"}, title="Forecasted Monthly Prices (2019)")
    st.plotly_chart(fig_fore, use_container_width=True)

    # Forecast vs Actual Comparison table
    st.subheader("Forecast vs Actual Comparison (2019)")
    # align lengths
    minlen = min(len(actual_2019), len(fmean1))
    if minlen > 0:
        comp_idx = actual_2019.index[:minlen]
        comp_df = pd.DataFrame({
            "Actual": actual_2019.values[:minlen],
            "Forecast": fmean1.values[:minlen]
        }, index=comp_idx)
    else:
        # fallback: show forecast values (still present)
        comp_df = pd.DataFrame({
            "Forecast": fmean1.values
        }, index=fmean1.index)
    st.dataframe(comp_df.style.format("{:.2f}"))

    # Metrics
    if "Actual" in comp_df.columns:
        mae1, rmse1, mape1 = compute_metrics(comp_df["Actual"], comp_df["Forecast"])
        st.subheader("Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae1:.3f}")
        col2.metric("RMSE", f"{rmse1:.3f}")
        col3.metric("MAPE", f"{mape1:.2f} %")
    else:
        st.info("Actual monthly data for 2019 was not found. Backtest metrics are not shown but forecast is available for download and presentation.")

    # Residual diagnostics
    st.subheader("Residual Diagnostics (Training period)")
    resid1 = fitted1.resid
    fig_res, ax = plt.subplots(figsize=(8, 3))
    ax.plot(resid1)
    ax.set_title("Residuals (Train)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    st.pyplot(fig_res)

    # ACF / PACF
    st.subheader("ACF and PACF (Training series)")
    fig_acf, ax_acf = plt.subplots(1, 2, figsize=(12, 3))
    plot_acf(train_series, ax=ax_acf[0], lags=24, zero=False)
    plot_pacf(train_series, ax=ax_acf[1], lags=24, zero=False, method='ywm')
    st.pyplot(fig_acf)

    # Statistical summary
    st.subheader("Statistical Summary (Training data)")
    st.table(train_series.describe().to_frame("value"))

    # Downloads and observation
    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", df_to_bytes(train_series.to_frame()), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast (CSV)", df_to_bytes(fmean1.to_frame(name="forecast")), file_name="project1_forecast_2019.csv")
    if "Actual" in comp_df.columns:
        st.download_button("Download comparison (CSV)", df_to_bytes(comp_df), file_name="project1_comparison_2019.csv")

    st.subheader("Observation")
    st.write(
        "The ARIMA model was trained on monthly closing prices from 2010 to 2018. "
        "The forecast for 2019 is shown and directly compared with actual 2019 monthly prices (when available). "
        "Performance metrics are provided for overlapping months. Residual plots and ACF/PACF plots assist in diagnostic assessment. "
        "Where forecast deviates from actual, this indicates market events or volatility not captured by the stationary model."
    )

# ----------------------
# Project 2
# ----------------------
else:
    st.header("Project 2 — Train: 2021–2025 | Forecast: 2026 (Backtest included)")

    train_p2_start = "2021-01-01"
    train_p2_end = "2025-12-31"
    forecast_p2_start = "2026-01-01"

    train_p2 = series_full.loc[train_p2_start:train_p2_end]
    train_p2 = ensure_series_1d(train_p2)

    st.subheader("1) Monthly Price Movement (2021–2025)")
    fig_p2a = px.line(x=train_p2.index, y=train_p2.values, labels={"x":"Date","y":"Price"}, title="Price (2021–2025)")
    st.plotly_chart(fig_p2a, use_container_width=True)

    # Fit ARIMA on full 2021-2025 for final forecast
    with st.spinner("Fitting ARIMA model on 2021–2025..."):
        fitted_p2 = fit_arima_safe(train_p2, order=order)

    # Forecast out-of-sample 2026
    fmean_p2, lower_p2, upper_p2 = forecast_with_ci(fitted_p2, steps=12, start_month=forecast_p2_start)

    # Graph 2: overlay
    st.subheader("2) ARIMA Forecast Overlaid on Actual")
    actual_2026 = series_full.loc[forecast_p2_start:"2026-12-31"]
    if len(actual_2026) > 0:
        actual_2026 = ensure_series_1d(actual_2026)
    plot_actual_and_forecast(train_p2 if len(actual_2026)==0 else pd.concat([train_p2, actual_2026]), fmean_p2, lower_p2, upper_p2, title="ARIMA Forecast vs Actual (2026 forecast shown)")

    # Graph 3: future forecast
    st.subheader("3) Future Forecast (2026)")
    fig_p2g3 = px.line(x=fmean_p2.index, y=fmean_p2.values, labels={"x":"Date","y":"Forecast"}, title="Forecasted Monthly Prices (2026)")
    st.plotly_chart(fig_p2g3, use_container_width=True)

    # Backtest to obtain real comparison: train on 2021-2024, test on 2025
    st.subheader("Forecast vs Actual Comparison via Backtest (Train 2021–2024, Test 2025)")
    back_train = series_full.loc["2021-01-01":"2024-12-31"]
    back_test = series_full.loc["2025-01-01":"2025-12-31"]
    back_train = ensure_series_1d(back_train)
    back_test = ensure_series_1d(back_test)

    fitted_back = fit_arima_safe(back_train, order=order)
    fmean_back, lower_back, upper_back = forecast_with_ci(fitted_back, steps=len(back_test), start_month=back_test.index[0] if len(back_test)>0 else "2025-01-01")
    # Align lengths
    min_len_b = min(len(back_test), len(fmean_back))
    comp_p2 = pd.DataFrame({
        "Actual": back_test.values[:min_len_b],
        "Forecast": fmean_back.values[:min_len_b]
    }, index=back_test.index[:min_len_b])
    st.dataframe(comp_p2.style.format("{:.2f}"))

    # Metrics
    if len(comp_p2) > 0:
        mae_p2, rmse_p2, mape_p2 = compute_metrics(comp_p2["Actual"], comp_p2["Forecast"])
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (backtest 2025)", f"{mae_p2:.3f}")
        col2.metric("RMSE (backtest 2025)", f"{rmse_p2:.3f}")
        col3.metric("MAPE (backtest 2025)", f"{mape_p2:.2f} %")
    else:
        st.info("Backtest not possible due to missing 2025 monthly data; final forecast for 2026 is still available.")

    # Residual diagnostics (full fit)
    st.subheader("Residual Diagnostics (Full fit 2021–2025)")
    resid_p2 = fitted_p2.resid
    fig_res_p2, axp = plt.subplots(figsize=(9, 3))
    axp.plot(resid_p2)
    axp.set_title("Residuals (Full fit)")
    st.pyplot(fig_res_p2)

    # ACF/PACF
    st.subheader("ACF and PACF (2021–2025)")
    fig_acf2, axes_acf2 = plt.subplots(1, 2, figsize=(12, 3))
    plot_acf(train_p2, ax=axes_acf2[0], lags=24, zero=False)
    plot_pacf(train_p2, ax=axes_acf2[1], lags=24, zero=False, method='ywm')
    st.pyplot(fig_acf2)

    # Statistical summary
    st.subheader("Statistical Summary (2021–2025)")
    st.table(train_p2.describe().to_frame("value"))

    # Downloads and observation
    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", df_to_bytes(train_p2.to_frame()), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast 2026 (CSV)", df_to_bytes(fmean_p2.to_frame(name="forecast")), file_name="project2_forecast_2026.csv")
    if len(comp_p2) > 0:
        st.download_button("Download backtest comparison (CSV)", df_to_bytes(comp_p2), file_name="project2_backtest_comparison_2025.csv")

    st.subheader("Observation")
    st.write(
        "Model fitted on monthly closing prices 2021–2025 produced a 12-month forecast for 2026. "
        "Since 2026 actual monthly prices are not broadly available yet, a backtest (train on 2021–2024 and test on 2025) is provided to quantify performance on a recent year. "
        "Backtest metrics show model accuracy on recent data and provide an indicative expectation for the 2026 forecast. Residual plots and ACF/PACF illustrate any remaining autocorrelation needing further modeling."
    )
