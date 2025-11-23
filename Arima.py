# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as ts_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Asian Paints ARIMA Forecasting Dashboard", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def fetch_monthly_series(ticker="ASIANPAINT.NS", period="25y"):
    df = yf.download(ticker, interval="1mo", period=period, progress=False)
    if df is None or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()

def ensure_1d_series(x):
    # Accept Series, DataFrame column, numpy arr
    if isinstance(x, pd.DataFrame):
        if "price" in x.columns:
            s = x["price"].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    # If there is a datetime-like index, normalize to month start
    try:
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    except Exception:
        pass
    s.name = "price"
    return s

def fit_arima_and_forecast(train_series, order=(1,1,1), steps=12, forecast_start=None):
    s = ensure_1d_series(train_series)
    if len(s) < 6:
        raise ValueError("Not enough monthly points to fit ARIMA (need >=6).")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    fc_res = fitted.get_forecast(steps=steps)
    fc_mean = pd.Series(np.array(fc_res.predicted_mean).flatten())
    ci = fc_res.conf_int(alpha=0.05)
    lower = pd.Series(np.array(ci.iloc[:,0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:,1]).flatten())

    # build forecast index (monthly starts) — start at given forecast_start or next month after train end
    if forecast_start is None:
        idx_start = s.index[-1] + pd.offsets.MonthBegin()
    else:
        idx_start = pd.to_datetime(forecast_start)
    fc_index = pd.date_range(start=idx_start, periods=steps, freq="MS")
    fc_mean.index = fc_index
    lower.index = fc_index
    upper.index = fc_index
    fc_mean.name = "forecast"
    lower.name = "lower"
    upper.name = "upper"
    return fitted, fc_mean, lower, upper

def plot_series(title, series, y_label="Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Value", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, template="simple_white", height=420)
    return fig

def plot_forecast_overlay(history, forecast, actual=None, lower=None, upper=None, title="Actual vs Forecast"):
    fig = go.Figure()
    if history is not None and len(history)>0:
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="History", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(width=2, dash="dash")))
    if lower is not None and upper is not None and len(lower)==len(upper)==len(forecast):
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name="95% CI", line=dict(width=0)))
    if actual is not None and len(actual)>0:
        # plot only overlapping portion
        min_len = min(len(actual), len(forecast))
        fig.add_trace(go.Scatter(x=actual.index[:min_len], y=actual.values[:min_len], mode="lines", name="Actual", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="simple_white", height=440)
    return fig

def metrics_table(actual, forecast):
    a = ensure_1d_series(actual)
    f = ensure_1d_series(forecast)
    n = min(len(a), len(f))
    if n==0:
        return (np.nan, np.nan, np.nan)
    a_n = a.values[:n].astype(float)
    f_n = f.values[:n].astype(float)
    mae = mean_absolute_error(a_n, f_n)
    rmse = np.sqrt(mean_squared_error(a_n, f_n))
    mape = np.mean(np.abs((a_n - f_n) / a_n)) * 100
    return mae, rmse, mape

def df_to_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

def compute_acf_plot(series, nlags=24):
    s = ensure_1d_series(series)
    acf_vals = ts_acf(s, nlags=nlags, fft=False, missing='conservative')
    idx = np.arange(len(acf_vals))
    fig = go.Figure([go.Bar(x=idx, y=acf_vals)])
    fig.update_layout(title="Autocorrelation (ACF)", xaxis_title="Lag", yaxis_title="ACF", template="simple_white", height=320)
    return fig

# -------------------------
# Main UI
# -------------------------
st.title("Asian Paints ARIMA Forecasting Dashboard")
st.write("Two independent projects — ARIMA only. Monthly data from Yahoo Finance. Professional charts, diagnostics, downloads and observations.")

# Sidebar controls
st.sidebar.header("Controls")
project_choice = st.sidebar.selectbox("Select Project", ["Project 1 (2010–2018 → 2019)", "Project 2 (2021–2025 → 2026)"])
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p,d,q)
forecast_months = int(st.sidebar.slider("Forecast horizon (months)", min_value=6, max_value=24, value=12))

# Load series once and check
with st.spinner("Fetching monthly prices from Yahoo Finance..."):
    series_full = fetch_monthly_series("ASIANPAINT.NS", period="25y")
if series_full.empty:
    st.error("Could not fetch monthly series for ASIANPAINT.NS. Check connectivity/ticker and retry.")
    st.stop()

# PROJECT 1
if project_choice.startswith("Project 1"):
    st.header("Project 1 — Train: 2010–2018 | Forecast: 2019")
    train = series_full.loc["2010-01-01":"2018-12-31"]
    actual_2019 = series_full.loc["2019-01-01":"2019-12-31"]

    # Graph 1: monthly train movement
    st.subheader("1) Monthly Price Movement (2010–2018)")
    st.plotly_chart(plot_series("Asian Paints 2010–2018", train), use_container_width=True)

    # Fit model and forecast for 2019 months (or horizon specified)
    try:
        fitted1, fc_mean1, lower1, upper1 = fit_arima_and_forecast(train, order=order, steps=forecast_months, forecast_start="2019-01-01")
    except Exception as e:
        st.error(f"Model fit failed: {e}")
        st.stop()

    # Graph 2: overlay forecast with history and actual if available
    st.subheader("2) ARIMA Forecast Overlaid on Actual")
    st.plotly_chart(plot_forecast_overlay(train, fc_mean1, actual=actual_2019, lower=lower1, upper=upper1, title="History + Forecast (2019)"), use_container_width=True)

    # Graph 3: forecast only
    st.subheader("3) Forecast Only (2019)")
    st.plotly_chart(plot_series("Forecast (2019)", fc_mean1), use_container_width=True)

    # Compare forecast vs actual for overlapping months
    st.subheader("Forecast vs Actual Comparison (2019)")
    actual = ensure_1d_series(actual_2019)
    forecast_for_compare = fc_mean1
    min_len = min(len(actual), len(forecast_for_compare))
    if min_len > 0:
        comp_df = pd.DataFrame({"Actual": actual.values[:min_len], "Forecast": forecast_for_compare.values[:min_len]}, index=actual.index[:min_len])
        st.dataframe(comp_df.style.format("{:.2f}"))
        mae, rmse, mape = metrics_table(actual[:min_len], forecast_for_compare[:min_len])
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")
        col3.metric("MAPE", f"{mape:.2f}%")
    else:
        # Show forecast table anyway so presentation doesn't look incomplete
        st.info("No actual monthly values found for 2019 in the fetched data. Forecast values are shown below.")
        st.dataframe(fc_mean1.to_frame("Forecast").style.format("{:.2f}"))

    # Residual diagnostics
    st.subheader("Residuals (training)")
    residuals1 = fitted1.resid
    st.plotly_chart(plot_series("Residuals (Training)", residuals1), use_container_width=True)

    st.subheader("ACF (training)")
    st.plotly_chart(compute_acf_plot(train, nlags=24), use_container_width=True)

    st.subheader("Statistical Summary (Training)")
    st.table(train.describe().to_frame("value"))

    # Downloads & observation
    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", data=df_to_bytes(train.to_frame("price")), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast (CSV)", data=df_to_bytes(fc_mean1.to_frame("forecast")), file_name="project1_forecast_2019.csv")
    if min_len>0:
        st.download_button("Download comparison (CSV)", data=df_to_bytes(comp_df), file_name="project1_comparison_2019.csv")

    st.subheader("Observation")
    st.write(
        "ARIMA model trained on monthly closing prices (2010–2018) and used to forecast 2019. "
        "When monthly actuals for 2019 are available they are compared with the forecast and metrics computed. "
        "Residuals and ACF provide model diagnostics; small residual variance and low ACF indicate a reasonable fit. "
        "Deviations between forecast and actual reflect market shocks or events not present in historical data."
    )

# PROJECT 2
else:
    st.header("Project 2 — Train: 2021–2025 | Forecast: 2026 (Backtest included)")
    train_p2 = series_full.loc["2021-01-01":"2025-12-31"]

    st.subheader("1) Monthly Price Movement (2021–2025)")
    st.plotly_chart(plot_series("Asian Paints 2021–2025", train_p2), use_container_width=True)

    # Fit on full requested period and forecast out-of-sample
    try:
        fitted_p2, fc_mean_p2, lower_p2, upper_p2 = fit_arima_and_forecast(train_p2, order=order, steps=forecast_months, forecast_start="2026-01-01")
    except Exception as e:
        st.error(f"Model fit failed: {e}")
        st.stop()

    st.subheader("2) ARIMA Forecast Overlaid on Actual (if available)")
    # overlay training and forecast; actual 2026 months may or may not exist
    actual_2026 = series_full.loc["2026-01-01":"2026-12-31"]
    st.plotly_chart(plot_forecast_overlay(train_p2, fc_mean_p2, actual=actual_2026, lower=lower_p2, upper=upper_p2, title="Training 2021–2025 + Forecast 2026"), use_container_width=True)

    st.subheader("3) Forecast Only (2026)")
    st.plotly_chart(plot_series("Forecast 2026", fc_mean_p2), use_container_width=True)

    # Backtest: train on 2021-2024 and test on 2025 for real comparison
    st.subheader("Forecast vs Actual (Backtest): Train 2021–2024, Test 2025")
    back_train = series_full.loc["2021-01-01":"2024-12-31"]
    back_test = series_full.loc["2025-01-01":"2025-12-31"]
    if len(back_test) > 0 and len(back_train) >= 6:
        fitted_bt, mean_bt, lb_bt, ub_bt = fit_arima_and_forecast(back_train, order=order, steps=len(back_test), forecast_start=back_test.index[0])
        minlen = min(len(back_test), len(mean_bt))
        comp_bt = pd.DataFrame({"Actual": back_test.values[:minlen], "Forecast": mean_bt.values[:minlen]}, index=back_test.index[:minlen])
        st.dataframe(comp_bt.style.format("{:.2f}"))
        mae_bt, rmse_bt, mape_bt = metrics_table(comp_bt["Actual"], comp_bt["Forecast"])
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (backtest)", f"{mae_bt:.3f}")
        c2.metric("RMSE (backtest)", f"{rmse_bt:.3f}")
        c3.metric("MAPE (backtest)", f"{mape_bt:.2f}%")
    else:
        st.info("Not enough 2025 monthly data available for backtest comparison. Forecast is still shown for presentation.")

    st.subheader("Residuals (full fit 2021–2025)")
    st.plotly_chart(plot_series("Residuals (Full Fit)", fitted_p2.resid), use_container_width=True)

    st.subheader("ACF (2021–2025)")
    st.plotly_chart(compute_acf_plot(train_p2, nlags=24), use_container_width=True)

    st.subheader("Statistical Summary (2021–2025)")
    st.table(train_p2.describe().to_frame("value"))

    st.subheader("Downloads")
    st.download_button("Download training series (CSV)", data=df_to_bytes(train_p2.to_frame("price")), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast 2026 (CSV)", data=df_to_bytes(fc_mean_p2.to_frame("forecast")), file_name="project2_forecast_2026.csv")
    if 'comp_bt' in locals():
        st.download_button("Download backtest comparison CSV", data=df_to_bytes(comp_bt), file_name="project2_backtest_comparison_2025.csv")

    st.subheader("Observation")
    st.write(
        "Model trained on monthly closing prices 2021–2025 and used to forecast 2026 months. "
        "Because full 2026 actuals are typically not available yet, a backtest (train on 2021–2024, test on 2025) is provided to quantify expected performance. "
        "Use the backtest metrics to judge likely forecast reliability for 2026."
    )
