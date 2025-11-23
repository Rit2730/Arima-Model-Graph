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

st.set_page_config(page_title="Asian Paints ARIMA Forecasting", layout="wide")

# -------------------------
# Utilities (defensive)
# -------------------------
def fetch_monthly_series(ticker="ASIANPAINT.NS", period="25y"):
    df = yf.download(ticker, interval="1mo", period=period, progress=False)
    if df is None or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().copy()
    # normalize to month start timestamp to avoid weird tick labels
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()

def ensure_1d_series(x):
    if isinstance(x, pd.DataFrame):
        # pick 'price' if present else first column
        if "price" in x.columns:
            s = x["price"].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    except Exception:
        pass
    s.name = "price"
    return s

def fit_arima_and_forecast(train_series, order=(1,1,1), steps=12, forecast_start=None):
    s = ensure_1d_series(train_series)
    if len(s) < 6:
        raise ValueError("Not enough monthly points to fit ARIMA (>=6 required).")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(res.predicted_mean).flatten())
    ci = res.conf_int()
    lower = pd.Series(np.array(ci.iloc[:,0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:,1]).flatten())
    # build forecast index reliably
    if forecast_start is None:
        idx_start = s.index[-1] + pd.offsets.MonthBegin()
    else:
        idx_start = pd.to_datetime(forecast_start)
    fc_index = pd.date_range(start=idx_start, periods=steps, freq="MS")
    mean.index = fc_index
    lower.index = fc_index
    upper.index = fc_index
    mean.name = "forecast"
    lower.name = "lower"
    upper.name = "upper"
    return fitted, mean, lower, upper

def plot_line_series(series, title, y_label="Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Value", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, template="simple_white", height=420)
    return fig

def plot_forecast_overlay(history, forecast, actual=None, lower=None, upper=None, title="Actual vs Forecast"):
    fig = go.Figure()
    if history is not None and len(history)>0:
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="History", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(width=2,dash="dash")))
    if (lower is not None) and (upper is not None) and len(lower)==len(upper)==len(forecast):
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name="95% CI", line=dict(width=0)))
    if actual is not None and len(actual)>0:
        # show only overlapping months to avoid index mismatch
        min_len = min(len(actual), len(forecast))
        fig.add_trace(go.Scatter(x=actual.index[:min_len], y=actual.values[:min_len], mode="lines", name="Actual", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="simple_white", height=440)
    return fig

def compute_metrics(actual, forecast):
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

def df_to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

def acf_plot(series, nlags=24):
    s = ensure_1d_series(series)
    vals = ts_acf(s, nlags=nlags, fft=False, missing='conservative')
    fig = go.Figure([go.Bar(x=np.arange(len(vals)), y=vals)])
    fig.update_layout(title="Autocorrelation (ACF)", xaxis_title="Lag", yaxis_title="ACF", template="simple_white", height=320)
    return fig

# -------------------------
# UI
# -------------------------
st.title("Asian Paints ARIMA Forecasting Dashboard")
st.write("Two projects: Project 1 (2010–2018 → 2019) and Project 2 (2021–2025 → 2026). ARIMA only. Professional charts, diagnostics, downloads and observations.")

st.sidebar.header("Controls")
project_choice = st.sidebar.selectbox("Project", ["Project 1", "Project 2"])
p = int(st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1))
d = int(st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1))
q = int(st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1))
order = (p,d,q)
forecast_months = int(st.sidebar.slider("Forecast months", 6, 24, 12))

# load base series
with st.spinner("Fetching monthly series..."):
    series_full = fetch_monthly_series("ASIANPAINT.NS", period="25y")
if series_full.empty:
    st.error("Cannot fetch data from Yahoo Finance. Please check network/ticker.")
    st.stop()

# Project 1
if project_choice == "Project 1":
    st.header("Project 1 — Training: 2010–2018 | Forecast: 2019")
    train = series_full.loc["2010-01-01":"2018-12-31"]
    actual_2019 = series_full.loc["2019-01-01":"2019-12-31"]

    st.subheader("1. Monthly Price Movement (2010–2018)")
    st.plotly_chart(plot_line_series(train, "Asian Paints 2010–2018"), use_container_width=True)

    # fit & forecast for requested months starting Jan 2019
    try:
        fitted1, mean1, lower1, upper1 = fit_arima_and_forecast(train, order=order, steps=forecast_months, forecast_start="2019-01-01")
    except Exception as e:
        st.error(f"Model fit failed: {e}")
        st.stop()

    st.subheader("2. ARIMA Forecast Overlaid on Actual")
    st.plotly_chart(plot_forecast_overlay(train, mean1, actual=actual_2019, lower=lower1, upper=upper1, title="History + Forecast (2019)"), use_container_width=True)

    st.subheader("3. Forecast Only (2019)")
    st.plotly_chart(plot_line_series(mean1, "Forecasted monthly prices (2019)"), use_container_width=True)

    # comparison table (overlap)
    st.subheader("Forecast vs Actual Comparison (2019)")
    actual = ensure_1d_series(actual_2019)
    min_len = min(len(actual), len(mean1))
    if min_len > 0:
        comp_df = pd.DataFrame({"Actual": actual.values[:min_len], "Forecast": mean1.values[:min_len]}, index=actual.index[:min_len])
        st.dataframe(comp_df.style.format("{:.2f}"))
        mae, rmse, mape = compute_metrics(actual[:min_len], mean1[:min_len])
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("MAPE", f"{mape:.2f}%")
    else:
        st.info("No actual monthly values for 2019 found. Forecast table displayed for presentation.")
        st.dataframe(mean1.to_frame("Forecast").style.format("{:.2f}"))

    # residuals and diagnostics
    st.subheader("Residuals (training 2010–2018)")
    st.plotly_chart(plot_line_series(fitted1.resid, "Residuals (Training)"), use_container_width=True)

    st.subheader("ACF (training)")
    st.plotly_chart(acf_plot(train, nlags=24), use_container_width=True)

    st.subheader("Statistical Summary (training)")
    # robust describe->table conversion
    desc = train.describe()
    desc_df = pd.DataFrame({"value": desc})
    st.table(desc_df)

    st.subheader("Downloads and Observation")
    st.download_button("Download training CSV", data=df_to_csv_bytes(train.to_frame("price")), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast CSV", data=df_to_csv_bytes(mean1.to_frame("forecast")), file_name="project1_forecast_2019.csv")
    if min_len>0:
        st.download_button("Download comparison CSV", data=df_to_csv_bytes(comp_df), file_name="project1_comparison_2019.csv")

    st.write(
        "Observation: ARIMA was trained on monthly prices 2010–2018. Forecast for 2019 is shown and compared to actuals where available. Residuals and ACF are provided for diagnostics."
    )

# Project 2
else:
    st.header("Project 2 — Training: 2021–2025 | Forecast: 2026 (Backtest included)")
    train_p2 = series_full.loc["2021-01-01":"2025-12-31"]

    st.subheader("1. Monthly Price Movement (2021–2025)")
    st.plotly_chart(plot_line_series(train_p2, "Asian Paints 2021–2025"), use_container_width=True)

    try:
        fitted_p2, mean_p2, lower_p2, upper_p2 = fit_arima_and_forecast(train_p2, order=order, steps=forecast_months, forecast_start="2026-01-01")
    except Exception as e:
        st.error(f"Model fit failed: {e}")
        st.stop()

    st.subheader("2. ARIMA Forecast Overlaid on Actual (if present)")
    actual_2026 = series_full.loc["2026-01-01":"2026-12-31"]
    st.plotly_chart(plot_forecast_overlay(train_p2, mean_p2, actual=actual_2026, lower=lower_p2, upper=upper_p2, title="Training + Forecast (2026)"), use_container_width=True)

    st.subheader("3. Forecast Only (2026)")
    st.plotly_chart(plot_line_series(mean_p2, "Forecasted Monthly Prices (2026)"), use_container_width=True)

    # Backtest: train 2021-2024, test 2025
    st.subheader("Forecast vs Actual (Backtest): Train 2021–2024, Test 2025")
    back_train = series_full.loc["2021-01-01":"2024-12-31"]
    back_test = series_full.loc["2025-01-01":"2025-12-31"]
    if len(back_train) >= 6 and len(back_test)>0:
        fitted_bt, mean_bt, lb_bt, ub_bt = fit_arima_and_forecast(back_train, order=order, steps=len(back_test), forecast_start=back_test.index[0])
        minlen = min(len(back_test), len(mean_bt))
        comp_bt = pd.DataFrame({"Actual": back_test.values[:minlen], "Forecast": mean_bt.values[:minlen]}, index=back_test.index[:minlen])
        st.dataframe(comp_bt.style.format("{:.2f}"))
        mae_bt, rmse_bt, mape_bt = compute_metrics(comp_bt["Actual"], comp_bt["Forecast"])
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (backtest)", f"{mae_bt:.3f}")
        c2.metric("RMSE (backtest)", f"{rmse_bt:.3f}")
        c3.metric("MAPE (backtest)", f"{mape_bt:.2f}%")
    else:
        st.info("Not enough 2025 monthly data for backtest — forecast is still shown for presentation.")

    st.subheader("Residuals (full fit 2021–2025)")
    st.plotly_chart(plot_line_series(fitted_p2.resid, "Residuals (Full Fit)"), use_container_width=True)

    st.subheader("ACF (2021–2025)")
    st.plotly_chart(acf_plot(train_p2, nlags=24), use_container_width=True)

    st.subheader("Statistical Summary (2021–2025)")
    desc2 = train_p2.describe()
    desc2_df = pd.DataFrame({"value": desc2})
    st.table(desc2_df)

    st.subheader("Downloads and Observation")
    st.download_button("Download training CSV", data=df_to_csv_bytes(train_p2.to_frame("price")), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast CSV", data=df_to_csv_bytes(mean_p2.to_frame("forecast")), file_name="project2_forecast_2026.csv")
    if 'comp_bt' in locals():
        st.download_button("Download backtest comparison CSV", data=df_to_csv_bytes(comp_bt), file_name="project2_backtest_comparison_2025.csv")

    st.write(
        "Observation: Model trained on 2021–2025 produces a 2026 forecast. Because 2026 actuals may not be present, a backtest (train 2021–2024, test 2025) quantifies recent performance."
    )
