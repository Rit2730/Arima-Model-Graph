# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Asian Paints — ARIMA Dashboard", layout="wide")

# ---- helpers ----
def safe_series(x):
    """Return a 1-D pandas Series of numeric values indexed by timestamps when possible."""
    if isinstance(x, pd.DataFrame):
        # prefer column named 'price' if exists
        if "price" in x.columns:
            s = x["price"].copy()
        else:
            s = x.iloc[:, 0].copy()
    elif isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    # coerce numbers, drop NaN, normalize index to month start if present
    s = pd.to_numeric(s, errors="coerce").dropna()
    if hasattr(s, "index"):
        try:
            s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
        except Exception:
            pass
    s.name = "price"
    return s

def fetch_yfinance_monthly(ticker="ASIANPAINT.NS", period="25y"):
    """Fetch monthly close prices from yfinance and return 1-D Series."""
    df = yf.download(ticker, interval="1mo", period=period, progress=False)
    if df is None or "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    s.name = "price"
    return s.sort_index()

def fit_arima_and_forecast(series, order=(1,1,1), steps=12, start_period=None):
    s = safe_series(series)
    if len(s) < 6:
        raise ValueError("Not enough data points to fit ARIMA.")
    model = ARIMA(s, order=order)
    fitted = model.fit()
    # forecast
    fc_res = fitted.get_forecast(steps=steps)
    mean = pd.Series(np.array(fc_res.predicted_mean).flatten())
    ci = fc_res.conf_int()
    lower = pd.Series(np.array(ci.iloc[:,0]).flatten())
    upper = pd.Series(np.array(ci.iloc[:,1]).flatten())
    # build index for forecast (monthly starts)
    if start_period is None:
        # start right after last training month
        idx_start = s.index[-1] + pd.offsets.MonthBegin()
    else:
        idx_start = pd.to_datetime(start_period)
    index = pd.date_range(start=idx_start, periods=steps, freq="MS")
    mean.index = index
    lower.index = index
    upper.index = index
    mean.name = "forecast"
    lower.name = "lower"
    upper.name = "upper"
    return fitted, mean, lower, upper

def compute_metrics(actual, forecast):
    a = safe_series(actual)
    f = safe_series(forecast)
    n = min(len(a), len(f))
    if n == 0:
        return (np.nan, np.nan, np.nan)
    a = a.values[:n].astype(float)
    f = f.values[:n].astype(float)
    mae = mean_absolute_error(a, f)
    rmse = np.sqrt(mean_squared_error(a, f))
    mape = np.mean(np.abs((a - f) / a)) * 100 if np.any(a != 0) else np.nan
    return mae, rmse, mape

def df_to_csv_bytes(df):
    from io import BytesIO
    buf = BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return buf

def plot_actual_forecast(actual, forecast, lower=None, upper=None, title="Actual vs Forecast"):
    fig = go.Figure()
    if len(actual) > 0:
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines", name="Actual", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(width=2, dash="dash")))
    if (lower is not None) and (upper is not None) and len(lower)==len(upper)==len(forecast):
        fig.add_trace(go.Scatter(x=upper.index, y=upper.values, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=lower.index, y=lower.values, mode="lines", fill='tonexty', fillcolor='rgba(173,216,230,0.2)', name="95% CI", line=dict(width=0)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="simple_white", height=480)
    st.plotly_chart(fig, use_container_width=True)

# ---- UI ----
st.header("Asian Paints — ARIMA Forecasting Dashboard")
st.write("Two project views: Project 1 (2010–2018 → forecast 2019) and Project 2 (2021–2025 → forecast 2026). Use the sidebar to select and run.")

st.sidebar.header("Options")
project = st.sidebar.selectbox("Project", ["Project 1", "Project 2"])
p = st.sidebar.number_input("ARIMA p", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("ARIMA d", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("ARIMA q", min_value=0, max_value=5, value=1)
order = (int(p), int(d), int(q))
forecast_months = st.sidebar.slider("Forecast months (out of sample)", 1, 24, 12)

# load data once
with st.spinner("Fetching monthly series from Yahoo Finance..."):
    series_full = fetch_yfinance_monthly(ticker="ASIANPAINT.NS", period="25y")
if series_full.empty:
    st.error("Failed to fetch data from Yahoo Finance. Confirm ticker and try again. App will stop.")
    st.stop()

if project == "Project 1":
    st.subheader("Project 1 — Train 2010–2018, Forecast 2019")
    train = series_full.loc["2010-01-01":"2018-12-31"]
    actual_2019 = series_full.loc["2019-01-01":"2019-12-31"]

    st.markdown("### Graph 1 — Monthly price movement (2010–2018)")
    fig = go.Figure(go.Scatter(x=train.index, y=train.values, mode="lines", name="Train"))
    fig.update_layout(template="simple_white", xaxis_title="Date", yaxis_title="Price", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # fit & forecast
    fitted1, mean1, lower1, upper1 = fit_arima_and_forecast(train, order=order, steps=forecast_months, start_period="2019-01-01")

    st.markdown("### Graph 2 — ARIMA forecast overlaid on history and actuals")
    plot_actual_forecast(pd.concat([train, actual_2019]) if len(actual_2019)>0 else train, mean1, lower1, upper1, title="Training + Forecast (2019)")

    st.markdown("### Graph 3 — Forecast only (2019 forecast months)")
    fig_fc = go.Figure(go.Scatter(x=mean1.index, y=mean1.values, mode="lines", name="Forecast"))
    fig_fc.update_layout(template="simple_white", xaxis_title="Date", yaxis_title="Forecasted Price", height=360)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Comparison table & metrics
    st.markdown("### Forecast vs Actual (2019)")
    min_len = min(len(actual_2019), len(mean1))
    if min_len > 0:
        comp_df = pd.DataFrame({"Actual": actual_2019.values[:min_len], "Forecast": mean1.values[:min_len]}, index=actual_2019.index[:min_len])
        st.dataframe(comp_df.style.format("{:.2f}"))
        mae, rmse, mape = compute_metrics(comp_df["Actual"], comp_df["Forecast"])
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("MAPE", f"{mape:.2f}%")
    else:
        st.info("No actual monthly values found for 2019 in the dataset. Forecast table is shown above.")

    # diagnostics
    st.markdown("### Residuals (training)")
    resid = fitted1.resid
    fig_res = go.Figure(go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Residuals"))
    fig_res.update_layout(template="simple_white", height=300)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("### Statistical summary (training)")
    st.table(train.describe().to_frame("value"))

    # downloads and observation
    st.download_button("Download training series (CSV)", data=df_to_csv_bytes(train.to_frame("price")), file_name="project1_train_2010_2018.csv")
    st.download_button("Download forecast (CSV)", data=df_to_csv_bytes(mean1.to_frame("forecast")), file_name="project1_forecast_2019.csv")
    if min_len > 0:
        st.download_button("Download comparison (CSV)", data=df_to_csv_bytes(comp_df), file_name="project1_comparison_2019.csv")

    st.markdown("### Observation")
    st.write(
        "The ARIMA model was trained on monthly closing prices from 2010 through 2018. "
        "The 2019 forecast is shown along with actual 2019 monthly values (when available). Metrics indicate model accuracy over overlapping months. Residuals and statistical summary are provided for diagnostic review."
    )

else:
    st.subheader("Project 2 — Train 2021–2025, Forecast 2026 (backtest provided)")
    train_p2 = series_full.loc["2021-01-01":"2025-12-31"]
    # Fit on full requested period and forecast out-of-sample
    fitted_p2, mean_p2, lower_p2, upper_p2 = fit_arima_and_forecast(train_p2, order=order, steps=forecast_months, start_period="2026-01-01")

    st.markdown("### Graph 1 — Monthly price movement (2021–2025)")
    figp = go.Figure(go.Scatter(x=train_p2.index, y=train_p2.values, mode="lines", name="Train"))
    figp.update_layout(template="simple_white", height=360)
    st.plotly_chart(figp, use_container_width=True)

    st.markdown("### Graph 2 — ARIMA forecast overlaid (2026 forecast shown)")
    # overlay training + forecast
    plot_actual_forecast(train_p2, mean_p2, lower_p2, upper_p2, title="Training 2021–2025 + Forecast 2026")

    st.markdown("### Graph 3 — Forecast only (2026)")
    figp3 = go.Figure(go.Scatter(x=mean_p2.index, y=mean_p2.values, mode="lines", name="Forecast 2026"))
    figp3.update_layout(template="simple_white", height=360)
    st.plotly_chart(figp3, use_container_width=True)

    # Backtest: train on 2021-2024, test on 2025 for real comparison
    st.markdown("### Forecast vs Actual Comparison (Backtest: Train 2021–2024, Test 2025)")
    back_train = series_full.loc["2021-01-01":"2024-12-31"]
    back_test = series_full.loc["2025-01-01":"2025-12-31"]
    fitted_bt, mean_bt, lb_bt, ub_bt = fit_arima_and_forecast(back_train, order=order, steps=len(back_test), start_period=back_test.index[0] if len(back_test)>0 else "2025-01-01")

    minlen2 = min(len(back_test), len(mean_bt))
    if minlen2 > 0:
        comp2 = pd.DataFrame({"Actual": back_test.values[:minlen2], "Forecast": mean_bt.values[:minlen2]}, index=back_test.index[:minlen2])
        st.dataframe(comp2.style.format("{:.2f}"))
        mae2, rmse2, mape2 = compute_metrics(comp2["Actual"], comp2["Forecast"])
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (backtest)", f"{mae2:.3f}")
        col2.metric("RMSE (backtest)", f"{rmse2:.3f}")
        col3.metric("MAPE (backtest)", f"{mape2:.2f}%")
    else:
        st.info("Not enough 2025 monthly data found for backtest comparison, but forecast for 2026 is provided for presentation.")

    st.markdown("### Residuals (full fit 2021–2025)")
    resid_p2 = fitted_p2.resid
    figp_res = go.Figure(go.Scatter(x=resid_p2.index, y=resid_p2.values, mode="lines", name="Residuals"))
    figp_res.update_layout(template="simple_white", height=300)
    st.plotly_chart(figp_res, use_container_width=True)

    st.markdown("### Statistical summary (2021–2025)")
    st.table(train_p2.describe().to_frame("value"))

    st.download_button("Download training series (CSV)", data=df_to_csv_bytes(train_p2.to_frame("price")), file_name="project2_train_2021_2025.csv")
    st.download_button("Download forecast 2026 (CSV)", data=df_to_csv_bytes(mean_p2.to_frame("forecast")), file_name="project2_forecast_2026.csv")
    if minlen2 > 0:
        st.download_button("Download backtest comparison (CSV)", data=df_to_csv_bytes(comp2), file_name="project2_backtest_comparison_2025.csv")

    st.markdown("### Observation")
    st.write(
        "Final forecast for 2026 is shown along with a backtest for 2025 (model trained on 2021–2024). "
        "Backtest metrics provide an empirical assessment of model accuracy and help interpret expected forecast reliability. Residuals and summary stats are included for model diagnostics."
    )
