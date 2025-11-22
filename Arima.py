import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Financial Forecasting App",
    layout="wide",
    page_icon="📈"
)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("📌 Dashboard")
project = st.sidebar.radio("Select Project", ["Project 1 – ARIMA Forecast", "Project 2 – Advanced Technical Analysis"])

# -----------------------------
# Helper: Load Data
# -----------------------------
def load_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period)
        df = df.dropna()
        return df
    except Exception:
        return None

# -----------------------------
# PROJECT 1 - ARIMA FORECASTING
# -----------------------------
if project == "Project 1 – ARIMA Forecast":

    st.title("📈 Project 1: ARIMA Time Series Forecasting")
    st.write("Upload a stock ticker and generate forecasting with ARIMA.")

    ticker = st.text_input("Enter Stock Symbol (Example: AAPL, RELIANCE.NS)", "AAPL")

    if st.button("Load Data"):
        df = load_data(ticker)

        if df is None or len(df) < 200:
            st.error("❌ Not enough data. Try a different stock or longer period.")
        else:
            st.success("Data loaded successfully!")

            # Show data table
            with st.expander("🔍 View Data"):
                st.dataframe(df.tail())

            # Actual Price Chart
            st.subheader("📌 Actual Closing Price Chart")
            fig1, ax1 = plt.subplots()
            ax1.plot(df.index, df["Close"])
            st.pyplot(fig1)

            # ARIMA MODEL
            st.subheader("📌 Building ARIMA Model")

            try:
                model = ARIMA(df["Close"], order=(5,1,2))
                model_fit = model.fit()

                forecast_steps = 30
                forecast = model_fit.forecast(steps=forecast_steps)

                # Forecast Chart
                st.subheader("📌 Forecast Chart (Next 30 Days)")
                fig2, ax2 = plt.subplots()
                ax2.plot(df.index, df["Close"], label="Actual")
                ax2.plot(forecast.index, forecast, label="Forecast", linestyle="--")
                ax2.legend()
                st.pyplot(fig2)

                # COMPARISON GRAPH
                st.subheader("📌 Actual vs Forecast Comparison")
                combined = pd.concat([df["Close"].tail(50), forecast])
                fig3, ax3 = plt.subplots()
                ax3.plot(combined.index, combined.values)
                st.pyplot(fig3)

                # Download Forecast Button
                st.download_button("📥 Download Forecast Data", forecast.to_csv(), "forecast.csv")

            except Exception as e:
                st.error("Model Error: " + str(e))


# -----------------------------
# PROJECT 2 - ADVANCED TECHNICAL ANALYSIS
# -----------------------------
else:
    st.title("📊 Project 2: Advanced Technical Analysis")
    st.write("Includes Moving Averages, RSI, MACD, Volatility & Correlation Heatmap.")

    ticker2 = st.text_input("Enter Stock Symbol for Project 2 (Example: TSLA, INFY.NS)", "TSLA")

    if st.button("Load Project 2 Data"):
        df2 = load_data(ticker2, period="3y")

        if df2 is None or len(df2) < 150:
            st.error("❌ Not enough data")
        else:
            st.success("Data Loaded Successfully!")

            # Moving Averages
            df2["MA20"] = df2["Close"].rolling(20).mean()
            df2["MA50"] = df2["Close"].rolling(50).mean()

            # RSI
            delta = df2["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df2["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            df2["EMA12"] = df2["Close"].ewm(span=12).mean()
            df2["EMA26"] = df2["Close"].ewm(span=26).mean()
            df2["MACD"] = df2["EMA12"] - df2["EMA26"]
            df2["Signal"] = df2["MACD"].ewm(span=9).mean()

            # VOLATILITY
            df2["Volatility"] = df2["Close"].pct_change().rolling(20).std()

            # -----------------------------
            # GRAPHS
            # -----------------------------

            # 1. PRICE + MA
            st.subheader("📌 Price with Moving Averages")
            fig4, ax4 = plt.subplots()
            ax4.plot(df2["Close"], label="Close")
            ax4.plot(df2["MA20"], label="MA20")
            ax4.plot(df2["MA50"], label="MA50")
            ax4.legend()
            st.pyplot(fig4)

            # 2. RSI
            st.subheader("📌 RSI Indicator")
            fig5, ax5 = plt.subplots()
            ax5.plot(df2["RSI"])
            ax5.axhline(30, linestyle="--")
            ax5.axhline(70, linestyle="--")
            st.pyplot(fig5)

            # 3. MACD
            st.subheader("📌 MACD Indicator")
            fig6, ax6 = plt.subplots()
            ax6.plot(df2["MACD"], label="MACD")
            ax6.plot(df2["Signal"], label="Signal")
            ax6.legend()
            st.pyplot(fig6)

            # 4. VOLATILITY
            st.subheader("📌 Volatility Chart")
            fig7, ax7 = plt.subplots()
            ax7.plot(df2["Volatility"])
            st.pyplot(fig7)

            # 5. Correlation Heatmap
            st.subheader("📌 Correlation Heatmap")
            fig8, ax8 = plt.subplots()
            sns.heatmap(df2[["Close","MA20","MA50","RSI","MACD","Volatility"]].corr(), annot=True, ax=ax8)
            st.pyplot(fig8)

            st.download_button("📥 Download Technical Data", df2.to_csv(), "technical.csv")
