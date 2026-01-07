import streamlit as st
import pandas as pd

from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.strategy import sma_crossover
from quant_project.backtest import run_backtest
from quant_project.metrics import compute_metrics

st.set_page_config(page_title="Quant Project – Backtesting", layout="wide")

st.title("Quant Project – Backtesting App")

# =========================
# Sidebar – Parameters
# =========================
st.sidebar.header("Strategy parameters")

symbol = st.sidebar.text_input("Symbol", "AAPL")
start = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("2023-01-01"))

fast = st.sidebar.slider("Fast SMA", 5, 100, 20)
slow = st.sidebar.slider("Slow SMA", 10, 200, 50)

initial_capital = st.sidebar.number_input(
    "Initial capital",
    min_value=1_000,
    value=10_000,
    step=1_000,
)

run = st.sidebar.button("Run backtest")

# =========================
# Run Backtest
# =========================
if run:
    df = fetch_ohlcv(
        symbol=symbol,
        start=str(start),
        end=str(end),
        interval="1d",
    )

    signal = sma_crossover(df, fast=fast, slow=slow)

    results = run_backtest(
        df=df,
        signal=signal,
        initial_capital=initial_capital,
    )

    metrics = compute_metrics(results)

    # =========================
    # Display metrics
    # =========================
    st.subheader("Performance metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Return", f"{metrics['total_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    # =========================
    # Plots
    # =========================
    st.subheader("Equity Curve")
    st.line_chart(results.set_index("date")["equity"])

    st.subheader("Price & Moving Averages")
    plot_df = results.set_index("date")[["close", "sma_fast", "sma_slow"]]
    st.line_chart(plot_df)
