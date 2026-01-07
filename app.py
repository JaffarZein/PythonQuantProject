import streamlit as st
import pandas as pd

from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.strategy import sma_crossover
from quant_project.backtest import run_backtest
from quant_project.metrics import compute_metrics, compute_bh_metrics


# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Quant Project – Backtesting App",
    layout="wide",
)

st.title("Quant Project – Backtesting App")


# =============================
# Sidebar – parameters
# =============================
st.sidebar.header("Strategy parameters")

symbol = st.sidebar.text_input("Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2023-01-01"))

fast = st.sidebar.slider("Fast SMA", min_value=5, max_value=100, value=20)
slow = st.sidebar.slider("Slow SMA", min_value=20, max_value=200, value=50)

initial_capital = st.sidebar.number_input(
    "Initial capital",
    min_value=1_000,
    max_value=1_000_000,
    value=10_000,
    step=1_000,
)

strategy_name = st.sidebar.selectbox(
    "Strategy",
    ["SMA Crossover"],  # extensible
)

run_button = st.sidebar.button("Run backtest")


# =============================
# Run backtest
# =============================
if run_button:
    with st.spinner("Running backtest..."):
        # --- Data ---
        df = fetch_ohlcv(
            symbol=symbol,
            start=str(start_date),
            end=str(end_date),
            interval="1d",
        )

        # --- Strategy ---
        if strategy_name == "SMA Crossover":
            signal = sma_crossover(df, fast=fast, slow=slow)

        # --- Backtest ---
        results = run_backtest(
            df=df,
            signal=signal,
            initial_capital=initial_capital,
        )

        # --- Metrics ---
        metrics_strategy = compute_metrics(results)
        metrics_bh = compute_bh_metrics(results)

    # =============================
    # Metrics display
    # =============================
    st.subheader("Performance metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strategy")
        st.metric("Total Return", f"{metrics_strategy['total_return']:.2%}")
        st.metric("Sharpe Ratio", f"{metrics_strategy['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics_strategy['max_drawdown']:.2%}")
        st.metric("Number of trades", metrics_strategy["n_trades"])
        st.metric("Win rate", f"{metrics_strategy['win_rate']:.2%}")
        st.metric("Exposure", f"{metrics_strategy['exposure']:.2%}")

    with col2:
        st.markdown("### Buy & Hold")
        st.metric("Total Return", f"{metrics_bh['total_return']:.2%}")
        st.metric("Sharpe Ratio", f"{metrics_bh['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics_bh['max_drawdown']:.2%}")

    # =============================
    # Equity curves
    # =============================
    st.subheader("Equity Curve")

    equity_df = results.set_index("date")[["equity", "bh_equity"]]
    st.line_chart(equity_df)

    # =============================
    # Price & SMAs
    # =============================
    st.subheader("Price & Moving Averages")

    price_df = results.set_index("date")[["close", "sma_fast", "sma_slow"]]
    st.line_chart(price_df)

    # =============================
    # Download CSV
    # =============================
    st.subheader("Download results")

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name=f"{symbol}_backtest_results.csv",
        mime="text/csv",
    )
