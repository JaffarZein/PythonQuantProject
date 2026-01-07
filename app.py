import streamlit as st
import pandas as pd
import numpy as np
import io

from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.quant_a.strategy import sma_crossover
from quant_project.quant_a.backtest import run_backtest
from quant_project.quant_a.metrics import compute_metrics


st.set_page_config(page_title="Quant Project – Backtesting App", layout="wide")

# =========================
# Sidebar – Parameters
# =========================
st.sidebar.title("Strategy parameters")

symbol = st.sidebar.text_input("Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2023-01-01"))

fast_sma = st.sidebar.slider("Fast SMA", min_value=5, max_value=100, value=20)
slow_sma = st.sidebar.slider("Slow SMA", min_value=10, max_value=300, value=50)

initial_capital = st.sidebar.number_input(
    "Initial capital", min_value=1_000, max_value=1_000_000, value=10_000, step=1_000
)

strategy_name = st.sidebar.selectbox(
    "Strategy",
    options=["SMA Crossover"],
)

run_button = st.sidebar.button("Run backtest")

# =========================
# Helper functions
# =========================
def compute_drawdown(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax()
    return equity / cummax - 1.0


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades = []

    position = 0
    entry_date = None
    entry_equity = None
    direction = None

    for date, row in df.iterrows():
        if position == 0 and row["position"] != 0:
            position = row["position"]
            entry_date = date
            entry_equity = row["equity"]
            direction = "Long" if position == 1 else "Short"

        elif position != 0 and row["position"] != position:
            exit_date = date
            exit_equity = row["equity"]

            pnl_eur = exit_equity - entry_equity
            pnl_pct = pnl_eur / entry_equity

            trades.append(
                {
                    "Entry date": entry_date,
                    "Exit date": exit_date,
                    "Direction": direction,
                    "PnL %": pnl_pct * 100,
                    "PnL €": pnl_eur,
                }
            )

            position = row["position"]
            entry_date = date if position != 0 else None
            entry_equity = row["equity"] if position != 0 else None
            direction = (
                "Long" if position == 1 else "Short" if position == -1 else None
            )

    return pd.DataFrame(trades)


# =========================
# Main
# =========================
st.title("Quant Project – Backtesting App")

if run_button:
    # -------- Data
    df = fetch_ohlcv(
        symbol=symbol,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
    )

    # -------- Strategy
    signal = sma_crossover(df, fast=fast_sma, slow=slow_sma)

    results = run_backtest(
        df=df,
        signal=signal,
        initial_capital=initial_capital,
    )

    # -------- Metrics
    metrics = compute_metrics(results)

    # Buy & Hold
    bh_equity = initial_capital * (1 + results["returns"]).cumprod()
    bh_df = results.copy()
    bh_df["equity"] = bh_equity
    bh_metrics = compute_metrics(bh_df)

    # -------- Performance metrics display
    st.subheader("Performance metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strategy")
        st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        st.metric("Number of trades", int(metrics["n_trades"]))
        st.metric("Win rate", f"{metrics['win_rate']*100:.2f}%")
        st.metric("Exposure", f"{metrics['exposure']*100:.2f}%")

    with col2:
        st.markdown("### Buy & Hold")
        st.metric("Total Return", f"{bh_metrics['total_return']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{bh_metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{bh_metrics['max_drawdown']*100:.2f}%")

    # -------- Equity curve
    st.subheader("Equity Curve")

    equity_df = pd.DataFrame(
        {
            "Strategy": results["equity"],
            "Buy & Hold": bh_equity,
        }
    )

    st.line_chart(equity_df)

    # -------- Drawdown
    st.subheader("Drawdown")

    dd_df = pd.DataFrame(
        {
            "Strategy": compute_drawdown(results["equity"]),
            "Buy & Hold": compute_drawdown(bh_equity),
        }
    )

    st.line_chart(dd_df)

    # -------- Price & MAs
    st.subheader("Price & Moving Averages")

    price_df = results[["close", "sma_fast", "sma_slow"]]
    st.line_chart(price_df)

    # -------- Trades table
    st.subheader("Trades")

    trades_df = extract_trades(results)

    if trades_df.empty:
        st.info("No trades executed.")
    else:
        st.dataframe(
            trades_df.style.format(
                {
                    "PnL %": "{:.2f}",
                    "PnL €": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    # -------- Download
    st.subheader("Download results")

   

    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="backtest_results.csv",
        mime="text/csv",
    )
