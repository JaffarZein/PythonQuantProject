import streamlit as st
import pandas as pd

from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.quant_a.strategy import sma_crossover
from quant_project.quant_a.backtest import run_backtest
from quant_project.quant_a.metrics import compute_metrics


# ============================================================
# Helpers
# ============================================================

def build_trades_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean trades table from trade_id.
    Works ONLY if trade_id exists (Quant A).
    """
    if "trade_id" not in df.columns:
        return pd.DataFrame()

    trades = []

    for trade_id, g in df.dropna(subset=["trade_id"]).groupby("trade_id"):
        entry_idx = g.index[0]
        exit_idx = g.index[-1]

        direction = "Long" if g["position"].iloc[0] > 0 else "Short"

        pnl_pct = g["strategy_returns"].sum() * 100
        pnl_eur = g["equity"].iloc[-1] - g["equity"].iloc[0]

        trades.append({
            "Entry index": entry_idx,
            "Exit index": exit_idx,
            "Direction": direction,
            "PnL %": round(pnl_pct, 2),
            "PnL €": round(pnl_eur, 2),
        })

    return pd.DataFrame(trades)


def compute_drawdown(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax()
    return equity / cummax - 1


# ============================================================
# Quant A
# ============================================================

def run_quant_a():
    st.header("Quant A — Backtesting Engine")

    # ---------- Sidebar parameters ----------
    symbol = st.sidebar.text_input("Symbol", "AAPL")
    start_date = st.sidebar.text_input("Start date", "2020-01-01")
    end_date = st.sidebar.text_input("End date", "2023-01-01")

    fast = st.sidebar.slider("Fast SMA", 5, 100, 20)
    slow = st.sidebar.slider("Slow SMA", 10, 300, 50)

    initial_capital = st.sidebar.number_input(
        "Initial capital",
        min_value=1_000,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
    )

    run_button = st.sidebar.button("Run backtest")

    if not run_button:
        st.info("Set parameters and click **Run backtest**.")
        return

    # ---------- Data ----------
    df = fetch_ohlcv(
        symbol=symbol,
        start=start_date,
        end=end_date,
        interval="1d",
    )

    # ---------- Strategy ----------
    signal = sma_crossover(df, fast=fast, slow=slow)

    results = run_backtest(
        df=df,
        signal=signal,
        initial_capital=initial_capital,
    )

    # ---------- Buy & Hold ----------
    bh_df = df.copy()
    bh_df["bh_returns"] = bh_df["close"].pct_change().fillna(0.0)
    bh_df["bh_equity"] = initial_capital * (1 + bh_df["bh_returns"]).cumprod()

    # ---------- Metrics ----------
    strat_metrics = compute_metrics(results)

    bh_metrics = compute_metrics(
        bh_df.assign(
            strategy_returns=bh_df["bh_returns"],
            equity=bh_df["bh_equity"],
            position=1,
        )
    )

    # ============================================================
    # Display
    # ============================================================

    st.subheader("Performance metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strategy")
        for k, v in strat_metrics.items():
            st.metric(k, f"{v:.2%}" if abs(v) < 2 else f"{v:.2f}")

    with col2:
        st.markdown("### Buy & Hold")
        for k, v in bh_metrics.items():
            st.metric(k, f"{v:.2%}" if abs(v) < 2 else f"{v:.2f}")

    # ---------- Equity Curve ----------
    st.subheader("Equity Curve")

    equity_df = pd.DataFrame({
        "Strategy": results["equity"],
        "Buy & Hold": bh_df["bh_equity"],
    })

    st.line_chart(equity_df)

    # ---------- Drawdown ----------
    st.subheader("Drawdown")

    dd_df = pd.DataFrame({
        "Strategy": compute_drawdown(results["equity"]),
        "Buy & Hold": compute_drawdown(bh_df["bh_equity"]),
    })

    st.line_chart(dd_df)

    # ---------- Price & MAs ----------
    st.subheader("Price & Moving Averages")

    price_df = pd.DataFrame({
        "Close": df["close"],
        "SMA Fast": df["close"].rolling(fast).mean(),
        "SMA Slow": df["close"].rolling(slow).mean(),
    })

    st.line_chart(price_df)

    # ---------- Trades ----------
    st.subheader("Trades")

    trades_df = build_trades_table(results)

    if trades_df.empty:
        st.info("No trades generated.")
    else:
        st.dataframe(trades_df, use_container_width=True)

        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download trades as CSV",
            csv,
            file_name="trades.csv",
            mime="text/csv",
        )


# ============================================================
# Quant B (placeholder)
# ============================================================

def run_quant_b():
    st.header("Quant B — Advanced Analysis")
    st.info("Workspace reserved for Quant B. To be implemented.")


# ============================================================
# App
# ============================================================

def main():
    st.set_page_config(layout="wide")
    st.title("Quant Project – Backtesting App")

    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Section", ["Quant A", "Quant B"])

    st.sidebar.header("Strategy parameters")

    if section == "Quant A":
        run_quant_a()
    else:
        run_quant_b()


if __name__ == "__main__":
    main()
