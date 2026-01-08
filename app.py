import streamlit as st
import pandas as pd

from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.quant_a.strategy import sma_crossover, momentum_strategy, predict_price
from quant_project.quant_a.backtest import run_backtest
from quant_project.quant_a.metrics import compute_metrics


# Initialize session state
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None


# ============================================================
# Helpers
# ============================================================

def build_trades_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build clean trades table from trade_id"""
    if "trade_id" not in df.columns:
        return pd.DataFrame()

    trades = []
    for trade_id, g in df.dropna(subset=["trade_id"]).groupby("trade_id"):
        entry_date = g["date"].iloc[0] if "date" in g.columns else g.index[0]
        exit_date = g["date"].iloc[-1] if "date" in g.columns else g.index[-1]

        direction = "Long" if g["position"].iloc[0] > 0 else "Short"
        pnl_pct = g["strategy_returns"].sum() * 100
        pnl_eur = g["equity"].iloc[-1] - g["equity"].iloc[0]

        trades.append({
            "Entry date": entry_date,
            "Exit date": exit_date,
            "Direction": direction,
            "PnL %": round(pnl_pct, 2),
            "PnL €": round(pnl_eur, 2),
        })
    return pd.DataFrame(trades)


def compute_drawdown(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax()
    return equity / cummax - 1


# ============================================================
# Quant A - Single function with tabs
# ============================================================

def run_quant_a():
    st.header("Quant A — Backtesting & Forecasting")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Strategy Testing", "📈 Forecasting"])
    
    # ==================== TAB 1: STRATEGY TESTING ====================
    with tab1:
        st.subheader("Strategy Testing")
        
        # Form for basic backtest parameters
        with st.form("backtest_form"):
            st.write("**📊 Backtest Parameters**")
            
            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Symbol", "AAPL")
                start_date = st.text_input("Start date", "2020-01-01")
            
            with col2:
                end_date = st.text_input("End date", "2023-01-01")
                initial_capital = st.number_input(
                    "Initial capital",
                    min_value=1_000,
                    max_value=1_000_000,
                    value=10_000,
                    step=1_000,
                )
            
            # Submit button prominently displayed
            st.form_submit_button("▶ Run Backtest", use_container_width=True)

        # Strategy selection section
        st.write("**🎯 Select Strategies to Compare**")
        col_strats = st.columns(3)
        
        with col_strats[0]:
            use_sma = st.checkbox("SMA Crossover", value=True)
        
        with col_strats[1]:
            use_momentum = st.checkbox("Momentum", value=False)
        
        with col_strats[2]:
            use_buyhold = st.checkbox("Buy & Hold", value=True)

        # Strategy parameters in expandable sections
        st.write("**⚙️ Strategy Parameters**")
        
        fast = 20
        slow = 50
        momentum_period = 14
        momentum_threshold = 0.02
        
        col_left, col_right = st.columns(2, gap="large")
        
        # SMA Parameters
        if use_sma:
            with col_left:
                with st.expander("📈 SMA Crossover Settings", expanded=True):
                    fast = st.slider("Fast SMA Period", 5, 100, 20, help="Short-term moving average")
                    slow = st.slider("Slow SMA Period", 10, 300, 50, help="Long-term moving average")
        
        # Momentum Parameters
        if use_momentum:
            with col_right:
                with st.expander("🔥 Momentum Settings", expanded=True):
                    momentum_period = st.slider("ROC Period", 5, 30, 14, help="Rate of change period in days")
                    momentum_threshold = st.slider("Threshold", 0.01, 0.10, 0.02, 0.01, help="Buy/Sell threshold")

        # Get form submission
        submitted = st.session_state.get("formSubmitted", False)
        if st.form_submit_button.__self__ is not None:
            submitted = True

        if not submitted:
            if st.session_state.backtest_result is None:
                return
        else:
            # Validate SMA strategy if selected
            if use_sma and fast >= slow:
                st.error("❌ Fast SMA must be less than Slow SMA")
                return

            # Fetch data once
            try:
                df = fetch_ohlcv(symbol, start_date, end_date, "1d")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

            # Run selected strategies
            results_dict = {}
            
            if use_sma:
                signal = sma_crossover(df, fast, slow)
                results_dict["SMA Crossover"] = run_backtest(df, signal, initial_capital)
            
            if use_momentum:
                signal = momentum_strategy(df, momentum_period, momentum_threshold)
                results_dict["Momentum"] = run_backtest(df, signal, initial_capital)
            
            # Store in session
            st.session_state.backtest_result = {
                "df": df,
                "results_dict": results_dict,
                "use_buyhold": use_buyhold,
                "fast": fast,
                "slow": slow,
                "initial_capital": initial_capital,
            }

        # Display results if available
        if st.session_state.backtest_result is not None:
            df = st.session_state.backtest_result["df"]
            results_dict = st.session_state.backtest_result["results_dict"]
            use_buyhold = st.session_state.backtest_result["use_buyhold"]
            fast = st.session_state.backtest_result["fast"]
            slow = st.session_state.backtest_result["slow"]
            initial_capital = st.session_state.backtest_result["initial_capital"]

            # Buy & Hold (benchmark)
            bh_df = df.copy()
            bh_df["bh_returns"] = bh_df["close"].pct_change().fillna(0.0)
            bh_df["bh_equity"] = initial_capital * (1 + bh_df["bh_returns"]).cumprod()

            # Display metrics for each strategy
            st.subheader("📊 Performance Metrics")
            
            # Create columns for each strategy
            metric_cols = st.columns(len(results_dict) + (1 if use_buyhold else 0))
            
            for idx, (strat_name, results) in enumerate(results_dict.items()):
                with metric_cols[idx]:
                    st.markdown(f"### {strat_name}")
                    metrics = compute_metrics(results)
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.2%}" if abs(v) < 2 else f"{v:.2f}")
            
            if use_buyhold:
                with metric_cols[len(results_dict)]:
                    st.markdown("### Buy & Hold")
                    bh_metrics = compute_metrics(
                        bh_df.assign(
                            strategy_returns=bh_df["bh_returns"],
                            equity=bh_df["bh_equity"],
                            position=1,
                        )
                    )
                    for k, v in bh_metrics.items():
                        st.metric(k, f"{v:.2%}" if abs(v) < 2 else f"{v:.2f}")

            # Equity Curve comparison
            st.subheader("📈 Equity Curve Comparison")
            equity_data = {}
            for strat_name, results in results_dict.items():
                equity_data[strat_name] = results["equity"].values
            
            if use_buyhold:
                equity_data["Buy & Hold"] = bh_df["bh_equity"].values
            
            equity_df = pd.DataFrame(equity_data, index=df["date"])
            st.line_chart(equity_df)

            # Drawdown comparison
            st.subheader("📉 Drawdown Comparison")
            dd_data = {}
            for strat_name, results in results_dict.items():
                dd_data[strat_name] = compute_drawdown(results["equity"]).values
            
            if use_buyhold:
                dd_data["Buy & Hold"] = compute_drawdown(bh_df["bh_equity"]).values
            
            dd_df = pd.DataFrame(dd_data, index=df["date"])
            st.line_chart(dd_df)

            # Price & MAs (only for SMA strategy)
            if "SMA Crossover" in results_dict:
                st.subheader("💹 Price & Moving Averages")
                price_df = pd.DataFrame({
                    "Close": df["close"].values,
                    "SMA Fast": df["close"].rolling(fast).mean().values,
                    "SMA Slow": df["close"].rolling(slow).mean().values,
                }, index=df["date"])
                st.line_chart(price_df)

            # Trades table for first strategy
            st.subheader("📋 Trades")
            first_strat = list(results_dict.keys())[0]
            trades_df = build_trades_table(results_dict[first_strat])

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
    
    # ==================== TAB 2: FORECASTING ====================
    with tab2:
        st.subheader("Price Forecasting")
        
        # Check if backtest data exists
        if st.session_state.backtest_result is None:
            st.warning("⚠️ Run a backtest first (in Strategy Testing tab) to get data for forecasting.")
            return
        
        df = st.session_state.backtest_result["df"]
        results_dict = st.session_state.backtest_result["results_dict"]
        
        st.write("**Forecast Parameters**")
        
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1:
            forecast_days = st.slider("Days ahead", 1, 60, 10, key="fc_days")
        
        with col2:
            confidence = st.selectbox(
                "Confidence",
                [0.90, 0.95, 0.99],
                index=1,
                format_func=lambda x: f"{int(x*100)}%",
                key="fc_conf"
            )
        
        with col3:
            forecast_type = st.selectbox(
                "Forecast on",
                ["Raw Price", "Strategy Equity"] + list(results_dict.keys()) if results_dict else ["Raw Price"],
                key="fc_type"
            )
        
        if st.button("Generate Forecast", key="fc_btn"):
            try:
                # Prepare data for forecast
                if forecast_type == "Raw Price":
                    # Forecast raw price
                    fc = predict_price(df, forecast_days, confidence)
                    forecast_title = "Price Forecast"
                else:
                    # Forecast strategy equity
                    if forecast_type in results_dict:
                        strat_df = df.copy()
                        strat_df["equity"] = results_dict[forecast_type]["equity"].values
                        fc = predict_price(strat_df[["date", "equity"]].rename(columns={"equity": "close"}), 
                                         forecast_days, confidence)
                        forecast_title = f"{forecast_type} Equity Forecast"
                    else:
                        st.error("Strategy not found")
                        return
                
                st.session_state.forecast_result = {
                    "data": fc,
                    "type": forecast_type,
                    "title": forecast_title
                }
            except Exception as e:
                st.error(f"Forecast error: {e}")
        
        # Display if available
        if st.session_state.forecast_result is not None:
            fc_info = st.session_state.forecast_result
            fc = fc_info["data"]
            
            st.success(f"✅ {fc_info['title']} ready")
            
            # Chart
            chart_df = pd.DataFrame({
                "Forecast": fc["predictions"],
                "Upper": fc["upper"],
                "Lower": fc["lower"],
            }, index=fc["dates"])
            st.line_chart(chart_df)
            
            # Table
            st.subheader("Forecast Details")
            table = pd.DataFrame({
                "Date": fc["dates"],
                "Forecast": fc["predictions"].round(2),
                "Upper": fc["upper"].round(2),
                "Lower": fc["lower"].round(2),
            })
            st.dataframe(table, use_container_width=True)


# ============================================================
# Quant B
# ============================================================

def run_quant_b():
    st.header("Quant B — Advanced Analysis")
    st.info("Workspace reserved for Quant B. To be implemented.")


# ============================================================
# Main App
# ============================================================

def main():
    st.set_page_config(layout="wide")
    st.title("Quant Project – Backtesting App")

    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Section", ["Quant A", "Quant B"])

    if section == "Quant A":
        run_quant_a()
    else:
        run_quant_b()


if __name__ == "__main__":
    main()
