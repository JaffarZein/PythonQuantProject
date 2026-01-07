from quant_project.fetch_yahoo import fetch_ohlcv
from quant_project.strategy import sma_crossover
from quant_project.backtest import run_backtest
from quant_project.metrics import compute_metrics

def run():
    df = fetch_ohlcv(
        symbol="AAPL",
        start="2020-01-01",
        end="2023-01-01",
        interval="1d",
    )

    signal = sma_crossover(df, fast=20, slow=50)

    results = run_backtest(
        df=df,
        signal=signal,
        initial_capital=10_000,
    )

    print(results[["close", "equity"]].tail())

    metrics = compute_metrics(results)

    print("\nPerformance metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(results[["signal", "position", "returns", "strategy_returns", "equity"]].tail(10))
    print(metrics)
