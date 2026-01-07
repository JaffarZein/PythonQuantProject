import pandas as pd

def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    df = df.copy()

    df["signal"] = signal.shift(1).fillna(0)
    df["returns"] = df["close"].pct_change().fillna(0)
    df["strategy_returns"] = df["signal"] * df["returns"]

    df["equity"] = initial_capital * (1 + df["strategy_returns"]).cumprod()

    return df
