import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute performance metrics from backtest results.
    Expects columns:
    ['equity', 'strategy_returns', 'position', 'trade_id']
    """

    returns = df["strategy_returns"]

    # --- Returns ---
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1

    annualized_return = (1 + total_return) ** (TRADING_DAYS / len(df)) - 1

    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = (
        annualized_return / annualized_vol
        if annualized_vol > 0
        else np.nan
    )

    # --- Drawdown ---
    cumulative_max = df["equity"].cummax()
    drawdown = df["equity"] / cumulative_max - 1
    max_drawdown = drawdown.min()

    # --- Trades ---
    trades = (
        df.dropna(subset=["trade_id"])
          .groupby("trade_id")["strategy_returns"]
          .sum()
    )

    n_trades = len(trades)
    win_rate = (trades > 0).mean() if n_trades > 0 else np.nan

    # --- Exposure ---
    exposure = (df["position"] != 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "exposure": exposure,
    }
def compute_bh_metrics(df: pd.DataFrame) -> dict:
    bh_df = df.copy()
    bh_df["strategy_returns"] = bh_df["bh_returns"]
    bh_df["equity"] = bh_df["bh_equity"]
    return compute_metrics(bh_df)
