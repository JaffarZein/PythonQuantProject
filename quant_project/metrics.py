import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute basic performance metrics from backtest results.
    Expects columns: ['equity', 'strategy_returns']
    """

    returns = df["strategy_returns"]

    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1

    annualized_return = (1 + total_return) ** (TRADING_DAYS / len(df)) - 1

    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = (
        annualized_return / annualized_vol
        if annualized_vol > 0
        else np.nan
    )

    cumulative_max = df["equity"].cummax()
    drawdown = df["equity"] / cumulative_max - 1
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
