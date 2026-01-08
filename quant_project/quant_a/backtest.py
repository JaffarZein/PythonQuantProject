import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Vectorized backtest.

    Conventions :
    - signal ∈ {-1, 0, +1}
    - signal[t] est décidé à la clôture t
    - position[t] = signal[t-1]
    - Un trade = période continue avec position constante et non nulle
    """

    df = df.copy()

    # --- Alignment temporel ---
    df["signal"] = signal.reindex(df.index).fillna(0)
    df["position"] = df["signal"].shift(1).fillna(0)

    # --- Returns ---
    df["returns"] = df["close"].pct_change().fillna(0.0)

    # --- Strategy returns ---
    df["strategy_returns"] = df["position"] * df["returns"]

    # --- Equity curve ---
    df["equity"] = initial_capital * (1.0 + df["strategy_returns"]).cumprod()

    # --- Trade identification ---
    # Nouveau trade à chaque changement de position (0→±1, ±1→0, ±1→∓1)
    trade_change = df["position"] != df["position"].shift(1)
    trade_id = trade_change.cumsum()

    # trade_id uniquement quand on est en position
    df["trade_id"] = np.where(df["position"] != 0, trade_id, np.nan)

    return df
