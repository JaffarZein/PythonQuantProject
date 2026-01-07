import pandas as pd
import numpy as np

def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Vectorized backtest.

    Convention :
    - signal ∈ {-1, 0, +1}
    - signal[t] est décidé à la clôture t
    - position[t] = signal[t-1]
    """

    df = df.copy()

    # --- Alignement temporel ---
    df["signal"] = signal.reindex(df.index).fillna(0)
    df["position"] = df["signal"].shift(1).fillna(0)

    # --- Returns ---
    df["returns"] = df["close"].pct_change().fillna(0.0)

    # --- Strategy returns ---
    df["strategy_returns"] = df["position"] * df["returns"]

    # --- Equity curve ---
    df["equity"] = initial_capital * (1.0 + df["strategy_returns"]).cumprod()

    # --- Trade identification ---
    df["trade_entry"] = (df["position"] != 0) & (df["position"].shift(1) == 0)
    df["trade_exit"]  = (df["position"] == 0) & (df["position"].shift(1) != 0)

    df["trade_id"] = df["trade_entry"].cumsum()
    df.loc[df["position"] == 0, "trade_id"] = np.nan

    # --- Buy & Hold benchmark ---
    df["bh_returns"] = df["returns"]
    df["bh_equity"] = initial_capital * (1.0 + df["bh_returns"]).cumprod()


    return df
