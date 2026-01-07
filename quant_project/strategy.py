import pandas as pd


def run_strategy():
    print("Running trading strategy")

def sma_crossover(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
) -> pd.Series:

    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()

    signal = pd.Series(0, index=df.index)
    signal[df["sma_fast"] > df["sma_slow"]] = 1
    signal[df["sma_fast"] < df["sma_slow"]] = -1  # optionnel

    return signal
