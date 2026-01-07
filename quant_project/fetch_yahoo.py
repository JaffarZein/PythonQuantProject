import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_ohlcv(symbol: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    df = df.reset_index()

    df.columns = [
        (c[0] if isinstance(c, tuple) else c)
            .lower()
            .replace(" ", "_")
        for c in df.columns
    ]

    return df

def save_ohlcv(df: pd.DataFrame, symbol: str, interval: str):
    path = DATA_DIR / f"{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    return path
