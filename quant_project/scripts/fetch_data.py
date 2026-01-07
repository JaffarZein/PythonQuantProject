from quant_project.fetch_yahoo import fetch_ohlcv, save_ohlcv

df = fetch_ohlcv(
    symbol="AAPL",
    start="2020-01-01",
    end="2023-01-01",
    interval="1d",
)

path = save_ohlcv(df, "AAPL", "1d")
print(f"Saved to {path}")
