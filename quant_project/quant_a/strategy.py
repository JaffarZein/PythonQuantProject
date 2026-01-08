import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st


@st.cache_data(ttl=3600, hash_funcs={pd.DataFrame: id})
def predict_price(
    df: pd.DataFrame,
    forecast_days: int = 10,
    confidence: float = 0.95,
) -> dict:
    """
    Linear regression forecast with confidence intervals.
    Uses hash_funcs to avoid recomputing when DataFrame reference changes.
    """
    # Prepare data for regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate standard error for confidence intervals
    residuals = y - model.predict(X)
    std_error = np.sqrt(np.mean(residuals ** 2))
    
    # Z-score for confidence level (95% ≈ 1.96)
    z_score = 1.96 if confidence == 0.95 else 1.645
    margin = z_score * std_error
    
    # Generate forecast points
    last_idx = len(df)
    forecast_idx = np.arange(last_idx, last_idx + forecast_days).reshape(-1, 1)
    forecast_prices = model.predict(forecast_idx).flatten()
    
    # Generate forecast dates
    last_date = pd.to_datetime(df["date"].iloc[-1])
    freq = pd.infer_freq(pd.to_datetime(df["date"]))
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq=freq)[1:]
    
    return {
        "dates": forecast_dates,
        "predictions": forecast_prices,
        "upper": forecast_prices + margin,
        "lower": forecast_prices - margin,
        "margin": margin,
    }


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
    signal[df["sma_fast"] < df["sma_slow"]] = -1

    return signal


def momentum_strategy(
    df: pd.DataFrame,
    period: int = 14,
    threshold: float = 0.02,
) -> pd.Series:
    """
    Momentum-based strategy using rate of change.
    Buys when momentum exceeds threshold, sells when below -threshold.
    """
    # Calculate momentum as percentage change over period
    momentum = df["close"].pct_change(period)
    
    signal = pd.Series(0, index=df.index)
    signal[momentum > threshold] = 1
    signal[momentum < -threshold] = -1
    
    return signal
