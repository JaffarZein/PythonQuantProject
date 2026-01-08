"""
Portfolio management and analysis module.
Handles multi-asset portfolio simulation and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import streamlit as st


def fetch_multiple_assets(symbols: List[str], start_date: str, end_date: str, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple assets.
    
    Args:
        symbols: List of asset symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        timeframe: Data timeframe ('1d', '1h', etc.)
    
    Returns:
        Dict mapping symbol to DataFrame with OHLCV data
    """
    from quant_project.fetch_yahoo import fetch_ohlcv
    
    assets_data = {}
    for symbol in symbols:
        try:
            assets_data[symbol] = fetch_ohlcv(symbol, start_date, end_date, timeframe)
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return {}
    
    return assets_data


def align_asset_data(assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple asset dataframes by date.
    
    Args:
        assets_data: Dict mapping symbol to DataFrame
    
    Returns:
        Single DataFrame with all assets aligned by date
    """
    if not assets_data:
        return None
    
    # Get closing prices for all assets
    dfs = []
    for symbol, df in assets_data.items():
        dfs.append(df[["date", "close"]].rename(columns={"close": symbol}))
    
    # Merge on date
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="inner")
    
    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def compute_portfolio_returns(prices_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Compute portfolio daily returns given asset weights.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols as columns
        weights: Dict mapping symbol to weight (should sum to 1.0)
    
    Returns:
        Series of portfolio returns
    """
    # Calculate daily returns for each asset
    returns = prices_df[[col for col in prices_df.columns if col != "date"]].pct_change().fillna(0.0)
    
    # Weight returns - initialize as Series of zeros
    portfolio_return = pd.Series(0.0, index=returns.index)
    for symbol, weight in weights.items():
        if symbol in returns.columns:
            portfolio_return += returns[symbol] * weight
    
    return portfolio_return


def compute_portfolio_equity(prices_df: pd.DataFrame, weights: Dict[str, float], initial_capital: float) -> pd.Series:
    """
    Compute portfolio equity curve given weights.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols as columns
        weights: Dict mapping symbol to weight (should sum to 1.0)
        initial_capital: Starting capital
    
    Returns:
        Series of portfolio equity values
    """
    port_returns = compute_portfolio_returns(prices_df, weights)
    equity = initial_capital * (1 + port_returns).cumprod()
    return equity


def compute_correlation_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of asset returns.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols as columns
    
    Returns:
        Correlation matrix (DataFrame)
    """
    returns = prices_df[[col for col in prices_df.columns if col != "date"]].pct_change().fillna(0.0)
    return returns.corr()


def compute_portfolio_volatility(prices_df: pd.DataFrame, weights: Dict[str, float]) -> float:
    """
    Compute annualized portfolio volatility.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols as columns
        weights: Dict mapping symbol to weight
    
    Returns:
        Annualized volatility (decimal)
    """
    returns = prices_df[[col for col in prices_df.columns if col != "date"]].pct_change().fillna(0.0)
    cov_matrix = returns.cov()
    
    # Portfolio variance
    symbols = [col for col in prices_df.columns if col != "date"]
    weight_list = [weights.get(s, 0) for s in symbols]
    
    port_var = 0
    for i, w_i in enumerate(weight_list):
        for j, w_j in enumerate(weight_list):
            port_var += w_i * w_j * cov_matrix.iloc[i, j]
    
    # Annualized (252 trading days)
    annual_vol = np.sqrt(port_var * 252)
    return annual_vol


def compute_portfolio_metrics(prices_df: pd.DataFrame, weights: Dict[str, float], initial_capital: float, risk_free_rate: float = 0.02) -> Dict:
    """
    Compute comprehensive portfolio performance metrics.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols
        weights: Dict mapping symbol to weight
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Dict with metrics: total_return, annual_return, volatility, sharpe_ratio, max_drawdown
    """
    equity = compute_portfolio_equity(prices_df, weights, initial_capital)
    returns = compute_portfolio_returns(prices_df, weights)
    volatility = compute_portfolio_volatility(prices_df, weights)
    
    # Total return
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital
    
    # Annual return (approximation: total return * 252 / days)
    days = len(prices_df)
    annual_return = (1 + total_return) ** (252 / days) - 1
    
    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


def rebalance_portfolio(prices_df: pd.DataFrame, weights: Dict[str, float], 
                       initial_capital: float, rebalance_freq: str = "monthly") -> pd.Series:
    """
    Simulate portfolio with periodic rebalancing.
    
    Args:
        prices_df: DataFrame with 'date' and asset symbols
        weights: Target weights (constant)
        initial_capital: Starting capital
        rebalance_freq: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    
    Returns:
        Series of portfolio equity with rebalancing
    """
    # Frequency in number of days
    freq_map = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
        "quarterly": 63,
        "yearly": 252,
    }
    
    rebalance_period = freq_map.get(rebalance_freq, 21)
    portfolio_values = []
    
    prices = prices_df[[col for col in prices_df.columns if col != "date"]].values
    symbols = [col for col in prices_df.columns if col != "date"]
    
    # Initialize holdings at target weights
    holdings = {}
    for j, symbol in enumerate(symbols):
        target_value = initial_capital * weights.get(symbol, 0)
        holdings[symbol] = target_value / prices[0, j]
    
    for i in range(len(prices)):
        # Calculate current equity at today's prices
        equity = sum(holdings[symbols[j]] * prices[i, j] for j in range(len(symbols)))
        portfolio_values.append(equity)
        
        # Rebalance if needed (not on first day)
        if i > 0 and i % rebalance_period == 0:
            for j, symbol in enumerate(symbols):
                target_value = equity * weights.get(symbol, 0)
                holdings[symbol] = target_value / prices[i, j]
    
    return pd.Series(portfolio_values, name="Portfolio Value")
