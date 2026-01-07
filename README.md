# Quant Project – Backtesting & Strategy Evaluation

## Objective
This project implements a simple quantitative trading backtesting framework
with a Streamlit interface for strategy evaluation.

## Quant A – Backtesting Engine
Quant A focuses on:
- Market data fetching (Yahoo Finance)
- Strategy signal generation (SMA crossover)
- Vectorized backtesting
- Performance metrics (return, volatility, Sharpe ratio, drawdown)
- Interactive Streamlit application

## Quant B – Advanced Analysis (planned)
Quant B will extend the project with:
- Alternative strategies
- Risk management techniques
- Advanced performance analytics
- Extended visualizations

## Project Structure

quant_project/
├── fetch_yahoo.py
├── strategy.py
├── backtest.py
├── metrics.py
├── core.py
├── main.py
scripts/
app.py
data/

## How to run
```bash
python -m quant_project.main
streamlit run app.py
