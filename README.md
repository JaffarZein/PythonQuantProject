# Quant Project – Advanced Trading & Portfolio Analysis Platform

A comprehensive quantitative trading platform built with Python and Streamlit, featuring single-asset backtesting with machine learning forecasting and multi-asset portfolio analysis with dynamic rebalancing.

## Overview

This project provides two integrated modules for quantitative analysis:

- **Quant A**: Single-asset strategy backtesting with multiple strategies and ML-powered price forecasting
- **Quant B**: Multi-asset portfolio analysis with correlation analysis and rebalancing simulation

## Key Features

### Quant A – Strategy Backtesting Engine
- **Multiple Strategies**:
  - SMA (Simple Moving Average) Crossover with fast/slow validation
  - Momentum strategy using Rate of Change (ROC)
  - Buy & Hold baseline for comparison
  
- **Machine Learning Forecasting**:
  - Linear regression-based price prediction
  - Configurable confidence intervals (90%, 95%, 99%)
  - Forecast on raw prices or strategy equity curves
  
- **Performance Metrics**:
  - Total return and annualized return
  - Volatility (annualized)
  - Sharpe Ratio
  - Maximum Drawdown
  - Win rate and profit factor
  
- **Advanced Features**:
  - Multi-strategy comparison and backtesting
  - Trade entry/exit analysis with full trade history
  - Interactive equity curve visualization
  - CSV export functionality

### Quant B – Portfolio Analysis & Rebalancing
- **Multi-Asset Support**:
  - Support for 1-5 assets simultaneously
  - Real-time data from Yahoo Finance
  - Custom weight allocation per asset
  
- **Portfolio Analysis**:
  - Portfolio return, volatility, and Sharpe ratio calculation
  - Correlation matrix between assets
  - Individual asset performance tracking
  - Equity curve visualization
  
- **Rebalancing Strategies**:
  - Monthly, quarterly, semi-annual, annual, and manual rebalancing
  - Performance comparison vs buy-and-hold strategy
  - Detailed rebalancing metrics and transaction analysis

## Project Structure

```
quant-project/
├── app.py                          # Main Streamlit application
├── README.md                       # This file
├── data/
│   └── AAPL_1d.csv               # Sample market data
├── quant_project/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── fetch_yahoo.py            # Yahoo Finance data fetching
│   ├── main.py                   # CLI entry point
│   ├── quant_a/
│   │   ├── __init__.py
│   │   ├── strategy.py           # Strategy definitions & ML forecasting
│   │   ├── backtest.py           # Backtesting engine
│   │   ├── core.py               # Core backtesting logic
│   │   └── metrics.py            # Performance metrics calculation
│   ├── quant_b/
│   │   ├── __init__.py
│   │   └── portfolio.py          # Portfolio analysis & rebalancing
│   └── scripts/
│       └── fetch_data.py         # Data fetching utilities
└── venv/                          # Python virtual environment
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/JaffarZein/PythonQuantProject.git
cd quant-project
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Required Dependencies
- `streamlit>=1.28.0` – Interactive web framework
- `pandas>=2.0.0` – Data manipulation
- `numpy>=1.24.0` – Numerical computing
- `yfinance>=0.2.28` – Yahoo Finance data fetching
- `scikit-learn>=1.3.0` – Machine learning models

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application provides an interactive web interface with two main modules accessible from the sidebar.

### Quant A Workflow

1. **Navigate to Quant A** from the sidebar
2. **Strategy Testing Tab**:
   - Enter stock symbol (e.g., AAPL)
   - Select date range (DD/MM/YY format)
   - Set initial capital
   - Choose strategies (SMA Crossover, Momentum, Buy & Hold)
   - Configure strategy parameters:
     - **SMA**: Fast period (5-50), Slow period (20-200) – Fast must be < Slow
     - **Momentum**: Period (5-50), Threshold (0.01-0.10)
   - Click "Run Backtest" to execute
   - Review performance metrics and equity curves
   - View detailed trade history

3. **Forecasting Tab**:
   - Configure forecast parameters (days ahead, confidence level)
   - Choose forecast target (raw price or strategy equity)
   - View confidence intervals and predictions
   - Analyze forecast uncertainty

### Quant B Workflow

1. **Navigate to Quant B** from the sidebar
2. **Setup Tab**:
   - Enter date range (DD/MM/YY format)
   - Set initial capital
   - Add 1-5 assets (ticker symbols)
   - Allocate weights (must sum to 100%)
   
3. **Analysis Tab**:
   - View portfolio metrics (return, volatility, Sharpe ratio)
   - Examine correlation matrix between assets
   - Analyze individual asset and portfolio equity curves
   
4. **Rebalancing Tab**:
   - Select rebalancing frequency
   - Compare buy-and-hold vs rebalanced performance
   - View rebalancing metrics and transaction impacts

## Key Functions

### Quant A Module (`quant_project/quant_a/strategy.py`)

```python
sma_crossover(df, fast, slow)
# Returns trading signals (-1: sell, 0: hold, 1: buy)

momentum_strategy(df, period, threshold)
# Returns momentum-based signals with ROC threshold

predict_price(df, forecast_days, confidence)
# Returns price predictions with confidence intervals
```

### Quant B Module (`quant_project/quant_b/portfolio.py`)

```python
fetch_multiple_assets(symbols, start_date, end_date)
# Fetches OHLCV data for multiple assets

align_asset_data(assets_data)
# Synchronizes multiple asset dataframes by date

compute_portfolio_metrics(prices_df, weights, initial_capital)
# Calculates comprehensive portfolio performance metrics

rebalance_portfolio(prices_df, weights, initial_capital, rebalance_freq)
# Simulates portfolio rebalancing and returns equity curve
```

## Date Format

All dates in the application use **DD/MM/YY** format for consistency:
- Example: `08/01/26` = January 8, 2026
- End date automatically defaults to today's date when left blank

## Technologies

- **Framework**: Streamlit 1.28.0+
- **Data Processing**: pandas, numpy
- **Data Source**: Yahoo Finance (via yfinance)
- **Machine Learning**: scikit-learn
- **Backtesting**: Custom vectorized engine
- **Language**: Python 3.8+

## Deployment

### Streamlit Cloud (Recommended)

The easiest way to share this project with others:

1. Push code to GitHub (already done)
2. Go to [streamlit.io](https://streamlit.io)
3. Click "Deploy" → "Connect GitHub repo"
4. Select repository and branch (`main`)
5. Share the public link with your professor

**Advantages:**
- Free tier available
- Automatic deployment on every push
- No server configuration needed
- Easy to share via URL

### Local Deployment

Run directly on your machine:
```bash
streamlit run app.py
```

## Project Status

- ✅ **Quant A** – Complete and tested (SMA, Momentum, forecasting)
- ✅ **Quant B** – Complete and tested (portfolio analysis, rebalancing)
- ✅ **Documentation** – Comprehensive README and code comments
- ✅ **Dependencies** – requirements.txt provided
- ✅ **Git** – All code on `main` branch, ready for deployment

### Version History

- **Latest**: Quant B fully restored with all portfolio features
- **Current**: Main branch contains both Quant A and Quant B complete implementations
- **Branches**: feature/quant-a-improvements, feature/quant-b available for reference

## Git Workflow

The project uses feature branches for development:

- `main` – Stable production code (ready for deployment)
- `feature/quant-a-improvements` – Quant A development history
- `feature/quant-b` – Quant B development history

## Limitations & Future Enhancements

- Single-asset Quant A does not support options or derivatives
- Portfolio analysis assumes daily rebalancing capability
- Backtesting does not include slippage or commission modeling (can be added)
- Forecast confidence intervals based on linear regression (can extend to other models)

## Contributing

Contributions are welcome! Please:
1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Submit a pull request with description

## Support

For issues, questions, or suggestions, please open an issue in the GitHub repository.

---

**Disclaimer**: This tool is for educational and research purposes only. It is not financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.
