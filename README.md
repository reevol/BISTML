# BIST AI Trading System

A comprehensive, hybrid AI-driven quantitative trading system specifically tailored for the Borsa Istanbul (BIST) equity market. This system generates high-probability, short-term trading signals (30-minute and hourly) by integrating Machine Learning (ML), Natural Language Processing (NLP), and advanced Flow/Whale Analysis.

## 🎯 Project Overview

**BIST High-Frequency Signal Generation and Whale Tracking Hybrid AI Quant System**

Core objective: Engineer a robust, modular, and data-intensive Python program capable of producing actionable, prioritized BUY/SELL/HOLD signals every 30 minutes and hourly for BIST-listed equities.

## 🌟 Key Features

### Data Engineering
- ✅ Comprehensive OHLCV data collection for all BIST 100 constituents
- ✅ Multiple timeframes: Daily, Hourly, 30-Minute, 15-Minute
- ✅ Fundamental data from quarterly and annual financial statements
- ✅ Macro indicators (Turkish CPI/PPI, Interest Rates, Global Indices)
- ✅ Whale/Takas data (Brokerage distribution analysis)
- ✅ Turkish news collection from major sources and KAP

### Feature Engineering
- ✅ 30+ Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- ✅ Fundamental metrics (P/E, P/B, EV/EBITDA, ROE, ROA, etc.)
- ✅ Whale Activity Index (WAI) for institutional flow tracking
- ✅ Accumulation/Distribution pattern detection
- ✅ Advanced feature engineering pipeline

### Machine Learning Models
- ✅ **Price Forecasting**: LSTM, GRU, XGBoost, LightGBM
- ✅ **Signal Classification**: Random Forest, ANN, Ensemble methods
- ✅ **Turkish NLP**: BERTurk-based sentiment analysis
- ✅ **LLM Integration**: News synthesis and impact scoring
- ✅ Model training orchestrator with hyperparameter tuning

### Signal Generation
- ✅ Multi-model ensemble signal generation
- ✅ Confidence scoring (0-100%)
- ✅ Signal prioritization with WAI and sentiment
- ✅ Automated scheduler (30-min/1-hr intervals)
- ✅ BIST trading hours and holiday detection

### Portfolio Management
- ✅ Position tracking with multiple cost basis methods (FIFO, LIFO, Average)
- ✅ Real-time P&L calculation
- ✅ Portfolio optimization (Kelly Criterion, Risk Parity, Mean-Variance)
- ✅ Multi-channel alerts (Email, Telegram, SMS)

### Backtesting & Validation
- ✅ Historical simulation engine
- ✅ Walk-forward analysis
- ✅ Monte Carlo simulation
- ✅ 25+ performance metrics (Sharpe, Sortino, Calmar, Win Rate, etc.)
- ✅ Transaction cost modeling (commission + slippage)

### User Interfaces
- ✅ Streamlit dashboard with real-time updates
- ✅ Command-line interface (CLI)
- ✅ PDF/HTML report generation

## 📁 Project Structure

```
BISTML/
├── src/
│   ├── data/              # Data collection and processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models (forecasting, classification, NLP)
│   ├── signals/           # Signal generation and scheduling
│   ├── portfolio/         # Portfolio management and optimization
│   ├── backtesting/       # Backtesting engine and metrics
│   ├── utils/             # Utilities (config, logging, validators)
│   └── ui/                # User interfaces (dashboard, CLI, reports)
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── examples/              # Usage examples
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── README.md             # This file
├── project.md            # Original project specification
└── claude.md             # Implementation plan
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BISTML.git
cd BISTML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (FRED, EVDS, OpenAI, etc.)
```

### Basic Usage

#### 1. Collect Data
```bash
python -m src.ui.cli collect-data --sources all --symbols THYAO,GARAN,AKBNK
```

#### 2. Train Models
```bash
python -m src.ui.cli train-models --model-types xgboost_regressor,lightgbm_regressor
```

#### 3. Generate Signals
```bash
python -m src.ui.cli run-signals --symbols THYAO,GARAN --output signals.json
```

#### 4. Launch Dashboard
```bash
python run_dashboard.py
# Open browser to http://localhost:8501
```

## 📊 Signal Output Format

| Column | Description |
|--------|-------------|
| Stock Code | BIST Ticker |
| Final Signal | Strong BUY / BUY / HOLD / SELL / Strong SELL |
| ML Target Price | Predicted price for next period |
| Prediction Confidence | 0-100% ensemble score |
| WAI Score | Whale Activity Index (institutional flow) |
| News Sentiment | -1 to +1 sentiment score |

## 🏗️ Architecture

### Data Flow
```
Data Collection → Cleaning/Validation → Feature Engineering →
ML Models → Signal Generation → Portfolio Management →
Backtesting/Reporting
```

### Model Pipeline
```
OHLCV + Fundamentals + Whale Data + News →
Technical Indicators + Fundamental Metrics + WAI + Sentiment →
Regression Models (Price) + Classification Models (Signal) →
Ensemble Aggregation → Final Signal + Confidence
```

## 📈 Performance Metrics

The system calculates and reports:
- **Returns**: Total, Annualized, CAGR
- **Risk-Adjusted**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Drawdown**: Maximum Drawdown, Duration, Recovery Factor
- **Trade Stats**: Win Rate, Profit Factor, Average Win/Loss, Expectancy
- **Alpha/Beta**: Performance vs. XU100 benchmark

## 🔧 Configuration

Key configuration files in `configs/`:
- `data_sources.yaml` - Data source settings and API keys
- `model_params.yaml` - ML model hyperparameters
- `trading_params.yaml` - Trading strategy parameters
- `scheduler_config.yaml` - Signal generation schedule

## 📚 Documentation

- [Implementation Plan](claude.md) - Detailed implementation roadmap
- [Project Specification](project.md) - Original project requirements
- [API Documentation](docs/) - Complete API reference
- [Examples](examples/) - Usage examples for all modules

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_data/ -v
pytest tests/test_models/ -v

# Generate coverage report
pytest --cov=src tests/
```

## 🤝 Contributing

This is a comprehensive AI trading system. Key areas for contribution:
1. Additional data sources
2. New ML models or features
3. Enhanced trading strategies
4. Performance optimizations
5. Documentation improvements

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Trading stocks involves substantial risk of loss. Always do your own research and consult with licensed financial advisors before making investment decisions.

The developers are not responsible for any financial losses incurred through the use of this system.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- BIST (Borsa Istanbul) for market data
- Turkish Central Bank (TCMB) for macro data
- KAP (Public Disclosure Platform) for regulatory filings
- Open source ML libraries: scikit-learn, TensorFlow, PyTorch, XGBoost
- Hugging Face Transformers for Turkish NLP models

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for quantitative trading on BIST**
