# BIST AI Trading System - Implementation Plan

## Project Structure

```
BISTML/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── bist_collector.py          # BIST OHLCV data collection
│   │   │   ├── fundamental_collector.py    # Financial statements
│   │   │   ├── macro_collector.py          # Macro indicators
│   │   │   ├── whale_collector.py          # Brokerage distribution data
│   │   │   └── news_collector.py           # Turkish news sources
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py                  # Data cleaning
│   │   │   ├── synchronizer.py             # Time series sync
│   │   │   └── validator.py                # Data validation
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── database.py                 # Database operations
│   │       └── cache.py                    # Caching layer
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical/
│   │   │   ├── __init__.py
│   │   │   ├── trend.py                    # SMA, EMA, HMA, Ichimoku
│   │   │   ├── momentum.py                 # RSI, MACD, Stochastic
│   │   │   ├── volatility.py               # Bollinger, ATR, Donchian
│   │   │   └── advanced.py                 # Custom indicators
│   │   ├── fundamental/
│   │   │   ├── __init__.py
│   │   │   ├── valuation.py                # P/E, P/B, EV/EBITDA
│   │   │   └── growth.py                   # Growth metrics
│   │   ├── whale/
│   │   │   ├── __init__.py
│   │   │   ├── activity_index.py           # Whale Activity Index (WAI)
│   │   │   ├── accumulation.py             # Accumulation/Distribution
│   │   │   └── flow_analysis.py            # Flow patterns
│   │   └── feature_engineering.py          # Feature combination
│   ├── models/
│   │   ├── __init__.py
│   │   ├── forecasting/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_model.py               # LSTM price forecasting
│   │   │   ├── gru_model.py                # GRU price forecasting
│   │   │   ├── xgboost_model.py            # XGBoost regression
│   │   │   └── lightgbm_model.py           # LightGBM regression
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── random_forest.py            # RF classifier
│   │   │   ├── ann_classifier.py           # ANN classifier
│   │   │   └── ensemble.py                 # Ensemble methods
│   │   ├── nlp/
│   │   │   ├── __init__.py
│   │   │   ├── turkish_sentiment.py        # Turkish NLP model
│   │   │   ├── news_analyzer.py            # News analysis
│   │   │   └── llm_integration.py          # LLM synthesis
│   │   └── trainer.py                      # Model training orchestrator
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── generator.py                    # Signal generation engine
│   │   ├── prioritizer.py                  # Signal prioritization
│   │   ├── confidence.py                   # Confidence scoring
│   │   └── scheduler.py                    # 30-min/1-hr scheduler
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── manager.py                      # Portfolio management
│   │   ├── alerts.py                       # Alert generation
│   │   └── optimization.py                 # Position sizing
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py                       # Backtesting engine
│   │   ├── metrics.py                      # Performance metrics
│   │   └── simulator.py                    # Historical simulation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                       # Configuration management
│   │   ├── logger.py                       # Logging utilities
│   │   ├── validators.py                   # Input validators
│   │   └── helpers.py                      # Helper functions
│   └── ui/
│       ├── __init__.py
│       ├── dashboard.py                    # Streamlit dashboard
│       ├── cli.py                          # Command-line interface
│       └── reports.py                      # Report generation
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_features/
│   ├── test_models/
│   ├── test_signals/
│   └── test_backtesting/
├── configs/
│   ├── data_sources.yaml
│   ├── model_params.yaml
│   ├── trading_params.yaml
│   └── backtesting_params.yaml
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_evaluation.ipynb
├── requirements.txt
├── setup.py
├── README.md
├── project.md
└── claude.md
```

## Implementation Phases

### Phase 1: Foundation (Data Infrastructure)
1. Database schema design
2. Data collectors for BIST, fundamentals, macro
3. Whale/Takas data collection
4. News collection from Turkish sources
5. Data cleaning and validation
6. Storage and caching systems

### Phase 2: Feature Engineering
1. Technical indicators (30+ indicators)
2. Fundamental metrics calculation
3. Whale Activity Index (WAI) development
4. Feature engineering pipeline
5. Feature selection and importance analysis

### Phase 3: Machine Learning Models
1. LSTM/GRU models for price forecasting
2. XGBoost/LightGBM regression models
3. Random Forest classification
4. ANN classification
5. Turkish NLP sentiment model
6. LLM integration for news synthesis
7. Model training pipeline

### Phase 4: Signal Generation
1. Signal generation engine
2. Confidence scoring system
3. Signal prioritization
4. Automated scheduler (30-min/1-hr)
5. Output formatting

### Phase 5: Portfolio & Backtesting
1. Portfolio management module
2. Alert system
3. Backtesting engine
4. Performance metrics calculation
5. Historical simulation

### Phase 6: User Interface
1. CLI interface
2. Streamlit dashboard
3. Real-time signal display
4. Backtesting visualization
5. Report generation

## Key Technologies

- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **NLP**: Transformers, NLTK, spaCy
- **Technical Analysis**: TA-Lib, pandas-ta
- **Database**: SQLite/PostgreSQL, Redis
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Framework**: Streamlit, Dash
- **Scheduling**: APScheduler
- **API**: Flask/FastAPI (optional)

## Critical Features

### Whale Activity Index (WAI)
- Track top N brokers' net flow
- Detect accumulation/distribution patterns
- Price discrepancy analysis
- Institutional ownership changes

### Signal Output Format
| Column | Description |
|--------|-------------|
| Stock Code | BIST Ticker |
| Final Signal | Strong BUY/BUY/HOLD/SELL/Strong SELL |
| ML Target Price | Predicted price |
| Prediction Confidence | 0-100% |
| WAI Score | Whale activity score |
| News Sentiment | -1 to +1 |

### Performance Metrics
- Win Rate
- Average Profit/Loss per trade
- Maximum Drawdown (MDD)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

## Development Priorities

1. **High Priority**: Data collection, technical indicators, basic ML models
2. **Medium Priority**: Whale analysis, NLP sentiment, backtesting
3. **Low Priority**: Advanced LLM integration, advanced visualization

## Subagent Task Distribution

The implementation will be distributed across 50+ specialized subagents, each focusing on specific modules for parallel development.
