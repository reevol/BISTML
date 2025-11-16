# BIST AI Trading System - CLI Usage Guide

## Overview

The BIST AI Trading System CLI provides a comprehensive command-line interface for interacting with the AI trading system. The CLI is built using Python's `argparse` module (standard library - no additional dependencies).

## Location

**File:** `/home/user/BISTML/src/ui/cli.py`

## Available Commands

### 1. run-signals - Generate Trading Signals

Generate trading signals for BIST stocks using trained AI models.

**Usage:**
```bash
python -m src.ui.cli run-signals [OPTIONS]
```

**Options:**
- `--symbols SYMBOLS` - Comma-separated list of stock symbols (default: THYAO,GARAN,AKBNK)
- `--min-confidence FLOAT` - Minimum confidence threshold for signals (default: 0.5)
- `--no-dynamic-thresholds` - Disable dynamic threshold adjustment
- `--no-risk-adjustment` - Disable risk-based signal adjustment
- `--output FILE, -o FILE` - Output file for generated signals (JSON format)

**Examples:**
```bash
# Generate signals for default symbols
python -m src.ui.cli run-signals

# Generate signals for specific symbols with output
python -m src.ui.cli run-signals --symbols THYAO,GARAN,AKBNK --output signals.json

# Generate signals with custom confidence threshold
python -m src.ui.cli run-signals --symbols THYAO --min-confidence 0.7 --output thyao_signals.json
```

---

### 2. backtest - Run Backtesting

Backtest trading strategies on historical data with comprehensive performance metrics.

**Usage:**
```bash
python -m src.ui.cli backtest [OPTIONS]
```

**Options:**
- `--symbol SYMBOL` - Stock symbol to backtest (default: THYAO)
- `--start-date DATE` - Start date for backtest (YYYY-MM-DD)
- `--end-date DATE` - End date for backtest (YYYY-MM-DD)
- `--initial-capital FLOAT` - Initial capital in TRY (default: 100000)
- `--commission-rate FLOAT` - Commission rate as decimal (default: 0.001 = 0.1%)
- `--slippage-rate FLOAT` - Slippage rate as decimal (default: 0.0005 = 0.05%)
- `--position-size FLOAT` - Position size as fraction of equity (default: 0.1 = 10%)
- `--min-confidence FLOAT` - Minimum confidence to execute trade (default: 0.5)
- `--no-stop-loss` - Disable stop loss orders
- `--no-take-profit` - Disable take profit orders
- `--signals-file FILE` - CSV file containing pre-computed signals
- `--output DIR, -o DIR` - Output directory for backtest results

**Examples:**
```bash
# Basic backtest with defaults
python -m src.ui.cli backtest --symbol THYAO

# Backtest with custom date range and capital
python -m src.ui.cli backtest --symbol THYAO \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --initial-capital 100000

# Backtest with custom parameters and save results
python -m src.ui.cli backtest --symbol GARAN \
  --initial-capital 200000 \
  --commission-rate 0.002 \
  --position-size 0.15 \
  --output backtest_results/
```

---

### 3. train-models - Train ML Models

Train or retrain machine learning models with cross-validation and hyperparameter tuning.

**Usage:**
```bash
python -m src.ui.cli train-models [OPTIONS]
```

**Options:**
- `--model-types TYPES` - Comma-separated model types to train
  - Available: lstm, gru, xgboost_regressor, lightgbm_regressor, random_forest_classifier, ann_classifier
- `--task-type TYPE` - Type of ML task (regression, classification, time_series_forecasting)
- `--data-file FILE` - CSV file containing training data
- `--n-samples INT` - Number of samples to generate if no data file provided
- `--train-size FLOAT` - Training set size as fraction (default: 0.7)
- `--val-size FLOAT` - Validation set size as fraction (default: 0.15)
- `--test-size FLOAT` - Test set size as fraction (default: 0.15)
- `--split-strategy STRATEGY` - Data splitting strategy (time_series, random, stratified)
- `--no-cross-validation` - Disable cross-validation
- `--cv-folds INT` - Number of cross-validation folds (default: 5)
- `--tune-hyperparameters` - Enable hyperparameter tuning
- `--no-save` - Do not save trained models
- `--model-dir DIR` - Directory to save trained models (default: models)
- `--verbose` - Enable verbose output

**Examples:**
```bash
# Train a single XGBoost model
python -m src.ui.cli train-models --model-types xgboost_regressor

# Train multiple models with hyperparameter tuning
python -m src.ui.cli train-models \
  --model-types xgboost_regressor,lightgbm_regressor \
  --tune-hyperparameters \
  --verbose

# Train classification models with custom data
python -m src.ui.cli train-models \
  --model-types random_forest_classifier,ann_classifier \
  --task-type classification \
  --data-file training_data.csv \
  --cv-folds 10
```

---

### 4. collect-data - Collect Market Data

Collect market data from various sources for BIST stocks.

**Usage:**
```bash
python -m src.ui.cli collect-data [OPTIONS]
```

**Options:**
- `--sources SOURCES` - Comma-separated data sources
  - Available: bist, fundamental, macro, news, whale, all
- `--symbols SYMBOLS` - Comma-separated stock symbols (default: THYAO,GARAN,AKBNK)
- `--start-date DATE` - Start date for data collection (YYYY-MM-DD)
- `--end-date DATE` - End date for data collection (YYYY-MM-DD)
- `--output DIR, -o DIR` - Output directory for collected data

**Examples:**
```bash
# Collect all available data for default symbols
python -m src.ui.cli collect-data --sources all

# Collect BIST price data for specific symbols
python -m src.ui.cli collect-data \
  --sources bist \
  --symbols THYAO,GARAN,AKBNK \
  --start-date 2023-01-01 \
  --output data/bist/

# Collect fundamental and news data
python -m src.ui.cli collect-data \
  --sources fundamental,news \
  --symbols THYAO \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output data/collected/
```

---

### 5. show-portfolio - Display Portfolio Status

Display portfolio positions, P&L, and performance metrics.

**Usage:**
```bash
python -m src.ui.cli show-portfolio [OPTIONS]
```

**Options:**
- `--portfolio-file FILE` - Portfolio file to load (.json or .pkl)
- `--prices-file FILE` - CSV file with current prices (columns: symbol, price)
- `--show-allocation` - Show portfolio allocation breakdown
- `--show-transactions` - Show recent transaction history
- `--show-metrics` - Show performance metrics
- `--output FILE, -o FILE` - Save portfolio to file (.json or .pkl)

**Examples:**
```bash
# Show demo portfolio with all details
python -m src.ui.cli show-portfolio \
  --show-allocation \
  --show-transactions \
  --show-metrics

# Load and display existing portfolio
python -m src.ui.cli show-portfolio \
  --portfolio-file my_portfolio.json \
  --show-allocation \
  --show-metrics

# Load portfolio with custom prices
python -m src.ui.cli show-portfolio \
  --portfolio-file portfolio.json \
  --prices-file current_prices.csv \
  --show-allocation \
  --output updated_portfolio.json
```

---

## General Options

All commands support:
- `--help, -h` - Show help message and exit
- `--version` - Show program version number

---

## Complete Workflow Examples

### Example 1: Full Trading Pipeline

```bash
# Step 1: Collect historical data
python -m src.ui.cli collect-data \
  --sources bist,fundamental \
  --symbols THYAO,GARAN \
  --start-date 2023-01-01 \
  --output data/

# Step 2: Train models
python -m src.ui.cli train-models \
  --model-types xgboost_regressor,lightgbm_regressor \
  --data-file data/training_data.csv \
  --tune-hyperparameters \
  --model-dir models/

# Step 3: Generate signals
python -m src.ui.cli run-signals \
  --symbols THYAO,GARAN \
  --min-confidence 0.6 \
  --output signals/current_signals.json

# Step 4: Backtest strategy
python -m src.ui.cli backtest \
  --symbol THYAO \
  --start-date 2023-01-01 \
  --initial-capital 100000 \
  --output backtest_results/

# Step 5: Check portfolio
python -m src.ui.cli show-portfolio \
  --portfolio-file portfolio.json \
  --show-allocation \
  --show-metrics
```

### Example 2: Quick Signal Generation

```bash
# Generate signals for top BIST stocks
python -m src.ui.cli run-signals \
  --symbols THYAO,GARAN,AKBNK,TUPRS,EREGL \
  --min-confidence 0.7 \
  --output signals_$(date +%Y%m%d).json
```

### Example 3: Comprehensive Backtesting

```bash
# Run extensive backtest with custom parameters
python -m src.ui.cli backtest \
  --symbol THYAO \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --initial-capital 100000 \
  --commission-rate 0.001 \
  --slippage-rate 0.0005 \
  --position-size 0.1 \
  --min-confidence 0.6 \
  --output backtest_results/thyao_2023/
```

---

## Technical Details

### Architecture

The CLI is built with:
- **argparse**: Python's standard library argument parser (no external dependencies)
- **Modular design**: Each command is implemented as a separate function
- **Integration**: Seamlessly integrates with existing system modules:
  - `src.signals.generator` - Signal generation
  - `src.backtesting.engine` - Backtesting
  - `src.models.trainer` - Model training
  - `src.data.collectors.*` - Data collection
  - `src.portfolio.manager` - Portfolio management

### File Structure

```
src/ui/cli.py
├── Command Functions
│   ├── cmd_run_signals()      # Signal generation
│   ├── cmd_backtest()         # Backtesting
│   ├── cmd_train_models()     # Model training
│   ├── cmd_collect_data()     # Data collection
│   └── cmd_show_portfolio()   # Portfolio display
├── Parser Setup
│   └── create_parser()        # Argument parser configuration
└── Main Entry Point
    └── main()                 # CLI entry point
```

### Error Handling

The CLI includes comprehensive error handling:
- Input validation
- File I/O error handling
- Module import error handling
- Graceful keyboard interrupt handling (Ctrl+C)
- Detailed error logging

---

## Development Notes

### Adding New Commands

To add a new command:

1. Create a command function:
```python
def cmd_new_command(args):
    """Command implementation"""
    # Your code here
```

2. Add parser configuration in `create_parser()`:
```python
parser_new = subparsers.add_parser('new-command', help='Description')
parser_new.add_argument('--option', type=str, help='Option help')
parser_new.set_defaults(func=cmd_new_command)
```

### Testing

Test individual commands:
```bash
# Test help
python -m src.ui.cli <command> --help

# Test execution
python -m src.ui.cli <command> [options]
```

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **File not found**: Use absolute paths or ensure you're running from the correct directory

3. **Permission denied**: Make the CLI executable:
   ```bash
   chmod +x src/ui/cli.py
   ```

---

## Version Information

- **CLI Version**: 1.0.0
- **Created**: 2025-11-16
- **Author**: BIST AI Trading System

---

## Support

For issues or feature requests, please refer to the main project documentation.
