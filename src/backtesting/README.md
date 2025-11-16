# Backtesting Engine - BIST AI Trading System

## Overview

The backtesting engine provides a comprehensive framework for evaluating trading strategies on historical data. It simulates trades based on historical signals, tracks equity curves, handles realistic transaction costs and slippage, and calculates detailed performance metrics.

## Features

- **Event-driven backtesting architecture**: Processes data chronologically to avoid look-ahead bias
- **Realistic transaction cost modeling**: Commission rates and slippage simulation
- **Equity curve tracking**: Monitor portfolio value over time
- **Comprehensive performance metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Calmar Ratio
  - And many more...
- **Trade-level analytics**: Track MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
- **Position sizing strategies**: Fixed amount, percentage of equity, signal strength-based, risk-based
- **Stop loss and take profit**: Automatic risk management
- **Benchmark comparison**: Compare strategy performance against benchmark (e.g., BIST 100)
- **Results export**: JSON, CSV, and text report formats

## Installation

The backtesting module is part of the BIST AI Trading System. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Backtest

```python
from src.backtesting import BacktestEngine, BacktestConfig, quick_backtest
import pandas as pd

# Load your historical price data
price_data = pd.read_csv('historical_prices.csv')
# Required columns: date, symbol, open, high, low, close, volume

# Load your trading signals
signals = pd.read_csv('trading_signals.csv')
# Required columns: date, symbol, signal, confidence
# signal values: 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'

# Run quick backtest with default settings
results = quick_backtest(
    price_data=price_data,
    signals=signals,
    initial_capital=100000.0,
    commission_rate=0.001  # 0.1%
)

# Display results
print(results)
```

### Advanced Configuration

```python
from src.backtesting import (
    BacktestEngine,
    BacktestConfig,
    PositionSizing,
    SlippageModel,
    export_results
)

# Create custom configuration
config = BacktestConfig(
    initial_capital=100000.0,
    commission_rate=0.001,              # 0.1% commission
    slippage_model=SlippageModel.FIXED_PERCENT,
    slippage_rate=0.0005,               # 0.05% slippage
    position_sizing=PositionSizing.PERCENT_EQUITY,
    position_size_value=0.20,           # 20% of equity per trade
    max_positions=10,                   # Maximum concurrent positions
    allow_shorting=False,
    use_stop_loss=True,
    use_take_profit=True,
    min_confidence=0.6,                 # Only trade signals with 60%+ confidence
    risk_free_rate=0.15                 # 15% annual (Turkey)
)

# Initialize engine
engine = BacktestEngine(config)

# Run backtest
results = engine.run(
    price_data=price_data,
    signals=signals,
    benchmark_data=benchmark_prices  # Optional
)

# Export results
export_results(results, output_dir='backtest_results')
```

## Configuration Options

### Position Sizing Methods

- `FIXED_AMOUNT`: Fixed cash amount per trade
- `FIXED_SHARES`: Fixed number of shares
- `PERCENT_EQUITY`: Percentage of current equity (recommended)
- `SIGNAL_STRENGTH`: Scale position by signal confidence
- `RISK_BASED`: Size based on risk per trade

### Slippage Models

- `NONE`: No slippage
- `FIXED_PERCENT`: Fixed percentage slippage (default)
- `VOLUME_BASED`: Based on volume and order size (future)
- `BID_ASK_SPREAD`: Based on bid-ask spread (future)

## Results Analysis

### Performance Metrics

The `BacktestResults` object provides comprehensive metrics:

```python
# Basic metrics
print(f"Total Return: {results.total_return_pct:.2f}%")
print(f"Annualized Return: {results.annualized_return:.2f}%")
print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
print(f"Max Drawdown: {results.max_drawdown:.2f}%")

# Trade statistics
print(f"Win Rate: {results.win_rate:.2f}%")
print(f"Profit Factor: {results.profit_factor:.3f}")
print(f"Average Win: {results.avg_win:,.2f} TRY")
print(f"Average Loss: {results.avg_loss:,.2f} TRY")

# Benchmark comparison (if provided)
if results.benchmark_return:
    print(f"Benchmark Return: {results.benchmark_return:.2f}%")
    print(f"Alpha: {results.alpha:.2f}%")
    print(f"Beta: {results.beta:.3f}")
```

### Equity Curve

```python
# Access equity curve data
equity_curve = results.equity_curve

# Plot equity curve (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(equity_curve.index, equity_curve['equity'], label='Portfolio Value')
plt.plot(equity_curve.index, equity_curve['peak'], label='Peak Value', linestyle='--')
plt.fill_between(
    equity_curve.index,
    equity_curve['equity'],
    equity_curve['peak'],
    alpha=0.3,
    color='red',
    label='Drawdown'
)
plt.xlabel('Date')
plt.ylabel('Portfolio Value (TRY)')
plt.title('Equity Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Trade Analysis

```python
# Access individual trades
for trade in results.trades[:5]:  # First 5 trades
    print(f"\n{trade.symbol}:")
    print(f"  Entry: {trade.entry_date.date()} @ {trade.entry_price:.2f} TRY")
    print(f"  Exit: {trade.exit_date.date()} @ {trade.exit_price:.2f} TRY")
    print(f"  P&L: {trade.pnl:,.2f} TRY ({trade.pnl_percent:.2f}%)")
    print(f"  Holding: {trade.holding_period} days")
    print(f"  MAE: {trade.mae:.2f}% | MFE: {trade.mfe:.2f}%")
    print(f"  Exit reason: {trade.exit_reason}")

# Convert trades to DataFrame for analysis
import pandas as pd
trades_df = pd.DataFrame([t.to_dict() for t in results.trades])

# Analyze by signal type
win_rate_by_signal = trades_df.groupby('entry_signal').apply(
    lambda x: (x['pnl'] > 0).sum() / len(x) * 100
)
print("\nWin Rate by Signal Type:")
print(win_rate_by_signal)

# Analyze holding period vs profit
avg_pnl_by_holding = trades_df.groupby(
    pd.cut(trades_df['holding_period'], bins=[0, 1, 5, 10, 30, 100])
)['pnl'].mean()
print("\nAverage P&L by Holding Period:")
print(avg_pnl_by_holding)
```

### Export Results

```python
from src.backtesting import export_results

# Export all results to directory
export_results(results, output_dir='my_backtest_results')

# This creates:
# - backtest_summary.json: Overall metrics
# - equity_curve.csv: Daily equity data
# - trades.csv: All trade details
# - backtest_report.txt: Formatted text report
```

## Integration with Signal Generator

The backtesting engine integrates seamlessly with the signal generator:

```python
from src.signals import SignalGenerator, create_model_output
from src.backtesting import BacktestEngine, BacktestConfig

# Create signal generator
signal_generator = SignalGenerator(
    enable_dynamic_thresholds=True,
    risk_adjustment=True
)

# Define a function to generate signals on-the-fly
def generate_signal(symbol, date, prices):
    """Generate trading signal for a given date"""
    # Your model predictions here
    model_outputs = [
        create_model_output(
            model_name='lstm',
            model_type='regression',
            prediction=predicted_price,
            confidence=0.75
        ),
        # ... more model outputs
    ]

    # Generate signal
    signal = signal_generator.generate_signal(
        stock_code=symbol,
        model_outputs=model_outputs,
        current_price=prices.loc[prices['symbol'] == symbol, 'close'].iloc[0]
    )

    return signal.signal.name, signal.confidence_score, signal.target_price

# Run backtest with signal generator
config = BacktestConfig(initial_capital=100000)
engine = BacktestEngine(config)

results = engine.run(
    price_data=price_data,
    signals=None,  # No pre-computed signals
    signal_generator=generate_signal  # Generate on-the-fly
)
```

## Example: Moving Average Crossover Strategy

```python
import pandas as pd
import numpy as np
from src.backtesting import BacktestEngine, BacktestConfig, PositionSizing

# Load historical data
price_data = pd.read_csv('THYAO_historical.csv')
price_data['date'] = pd.to_datetime(price_data['date'])
price_data.set_index('date', inplace=True)

# Generate signals using simple MA crossover
price_data['sma_20'] = price_data['close'].rolling(20).mean()
price_data['sma_50'] = price_data['close'].rolling(50).mean()

signals_list = []
for date, row in price_data.iterrows():
    if pd.isna(row['sma_50']):
        continue

    if row['sma_20'] > row['sma_50']:
        signal = 'BUY'
        confidence = min(0.9, (row['sma_20'] - row['sma_50']) / row['sma_50'] * 10)
    elif row['sma_20'] < row['sma_50']:
        signal = 'SELL'
        confidence = min(0.9, (row['sma_50'] - row['sma_20']) / row['sma_20'] * 10)
    else:
        continue

    signals_list.append({
        'date': date,
        'symbol': 'THYAO',
        'signal': signal,
        'confidence': confidence
    })

signals = pd.DataFrame(signals_list)
signals.set_index('date', inplace=True)

# Configure and run backtest
config = BacktestConfig(
    initial_capital=100000,
    commission_rate=0.001,
    position_sizing=PositionSizing.PERCENT_EQUITY,
    position_size_value=0.5,  # Use 50% of equity per trade
    use_stop_loss=True,
    use_take_profit=True,
    min_confidence=0.5
)

engine = BacktestEngine(config)
results = engine.run(price_data, signals)

print(results)
```

## Best Practices

1. **Avoid Look-Ahead Bias**: Ensure signals are generated only using data available at that point in time
2. **Realistic Costs**: Use accurate commission and slippage rates for your broker
3. **Position Sizing**: Don't over-allocate to single positions (typically 10-20% max)
4. **Risk Management**: Always use stop losses in live trading
5. **Walk-Forward Testing**: Test on out-of-sample data to avoid overfitting
6. **Multiple Timeframes**: Test strategies across different time periods
7. **Benchmark Comparison**: Always compare against a buy-and-hold benchmark

## Performance Considerations

- For large datasets (>10,000 trading days), consider:
  - Using vectorized operations where possible
  - Filtering signals to reduce unnecessary computations
  - Running backtests in parallel for multiple strategies
  - Storing intermediate results to disk

## Troubleshooting

### Common Issues

1. **"No position in symbol" error**
   - Ensure signals DataFrame has correct 'symbol' column
   - Check that price_data and signals have matching symbols

2. **Very low number of trades**
   - Check `min_confidence` threshold
   - Verify signal generation logic
   - Ensure `max_positions` is not too restrictive

3. **Unrealistic returns**
   - Verify commission and slippage settings
   - Check for look-ahead bias in signals
   - Review position sizing configuration

4. **Memory issues with large datasets**
   - Process data in chunks
   - Reduce granularity (e.g., daily instead of hourly)
   - Use date range filtering

## Future Enhancements

Planned features:
- Volume-based slippage model
- Multi-asset portfolio optimization
- Options and futures support
- Monte Carlo simulation
- Walk-forward optimization framework
- Real-time paper trading mode

## Support

For issues, questions, or contributions, please refer to the main project documentation.

## License

Part of the BIST AI Trading System.

---

Last Updated: 2025-11-16
