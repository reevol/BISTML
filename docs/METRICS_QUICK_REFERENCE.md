# Backtesting Metrics Module - Quick Reference

## Overview

The `src/backtesting/metrics.py` module provides comprehensive performance metrics for backtesting trading strategies. It calculates all industry-standard metrics including risk-adjusted returns, drawdown analysis, and trade statistics.

**File Location:** `/home/user/BISTML/src/backtesting/metrics.py`
**Lines of Code:** 996
**Functions:** 15 public functions + utility functions
**Dependencies:** numpy, pandas

---

## Core Metrics

### 1. Win Rate
```python
from src.backtesting import calculate_win_rate

win_rate = calculate_win_rate(returns)
# Returns: Percentage of profitable trades (0-100)
```

### 2. Average Profit/Loss
```python
from src.backtesting import calculate_average_profit_loss

avg_metrics = calculate_average_profit_loss(returns)
# Returns: Dict with avg_profit, avg_loss, avg_profit_loss
```

### 3. Maximum Drawdown
```python
from src.backtesting import calculate_max_drawdown

max_dd = calculate_max_drawdown(equity_curve)
# Returns: Maximum drawdown as decimal (e.g., 0.15 = 15%)

# With drawdown series
max_dd, dd_series = calculate_max_drawdown(equity_curve, return_series=True)
```

### 4. Sharpe Ratio
```python
from src.backtesting import calculate_sharpe_ratio

sharpe = calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.02,  # 2% annual
    periods_per_year=252,  # Daily trading
    annualize=True
)
# Returns: Annualized Sharpe ratio
```

### 5. Sortino Ratio
```python
from src.backtesting import calculate_sortino_ratio

sortino = calculate_sortino_ratio(
    returns,
    risk_free_rate=0.02,
    periods_per_year=252,
    annualize=True
)
# Returns: Annualized Sortino ratio (downside risk-adjusted)
```

### 6. Calmar Ratio
```python
from src.backtesting import calculate_calmar_ratio

calmar = calculate_calmar_ratio(
    returns,
    equity_curve=None,  # Optional, calculated if not provided
    periods_per_year=252
)
# Returns: Annualized return / Maximum drawdown
```

### 7. Profit Factor
```python
from src.backtesting import calculate_profit_factor

pf = calculate_profit_factor(returns)
# Returns: Gross profits / Gross losses (>1 is profitable)
```

### 8. Recovery Factor
```python
from src.backtesting import calculate_recovery_factor

rf = calculate_recovery_factor(returns, equity_curve=None)
# Returns: Net profit / Maximum drawdown
```

---

## Comprehensive Analysis

### Calculate All Metrics at Once

```python
from src.backtesting import calculate_all_metrics
import numpy as np

# Your trading returns
returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

# Calculate all metrics
metrics = calculate_all_metrics(
    returns=returns,
    equity_curve=None,  # Optional, calculated if not provided
    risk_free_rate=0.02,  # 2% annual risk-free rate
    periods_per_year=252,  # Daily trading (252), Monthly (12), etc.
    initial_capital=100000.0
)

# Print comprehensive summary
print(metrics)

# Access individual metrics
print(f"Win Rate: {metrics.win_rate:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
print(f"Profit Factor: {metrics.profit_factor:.3f}")
print(f"Total Return: {metrics.total_return_pct:.2f}%")
```

### PerformanceMetrics Object

The `PerformanceMetrics` dataclass contains:

**Trade Statistics:**
- `total_trades` - Total number of trades
- `winning_trades` - Number of winning trades
- `losing_trades` - Number of losing trades
- `win_rate` - Win rate percentage
- `max_consecutive_wins` - Maximum consecutive winning trades
- `max_consecutive_losses` - Maximum consecutive losing trades

**Return Metrics:**
- `total_return` - Total return (decimal)
- `total_return_pct` - Total return (percentage)
- `annualized_return` - Annualized return
- `annualized_volatility` - Annualized volatility
- `avg_profit` - Average profit per winning trade
- `avg_loss` - Average loss per losing trade
- `avg_profit_loss` - Average profit/loss across all trades
- `largest_win` - Largest winning trade
- `largest_loss` - Largest losing trade
- `expectancy` - Expected value per trade

**Risk Metrics:**
- `max_drawdown` - Maximum drawdown (decimal)
- `max_drawdown_pct` - Maximum drawdown (percentage)
- `sharpe_ratio` - Sharpe ratio (annualized)
- `sortino_ratio` - Sortino ratio (annualized)
- `calmar_ratio` - Calmar ratio
- `profit_factor` - Profit factor
- `recovery_factor` - Recovery factor
- `risk_reward_ratio` - Average win to average loss ratio

---

## Utility Functions

### Rolling Metrics

```python
from src.backtesting import rolling_sharpe_ratio, rolling_max_drawdown
import pandas as pd

# Calculate rolling Sharpe ratio
rolling_sharpe = rolling_sharpe_ratio(
    returns=returns_series,
    window=30,  # 30-period window
    risk_free_rate=0.02,
    periods_per_year=252
)

# Calculate rolling maximum drawdown
rolling_dd = rolling_max_drawdown(
    equity_curve=equity_series,
    window=30
)
```

### Compare Multiple Strategies

```python
from src.backtesting import compare_strategies

# Returns from different strategies
strategies = {
    'Strategy A': strategy_a_returns,
    'Strategy B': strategy_b_returns,
    'Strategy C': strategy_c_returns
}

# Compare all strategies
comparison = compare_strategies(
    strategies_returns=strategies,
    risk_free_rate=0.02,
    periods_per_year=252
)

# View comparison DataFrame
print(comparison[['sharpe_ratio', 'max_drawdown_pct', 'win_rate']])
```

---

## Export Options

### Export to Dictionary

```python
metrics_dict = metrics.to_dict()
# Returns: Dictionary with all metrics
```

### Export to DataFrame

```python
metrics_df = metrics.to_dataframe()
# Returns: pandas DataFrame with metrics as rows

# Save to CSV
metrics_df.to_csv('backtest_metrics.csv')
```

### Print Summary

```python
print(metrics)
# Prints formatted summary with all metrics
```

---

## Quick Start Examples

### Example 1: Basic Usage

```python
from src.backtesting import calculate_all_metrics
import numpy as np

# Sample trading returns (as decimals, e.g., 0.02 = 2%)
returns = np.array([0.025, -0.010, 0.030, -0.015, 0.020])

# Calculate metrics
metrics = calculate_all_metrics(returns, risk_free_rate=0.02)

# Print summary
print(metrics)
```

### Example 2: With Equity Curve

```python
import pandas as pd

# Your equity curve
equity = [100000, 102500, 101500, 104500, 103000, 105000]

# Your returns
returns = pd.Series(equity).pct_change().dropna()

# Calculate metrics
metrics = calculate_all_metrics(
    returns=returns,
    equity_curve=equity,
    risk_free_rate=0.02,
    periods_per_year=252
)
```

### Example 3: Strategy Comparison

```python
from src.backtesting import compare_strategies
import numpy as np

strategy_1 = np.random.normal(0.015, 0.02, 100)
strategy_2 = np.random.normal(0.010, 0.01, 100)

comparison = compare_strategies({
    'Aggressive': strategy_1,
    'Conservative': strategy_2
})

print(comparison)
```

---

## Common Use Cases

### Evaluating a Trading Strategy

```python
# After running a backtest, you have trade returns
trade_returns = [0.02, -0.01, 0.03, -0.02, 0.01, ...]

# Calculate comprehensive metrics
metrics = calculate_all_metrics(
    returns=trade_returns,
    risk_free_rate=0.02,  # Turkey TCMB rate or US Treasury
    periods_per_year=252   # Daily trades
)

# Check if strategy is profitable
if metrics.sharpe_ratio > 1.0 and metrics.profit_factor > 1.5:
    print("Strategy shows promise!")
    print(f"Sharpe: {metrics.sharpe_ratio:.3f}")
    print(f"Max DD: {metrics.max_drawdown_pct:.2f}%")
```

### Risk Assessment

```python
# Focus on risk metrics
metrics = calculate_all_metrics(returns)

print("Risk Assessment:")
print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
print(f"Largest Loss: {metrics.largest_loss * 100:.2f}%")
print(f"Recovery Factor: {metrics.recovery_factor:.3f}")

# Is the risk acceptable?
if metrics.max_drawdown_pct > 30:
    print("WARNING: High drawdown risk!")
```

### Comparing to Benchmark

```python
# Your strategy returns
strategy_returns = [...]

# BIST 100 benchmark returns
benchmark_returns = [...]

# Compare
comparison = compare_strategies({
    'My Strategy': strategy_returns,
    'BIST 100': benchmark_returns
})

# Check if you're beating the benchmark
if comparison.loc['My Strategy', 'sharpe_ratio'] > comparison.loc['BIST 100', 'sharpe_ratio']:
    print("Outperforming benchmark on risk-adjusted basis!")
```

---

## Parameters Reference

### Periods Per Year

Common values for `periods_per_year`:
- **Daily trading:** 252 (trading days per year)
- **Weekly trading:** 52
- **Monthly trading:** 12
- **Hourly trading (BIST):** ~2,000 (252 days × ~8 hours)
- **30-minute intervals:** ~4,000

### Risk-Free Rate

For Turkey (BIST):
- Use current TCMB (Turkish Central Bank) policy rate
- Convert to annual decimal (e.g., 15% = 0.15)

For international comparison:
- US: 10-year Treasury yield
- Typical range: 0.01 - 0.05 (1% - 5%)

---

## Integration with Backtesting Engine

The metrics module integrates seamlessly with the backtesting engine:

```python
from src.backtesting import BacktestEngine, calculate_all_metrics

# Run backtest
engine = BacktestEngine(config)
results = engine.run(data)

# Calculate metrics from results
metrics = calculate_all_metrics(
    returns=results.returns,
    equity_curve=results.equity_curve,
    risk_free_rate=0.02
)

print(metrics)
```

---

## Error Handling

```python
from src.backtesting import (
    calculate_all_metrics,
    InsufficientDataError,
    InvalidDataError,
    MetricsError
)

try:
    metrics = calculate_all_metrics(returns)
except InsufficientDataError:
    print("Not enough data for metrics calculation")
except InvalidDataError:
    print("Invalid data format")
except MetricsError as e:
    print(f"Metrics calculation error: {e}")
```

---

## Best Practices

1. **Minimum Data:** Have at least 30 trades for meaningful statistics
2. **Annualization:** Always specify correct `periods_per_year`
3. **Risk-Free Rate:** Use current rates for accurate Sharpe/Sortino
4. **Equity Curve:** Provide actual equity curve when available
5. **Comparison:** Always compare against benchmark (BIST 100)

---

## Additional Resources

- **Full Examples:** `/home/user/BISTML/examples/backtesting_metrics_example.py`
- **Module Source:** `/home/user/BISTML/src/backtesting/metrics.py`
- **API Documentation:** Run `help(calculate_all_metrics)` in Python

---

## Summary

The metrics module provides:
- ✅ **8 core metrics:** Win rate, avg profit/loss, max drawdown, Sharpe, Sortino, Calmar, profit factor, recovery factor
- ✅ **Comprehensive analysis:** 25+ metrics in one function call
- ✅ **Flexible input:** Supports Series, arrays, lists
- ✅ **Export options:** Dictionary, DataFrame, CSV
- ✅ **Rolling metrics:** Calculate metrics over time windows
- ✅ **Strategy comparison:** Compare multiple strategies
- ✅ **Production-ready:** Type hints, error handling, logging

For support or questions, refer to the module docstrings or example file.
