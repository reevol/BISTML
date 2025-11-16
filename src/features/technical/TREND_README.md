# Trend Indicators Module

## Overview

The `trend.py` module provides comprehensive trend analysis indicators for technical analysis of financial time series data, specifically designed for the Borsa Istanbul (BIST) equity market.

## Features

- **Multiple Indicators**: SMA, EMA, WMA, HMA, and Ichimoku Cloud
- **Multiple Timeframes**: Calculate indicators across different time periods simultaneously
- **Flexible Backend**: Supports pandas-ta, TA-Lib, or pure NumPy implementations
- **Production Ready**: Comprehensive error handling, logging, and validation

## Installation Requirements

```bash
# Required
pip install pandas numpy

# Optional (for better performance)
pip install pandas-ta
pip install ta-lib
```

## Quick Start

### Basic Usage

```python
from src.features.technical.trend import TrendIndicators
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Initialize calculator
calc = TrendIndicators(data, price_column='close')

# Calculate indicators
sma_20 = calc.sma(period=20)
ema_12 = calc.ema(period=12)
wma_20 = calc.wma(period=20)
hma_9 = calc.hma(period=9)
ichimoku = calc.ichimoku()
```

### Convenience Functions

```python
from src.features.technical.trend import (
    calculate_sma,
    calculate_ema,
    calculate_ichimoku
)

# Quick calculations
sma = calculate_sma(data, period=20)
ema = calculate_ema(data, timeframes=[12, 26, 50])
ichimoku = calculate_ichimoku(data)
```

## Indicators

### 1. Simple Moving Average (SMA)

The arithmetic mean of prices over a specified period.

```python
# Single period
sma_20 = calc.sma(period=20)

# Multiple periods
sma_multi = calc.sma(timeframes=[10, 20, 50, 200])
```

**Common periods:**
- Short-term: 10, 20
- Medium-term: 50, 100
- Long-term: 200

### 2. Exponential Moving Average (EMA)

Gives more weight to recent prices, making it more responsive to new information.

```python
# Single period
ema_12 = calc.ema(period=12)

# Multiple periods
ema_multi = calc.ema(timeframes=[8, 12, 21, 26, 50])
```

**Common periods:**
- Fast: 8, 12
- Medium: 21, 26
- Slow: 50, 100, 200

### 3. Weighted Moving Average (WMA)

Assigns linearly increasing weights to recent prices.

```python
# Single period
wma_20 = calc.wma(period=20)

# Multiple periods
wma_multi = calc.wma(timeframes=[10, 20, 30])
```

### 4. Hull Moving Average (HMA)

Reduces lag while maintaining smoothness. Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

```python
# Single period
hma_16 = calc.hma(period=16)

# Multiple periods
hma_multi = calc.hma(timeframes=[9, 16, 25])
```

**Common periods:** 9, 16, 25, 49

### 5. Ichimoku Cloud

A comprehensive indicator system providing support/resistance, trend direction, and momentum.

```python
ichimoku = calc.ichimoku(
    tenkan_period=9,      # Conversion Line
    kijun_period=26,      # Base Line
    senkou_b_period=52,   # Leading Span B
    displacement=26        # Cloud displacement
)

# Access components
tenkan = ichimoku['tenkan_sen']
kijun = ichimoku['kijun_sen']
span_a = ichimoku['senkou_span_a']
span_b = ichimoku['senkou_span_b']
chikou = ichimoku['chikou_span']
```

**Components:**
- **Tenkan-sen** (Conversion Line): (9-period high + 9-period low) / 2
- **Kijun-sen** (Base Line): (26-period high + 26-period low) / 2
- **Senkou Span A**: (Tenkan + Kijun) / 2, shifted +26 periods
- **Senkou Span B**: (52-period high + 52-period low) / 2, shifted +26 periods
- **Chikou Span**: Close price shifted -26 periods

**Interpretation:**
- Price above cloud = Bullish
- Price below cloud = Bearish
- Green cloud (Span A > Span B) = Bullish
- Red cloud (Span A < Span B) = Bearish

## Multi-Timeframe Analysis

Calculate multiple indicators across different timeframes:

```python
analysis = calc.multi_timeframe_analysis(
    indicators=['sma', 'ema', 'wma', 'hma'],
    timeframes={
        'sma': [20, 50, 200],
        'ema': [12, 26, 50],
        'wma': [10, 20],
        'hma': [9, 16]
    }
)
```

## Trend Signals

Generate trading signals based on moving average crossovers:

```python
# Golden Cross / Death Cross signals
signals = calc.get_trend_signal(
    fast_period=50,
    slow_period=200,
    indicator='sma'
)

# Returns: 1 (bullish), -1 (bearish), 0 (neutral)
```

## Backend Selection

The module automatically selects the best available backend:

```python
# Auto-select (prefers pandas-ta, then talib, then numpy)
calc = TrendIndicators(data, backend='auto')

# Force specific backend
calc = TrendIndicators(data, backend='pandas_ta')
calc = TrendIndicators(data, backend='talib')
calc = TrendIndicators(data, backend='numpy')
```

## Common Trading Strategies

### Golden Cross / Death Cross

```python
sma_50 = calc.sma(period=50)
sma_200 = calc.sma(period=200)

# Golden Cross: 50 SMA crosses above 200 SMA (bullish)
# Death Cross: 50 SMA crosses below 200 SMA (bearish)
```

### MACD with EMA

```python
ema_12 = calc.ema(period=12)
ema_26 = calc.ema(period=26)
macd_line = ema_12 - ema_26
signal_line = macd_line.ewm(span=9).mean()
```

### Ichimoku Strategy

```python
ichimoku = calc.ichimoku()

# Strong bullish: Price above cloud + Tenkan > Kijun + Green cloud
# Strong bearish: Price below cloud + Tenkan < Kijun + Red cloud
```

## Performance Considerations

- **pandas-ta**: Good balance of speed and features
- **TA-Lib**: Fastest, but requires C library installation
- **NumPy**: Pure Python, slowest but always available

For production systems with large datasets, install pandas-ta or TA-Lib.

## Error Handling

The module includes comprehensive error handling:

```python
try:
    sma = calc.sma(period=20)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

## Examples

See `trend_examples.py` for comprehensive usage examples:

```bash
cd src/features/technical
python trend_examples.py
```

## API Reference

### TrendIndicators Class

```python
class TrendIndicators:
    def __init__(data, price_column='close', backend='auto')
    def sma(data, period, column, timeframes)
    def ema(data, period, column, timeframes, adjust)
    def wma(data, period, column, timeframes)
    def hma(data, period, column, timeframes)
    def ichimoku(data, tenkan_period, kijun_period, senkou_b_period, displacement)
    def multi_timeframe_analysis(data, indicators, timeframes, column)
    def get_trend_signal(data, fast_period, slow_period, indicator)
```

### Convenience Functions

```python
calculate_sma(data, period, column, timeframes)
calculate_ema(data, period, column, timeframes)
calculate_wma(data, period, column, timeframes)
calculate_hma(data, period, column, timeframes)
calculate_ichimoku(data, tenkan_period, kijun_period, senkou_b_period, displacement)
calculate_trend_signals(data, fast_period, slow_period, indicator, column)
```

## Testing

Run the module directly to execute built-in tests:

```bash
python trend.py
```

## Integration with BIST Trading System

```python
from src.features.technical import TrendIndicators
from src.data.collectors.bist_collector import BISTCollector

# Collect data
collector = BISTCollector()
data = collector.get_stock_data('THYAO', period='1y')

# Calculate indicators
calc = TrendIndicators(data)
signals = calc.get_trend_signal(fast_period=12, slow_period=26)

# Use in trading strategy
if signals.iloc[-1] == 1:
    print("BUY signal")
elif signals.iloc[-1] == -1:
    print("SELL signal")
```

## License

Part of the BIST AI Trading System project.

## Support

For issues or questions, please refer to the main project documentation.
