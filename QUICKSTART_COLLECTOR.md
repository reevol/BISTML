# BIST Collector Quick Start Guide

## Installation

```bash
# Install dependencies
pip install yfinance pandas numpy

# Or install all project dependencies
pip install -r requirements.txt
```

## 5-Minute Quick Start

### 1. Import and Initialize

```python
from src.data.collectors.bist_collector import BISTCollector

collector = BISTCollector()
```

### 2. Get Daily Data

```python
# Single stock - daily data
df = collector.get_historical_data(
    symbol='THYAO',      # Turkish Airlines
    timeframe='daily',   # or '1d'
    period='1mo'         # Last month
)
```

### 3. Get Intraday Data

```python
# Hourly data
df_1h = collector.get_historical_data('GARAN', '1h', period='5d')

# 30-minute data
df_30m = collector.get_historical_data('AKBNK', '30min', period='5d')

# 15-minute data
df_15m = collector.get_historical_data('ISCTR', '15min', period='5d')
```

### 4. Multiple Stocks

```python
symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'TUPRS']
data = collector.get_multiple_stocks(symbols, 'daily', period='1mo')

# Access individual stock data
thyao_df = data['THYAO']
```

### 5. BIST Indices

```python
xu100 = collector.get_bist_index('XU100', 'daily', period='6mo')
xu030 = collector.get_bist_index('XU030', 'daily', period='6mo')
```

## Common Patterns

### Date Range Query

```python
df = collector.get_historical_data(
    symbol='EREGL',
    timeframe='daily',
    start_date='2024-01-01',
    end_date='2024-11-16'
)
```

### Export to CSV

```python
# Single file
collector.export_to_csv(df, 'thyao_data.csv')

# Multiple files (creates directory)
collector.export_to_csv(data_dict, 'output_data/')
```

### Latest Price

```python
latest = collector.get_latest_price('THYAO')
print(f"Price: {latest['current_price']}")
print(f"Volume: {latest['volume']}")
```

## Popular BIST Stocks

```python
# Banking
banks = ['GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'HALKB']

# Industrials  
industrials = ['EREGL', 'TUPRS', 'PETKM', 'SISE', 'THYAO']

# Technology & Defense
tech = ['ASELS', 'TTKOM']

# Holdings
holdings = ['SAHOL', 'KCHOL', 'TAVHL']
```

## Timeframe Reference

| Code | Description | Max Period |
|------|-------------|------------|
| `daily` or `1d` | Daily | ~2 years |
| `hourly` or `1h` | Hourly | ~2 years |
| `30min` or `30m` | 30 minutes | 60 days |
| `15min` or `15m` | 15 minutes | 60 days |
| `5min` or `5m` | 5 minutes | 60 days |

## Error Handling

```python
try:
    df = collector.get_historical_data('THYAO', 'daily', period='1mo')
except Exception as e:
    print(f"Error: {e}")
```

## Running Examples

```bash
# Simple usage example
python example_usage.py

# Run all tests
python test_collector.py --all

# Run specific test
python test_collector.py --test 1
```

## Tips

1. Use `period='1mo'` for monthly data, `'1y'` for yearly
2. Symbol can be with or without `.IS` suffix (auto-handled)
3. Set `rate_limit_delay=1.0` if you hit rate limits
4. Use `continue_on_error=True` when fetching multiple stocks
5. Check logs for detailed error messages

## Getting Help

- Full documentation: `README_COLLECTOR.md`
- Test examples: `test_collector.py`
- Usage examples: `example_usage.py`

---

**Need more info?** See the complete documentation in README_COLLECTOR.md
