# BIST Data Collector

A robust Python module for collecting OHLCV (Open, High, Low, Close, Volume) data from Borsa Istanbul (BIST) stocks using the yfinance library.

## Features

- **Multiple Timeframes**: Support for daily, hourly, 30-min, 15-min, 5-min, and 1-min data
- **BIST Symbol Handling**: Automatic `.IS` suffix management for BIST stocks
- **Error Handling**: Comprehensive error handling with retry logic and exponential backoff
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Data Validation**: Automatic validation of downloaded data
- **Batch Processing**: Fetch data for multiple stocks efficiently
- **Index Support**: Collect BIST index data (XU100, XU030, XU050, etc.)
- **Export Capabilities**: Easy CSV export functionality
- **Logging**: Comprehensive logging for debugging and monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Note: The main dependencies are:
- yfinance >= 0.2.28
- pandas >= 2.0.0
- numpy >= 1.24.0

## Quick Start

### Basic Usage

```python
from src.data.collectors.bist_collector import BISTCollector

# Initialize the collector
collector = BISTCollector()

# Get daily data for a single stock
df = collector.get_historical_data(
    symbol='THYAO',      # Turkish Airlines
    timeframe='daily',
    period='6mo'         # Last 6 months
)

print(df.tail())
```

### Multiple Timeframes

```python
# Hourly data
df_hourly = collector.get_historical_data(
    symbol='GARAN',
    timeframe='1h',
    period='5d'
)

# 30-minute data
df_30min = collector.get_historical_data(
    symbol='AKBNK',
    timeframe='30min',
    period='5d'
)

# 15-minute data
df_15min = collector.get_historical_data(
    symbol='ISCTR',
    timeframe='15min',
    period='5d'
)
```

### Multiple Stocks

```python
# Fetch data for multiple stocks
symbols = ['THYAO', 'GARAN', 'AKBNK', 'ISCTR', 'EREGL']

data_dict = collector.get_multiple_stocks(
    symbols=symbols,
    timeframe='daily',
    period='1mo',
    continue_on_error=True  # Continue if a stock fails
)

# Access individual stock data
thyao_data = data_dict['THYAO']
garan_data = data_dict['GARAN']
```

### Specific Date Range

```python
# Using specific start and end dates
df = collector.get_historical_data(
    symbol='EREGL',
    timeframe='daily',
    start_date='2024-01-01',
    end_date='2024-11-16'
)
```

### BIST Indices

```python
# Get BIST 100 index data
xu100 = collector.get_bist_index(
    index_name='XU100',
    timeframe='daily',
    period='1y'
)

# Other indices
xu030 = collector.get_bist_index('XU030', timeframe='daily', period='6mo')
xu050 = collector.get_bist_index('XU050', timeframe='daily', period='6mo')
```

### Latest Price Information

```python
# Get current price and market data
latest = collector.get_latest_price('THYAO')

print(f"Current Price: {latest['current_price']}")
print(f"Day High: {latest['day_high']}")
print(f"Day Low: {latest['day_low']}")
print(f"Volume: {latest['volume']}")
```

### Export to CSV

```python
# Export single stock data
df = collector.get_historical_data('THYAO', 'daily', period='1mo')
collector.export_to_csv(df, 'thyao_data.csv')

# Export multiple stocks
data_dict = collector.get_multiple_stocks(
    symbols=['THYAO', 'GARAN', 'AKBNK'],
    timeframe='daily',
    period='1mo'
)
collector.export_to_csv(data_dict, 'output_directory/')
```

## Advanced Configuration

### Custom Initialization

```python
collector = BISTCollector(
    rate_limit_delay=1.0,    # Wait 1 second between API calls
    validate_data=True,      # Enable data validation
    auto_adjust=True         # Auto-adjust for splits/dividends
)
```

### Retry Configuration

The retry decorator can be customized by modifying the `@retry_on_failure` parameters:

```python
@retry_on_failure(max_retries=5, delay=3, backoff=2.5)
def custom_fetch():
    # Your custom fetching logic
    pass
```

## Supported Timeframes

| Timeframe | Alias | Max Period | Description |
|-----------|-------|------------|-------------|
| `'daily'` | `'1d'` | ~730 days | Daily OHLCV |
| `'hourly'` | `'1h'` | ~730 days | Hourly OHLCV |
| `'30min'` | `'30m'` | ~60 days | 30-minute OHLCV |
| `'15min'` | `'15m'` | ~60 days | 15-minute OHLCV |
| `'5min'` | `'5m'` | ~60 days | 5-minute OHLCV |
| `'1min'` | `'1m'` | ~7 days | 1-minute OHLCV |

## Common BIST Stocks

### BIST 30 Stocks (Sample)

```python
BIST30_SAMPLE = [
    'THYAO',   # Türk Hava Yolları
    'GARAN',   # Garanti Bankası
    'AKBNK',   # Akbank
    'ISCTR',   # İş Bankası (C)
    'EREGL',   # Ereğli Demir Çelik
    'TUPRS',   # Tüpraş
    'SAHOL',   # Sabancı Holding
    'KCHOL',   # Koç Holding
    'PETKM',   # Petkim
    'ASELS',   # Aselsan
    'SISE',    # Şişe Cam
    'TTKOM',   # Türk Telekom
    'KOZAL',   # Koza Altın
    'TAVHL',   # TAV Havalimanları
    'BIMAS'    # BIM
]
```

## Error Handling

The collector includes comprehensive error handling:

```python
from src.data.collectors.bist_collector import (
    BISTCollectorError,
    APIError,
    DataValidationError
)

try:
    df = collector.get_historical_data('THYAO', 'daily', period='1mo')
except APIError as e:
    print(f"API Error: {e}")
except DataValidationError as e:
    print(f"Data Validation Error: {e}")
except BISTCollectorError as e:
    print(f"General Collector Error: {e}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_collector.py --all

# Run specific test
python test_collector.py --test 1

# Available tests:
# 1. Single Stock Daily
# 2. Single Stock Intraday
# 3. Multiple Stocks
# 4. BIST Indices
# 5. Latest Price
# 6. Date Range
# 7. Error Handling
```

## Data Structure

The returned DataFrame has the following structure:

```
Columns:
- Open: Opening price
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Volume: Trading volume
- Symbol: Stock symbol (without .IS suffix)

Index:
- Date: Datetime index with timezone information
```

Example:

```
                            Open    High     Low   Close      Volume  Symbol
Date
2024-11-11 00:00:00+03:00  123.45  125.30  122.80  124.50  15234567  THYAO
2024-11-12 00:00:00+03:00  124.60  126.20  123.90  125.80  18456789  THYAO
```

## Logging

The module uses Python's built-in logging. Configure it as needed:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Or for less verbose output
logging.basicConfig(level=logging.WARNING)
```

## Performance Tips

1. **Use appropriate timeframes**: Shorter timeframes have limited historical data
2. **Batch requests**: Use `get_multiple_stocks()` for multiple symbols
3. **Rate limiting**: Adjust `rate_limit_delay` based on your needs
4. **Caching**: Consider implementing caching for frequently accessed data
5. **Period limits**: Be aware of yfinance's period limitations for each timeframe

## Limitations

1. **Data Source**: Relies on Yahoo Finance API via yfinance
2. **Historical Data**: Limited by yfinance's data availability
3. **Real-time Data**: Small delay in data updates (typically a few minutes)
4. **Rate Limits**: Yahoo Finance may impose rate limits
5. **Intraday Data**: Limited historical range for high-frequency data

## Troubleshooting

### No data returned

```python
# Check if symbol is correct
collector = BISTCollector()
try:
    df = collector.get_historical_data('THYAO', 'daily', period='1mo')
    if df.empty:
        print("Empty DataFrame - check symbol and date range")
except Exception as e:
    print(f"Error: {e}")
```

### Rate limiting issues

```python
# Increase delay between requests
collector = BISTCollector(rate_limit_delay=2.0)
```

### Data validation errors

```python
# Disable validation if needed (not recommended)
collector = BISTCollector(validate_data=False)
```

## Contributing

When contributing to this module:

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include error handling
4. Add tests for new features
5. Update documentation

## License

Part of the BIST AI Trading System project.

## Contact

For issues and questions, please refer to the main project documentation.

---

**Note**: This collector is designed for the BIST AI Trading System but can be used as a standalone module for any BIST data collection needs.
