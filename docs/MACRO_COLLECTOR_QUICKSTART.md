# Macro Collector Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install fredapi evds yfinance pandas numpy
```

### Step 2: Get API Keys

#### FRED API (Required for global indicators)
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill in your information (free account)
4. Copy your API key

#### EVDS API (Required for Turkish indicators)
1. Go to https://evds2.tcmb.gov.tr/
2. Register for an account
3. Navigate to user settings
4. Copy your API key

### Step 3: Set Environment Variables

**Linux/Mac:**
```bash
export FRED_API_KEY='your_fred_api_key_here'
export EVDS_API_KEY='your_evds_api_key_here'
```

**Windows (Command Prompt):**
```cmd
set FRED_API_KEY=your_fred_api_key_here
set EVDS_API_KEY=your_evds_api_key_here
```

**Or use .env file:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys
nano .env
```

### Step 4: Test It!

```python
from src.data.collectors.macro_collector import MacroCollector

# Initialize
collector = MacroCollector()

# Fetch S&P 500 (last 30 days)
sp500 = collector.get_indicator('SP500')
print(sp500.tail())
```

## 📊 Common Use Cases

### 1. Get Latest Market Data

```python
from datetime import datetime, timedelta
from src.data.collectors.macro_collector import MacroCollector

collector = MacroCollector()

# Get last 7 days of data
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

# Fetch multiple indicators
data = collector.get_multiple_indicators(
    ['SP500', 'DAX', 'VIX'],
    start,
    end
)

print(data)
```

**Output:**
```
            SP500      DAX     VIX
2024-11-09  4500.2  15234.5  18.4
2024-11-10  4512.8  15289.1  17.9
...
```

### 2. Analyze Turkish Economy

```python
# Requires EVDS_API_KEY
turkish_data = collector.get_all_turkish_indicators(
    start_date='2024-01-01',
    end_date='2024-11-01'
)

print(turkish_data[['TR_CPI', 'TR_INTEREST_RATE']].tail())
```

### 3. Generate ML Features

```python
# Get comprehensive features for machine learning
features = collector.get_macro_features(
    start_date='2023-01-01',
    end_date='2024-11-01',
    include_derived=True  # Adds returns, moving averages, etc.
)

print(f"Generated {len(features.columns)} features")
print(features.columns.tolist()[:10])  # First 10 features
```

**Output:**
```
Generated 84 features
['SP500', 'DAX', 'VIX', 'US_INTEREST_RATE', 'TR_CPI', 'TR_PPI',
 'SP500_return', 'SP500_mom', 'SP500_yoy', 'SP500_ma30']
```

### 4. Correlation Analysis

```python
# Analyze relationships between indicators
data = collector.get_multiple_indicators(
    ['SP500', 'VIX', 'US_INTEREST_RATE'],
    start_date='2023-01-01'
)

# Calculate correlations
correlations = data.pct_change().corr()
print(correlations)
```

**Output:**
```
                   SP500    VIX  US_INTEREST_RATE
SP500              1.000 -0.723             0.124
VIX               -0.723  1.000            -0.089
US_INTEREST_RATE   0.124 -0.089             1.000
```

### 5. Market Sentiment Indicator

```python
# Create a simple market sentiment score
data = collector.get_multiple_indicators(['VIX', 'SP500'])

# Calculate sentiment
data['sentiment'] = (
    (data['SP500'].pct_change() > 0).astype(int) -  # SP500 up = +1
    (data['VIX'] > 25).astype(int)                   # High VIX = -1
)

print(data[['SP500', 'VIX', 'sentiment']].tail())
```

## 🎯 Available Indicators

### Turkish Indicators (Requires EVDS API Key)

| Indicator | Description | Code |
|-----------|-------------|------|
| `TR_CPI` | Consumer Price Index | Monthly |
| `TR_PPI` | Producer Price Index | Monthly |
| `TR_INTEREST_RATE` | CBRT Policy Rate | Weekly |
| `TR_USD_TRY` | USD/TRY Exchange Rate | Daily |

### Global Indicators

| Indicator | Description | Source | Code |
|-----------|-------------|--------|------|
| `SP500` | S&P 500 Index | FRED | Daily |
| `DAX` | DAX Index | Yahoo | Daily |
| `VIX` | Volatility Index | Yahoo | Daily |
| `US_INTEREST_RATE` | Fed Funds Rate | FRED | Daily |
| `EUR_USD` | EUR/USD Rate | FRED | Daily |
| `GOLD` | Gold Price | FRED | Daily |
| `OIL_BRENT` | Brent Oil Price | FRED | Daily |

### List All Indicators

```python
indicators = collector.list_available_indicators()
print(indicators)
```

## ⚙️ Configuration

### Custom Date Ranges

```python
# Specific date range
data = collector.get_indicator(
    'SP500',
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

### Merge Strategies

```python
# Inner join - only dates with all indicators
data = collector.get_multiple_indicators(
    ['SP500', 'DAX'],
    merge_type='inner'
)

# Outer join - all dates (may have NaN)
data = collector.get_multiple_indicators(
    ['SP500', 'TR_CPI'],  # Different frequencies
    merge_type='outer'
)

# Forward fill missing values
data = data.fillna(method='ffill')
```

### Pass API Keys Directly

```python
# Instead of environment variables
collector = MacroCollector(
    fred_api_key='your_fred_key',
    evds_api_key='your_evds_key'
)
```

## 🔧 Troubleshooting

### Issue: "FRED API not available"

**Solution:**
```bash
# Check if key is set
echo $FRED_API_KEY

# If empty, set it
export FRED_API_KEY='your_key'

# Verify fredapi is installed
pip install fredapi
```

### Issue: "No data returned"

**Possible causes:**
1. Invalid date range (e.g., future dates)
2. Weekend/holiday (no market data)
3. Indicator code changed

**Solution:**
```python
# Try with broader date range
data = collector.get_indicator(
    'SP500',
    start_date='2024-01-01',  # Start earlier
    end_date='2024-11-01'
)

# Check for data
if data.empty:
    print("No data returned - check dates and indicator code")
```

### Issue: Empty DataFrame with NaN values

**Solution:**
```python
# Drop rows with missing data
data = data.dropna()

# Or forward fill
data = data.fillna(method='ffill')

# Or interpolate
data = data.interpolate(method='linear')
```

### Issue: Rate Limit Exceeded

**Solution:**
```python
import time

# Add delay between requests
for indicator in ['SP500', 'DAX', 'VIX']:
    data = collector.get_indicator(indicator)
    time.sleep(1)  # 1 second delay
```

## 📈 Integration with BIST Trading System

### Add to Stock Features

```python
import pandas as pd

# Your stock data
stock_data = pd.read_csv('bist_stocks.csv', index_col='date', parse_dates=True)

# Get macro features
macro_features = collector.get_macro_features(
    start_date=stock_data.index[0].strftime('%Y-%m-%d'),
    end_date=stock_data.index[-1].strftime('%Y-%m-%d')
)

# Merge
combined = stock_data.join(macro_features, how='left')

# Forward fill macro data (updates less frequently)
combined = combined.fillna(method='ffill')

print(f"Combined shape: {combined.shape}")
```

### Real-time Signal Generation

```python
# Get latest macro indicators
latest = collector.get_macro_features(
    start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
)

# Extract current values
current_vix = latest['VIX'].iloc[-1]
current_sp500 = latest['SP500'].iloc[-1]

# Use in trading logic
if current_vix > 30:
    print("High volatility - reduce position sizes")
elif current_vix < 15:
    print("Low volatility - can increase exposure")
```

## 📝 Examples

Run the comprehensive examples:

```bash
python examples/macro_data_example.py
```

## 🧪 Testing

Run tests:

```bash
# All tests
pytest tests/test_data/test_macro_collector.py -v

# Skip integration tests (no API keys needed)
pytest tests/test_data/test_macro_collector.py -v -m "not integration"
```

## 📚 Further Reading

- **Full Documentation**: `/src/data/collectors/README_MACRO.md`
- **API Documentation**:
  - FRED: https://fred.stlouisfed.org/docs/api/
  - EVDS: https://evds2.tcmb.gov.tr/
- **Configuration**: `/configs/data_sources.yaml`

## 💡 Tips

1. **Cache results** to avoid redundant API calls
2. **Use broad date ranges** then filter locally
3. **Forward fill** macro data when merging with high-frequency stock data
4. **Monitor rate limits** - FRED allows 120 req/min
5. **Handle missing data** appropriately for your use case

## 🆘 Support

For issues or questions:
1. Check the full documentation in `README_MACRO.md`
2. Review example code in `examples/macro_data_example.py`
3. Run tests to verify setup
4. Check indicator codes at source websites

---

**Happy Trading! 📊📈**
