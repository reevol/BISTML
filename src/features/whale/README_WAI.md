# Whale Activity Index (WAI) Module

## Overview

The Whale Activity Index (WAI) module provides comprehensive analysis of institutional (whale) trading activity on the Borsa Istanbul (BIST). It tracks large broker movements, identifies accumulation/distribution patterns, and detects unusual institutional activity that may indicate informed trading or price manipulation.

## Key Features

### 1. **Whale Activity Index (WAI) Calculation**
- Tracks top N brokers' net flow relative to average daily volume
- Provides normalized scores (0-100 scale)
- Includes directional (signed) and momentum-adjusted variants
- Multi-component scoring system

### 2. **Unusual Activity Detection**
- Statistical anomaly detection using z-scores
- Identifies unusual accumulation/distribution events
- Flags abnormal whale participation levels
- Calculates unusual activity strength metrics

### 3. **Price-Flow Discrepancy Analysis**
- Detects when price movement doesn't match institutional flow
- Identifies "accumulation under pressure" (whales buying while price falls)
- Flags "distribution during pump" (whales selling while price rises)
- Potential manipulation detection

### 4. **Accumulation/Distribution Scoring**
- Multi-timeframe analysis (short-term vs long-term)
- Phase classification (ACCUMULATION, DISTRIBUTION, NEUTRAL, etc.)
- Trend alignment detection
- Phase strength scoring (0-100)

### 5. **Institutional Pressure Metrics**
- Buy/Sell pressure percentages
- Pressure intensity and consistency
- Combined institutional pressure score
- Pressure level classification

### 6. **Signal Generation**
- Combines all metrics into actionable trading signals
- Confidence scoring (0-100)
- Signal types: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
- Configurable confidence thresholds

## Installation

The module requires the following dependencies (already in requirements.txt):

```bash
pip install numpy pandas scipy
```

## Quick Start

### Basic WAI Calculation

```python
from src.features.whale import WhaleActivityIndex, calculate_wai

# Using the class
wai = WhaleActivityIndex(
    brokerage_data=brokerage_df,
    top_n_brokers=10,
    lookback_period=20
)

wai_scores = wai.calculate_wai(symbol='THYAO')

# Using convenience function
wai_scores = calculate_wai(
    brokerage_data=brokerage_df,
    symbol='THYAO',
    top_n_brokers=10
)
```

### Detect Unusual Activity

```python
unusual = wai.detect_unusual_activity(
    symbol='AKBNK',
    z_threshold=2.5,
    min_participation=20.0
)

# Filter unusual days
unusual_days = unusual[unusual['unusual_activity'] == True]
```

### Generate Trading Signals

```python
from src.features.whale import generate_whale_signals

signals = generate_whale_signals(
    brokerage_data=brokerage_df,
    symbol='GARAN',
    price_data=price_df,  # optional but recommended
    confidence_threshold=70.0
)

# Get actionable signals
actionable = signals[signals['signal'] != 'HOLD']
```

## Data Requirements

### Brokerage Data Format

The brokerage data DataFrame must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| symbol | string | Stock symbol (e.g., 'THYAO') |
| broker_code | string | Broker identifier |
| buy_volume | float | Buy volume for the broker |
| sell_volume | float | Sell volume for the broker |
| net_volume | float | Net volume (buy - sell) |
| buy_value | float | Total buy value (optional) |
| sell_value | float | Total sell value (optional) |

### Price Data Format (Optional)

For price-flow discrepancy analysis:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| symbol | string | Stock symbol |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| volume | float | Total volume (optional) |

## Core Metrics Explained

### 1. WAI Score

The Whale Activity Index score combines two key components:

**WAI = (Whale Participation % × 0.4) + (Net Flow Intensity × 100 × 0.6)**

Where:
- **Whale Participation %**: Percentage of total volume traded by top N brokers
- **Net Flow Intensity**: Absolute net flow relative to average daily volume

**Interpretation:**
- **0-20**: Low whale activity
- **20-40**: Moderate whale activity
- **40-60**: High whale activity
- **60-80**: Very high whale activity
- **80-100**: Extreme whale activity

### 2. WAI Directional

Signed version of WAI that indicates direction:
- **Positive**: Net accumulation by whales
- **Negative**: Net distribution by whales
- **Magnitude**: Strength of the movement

### 3. WAI Momentum

WAI adjusted for directional consistency:

**WAI Momentum = WAI Directional × Directional Consistency**

Stronger when whales maintain consistent buy/sell direction over multiple days.

### 4. Net Flow Ratio

Net flow of top brokers relative to average daily volume:

**Net Flow Ratio = Whale Net Volume / Average Daily Volume**

**Interpretation:**
- **> 0.3**: Strong institutional buying
- **0.1 to 0.3**: Moderate buying
- **-0.1 to 0.1**: Neutral
- **-0.3 to -0.1**: Moderate selling
- **< -0.3**: Strong institutional selling

### 5. Unusual Activity Z-Score

Statistical measure of how unusual the current activity is:

**Z-Score = (Current Value - Mean) / Standard Deviation**

**Interpretation:**
- **|Z| > 2.5**: Highly unusual (flag for investigation)
- **|Z| > 2.0**: Unusual
- **|Z| < 2.0**: Normal variation

### 6. Accumulation/Distribution Phase

Classification based on multi-timeframe flow analysis:

- **STRONG_ACCUMULATION**: Sustained whale buying across timeframes
- **ACCUMULATION**: Net whale buying
- **NEUTRAL**: Balanced or low activity
- **DISTRIBUTION**: Net whale selling
- **STRONG_DISTRIBUTION**: Sustained whale selling across timeframes

### 7. Institutional Pressure Score

Combined metric of buying/selling pressure:

**Pressure Score = (Net Pressure / 100) × Intensity × Consistency × 100**

Where:
- **Net Pressure**: (Buy Days % - Sell Days %)
- **Intensity**: Average flow magnitude
- **Consistency**: Directional stability

**Interpretation:**
- **> 30**: Strong buy pressure
- **15 to 30**: Moderate buy pressure
- **-15 to 15**: Neutral
- **-30 to -15**: Moderate sell pressure
- **< -30**: Strong sell pressure

## Advanced Usage

### Multi-Symbol Analysis

```python
symbols = ['THYAO', 'AKBNK', 'GARAN', 'EREGL', 'TUPRS']

results = {}
for symbol in symbols:
    wai = WhaleActivityIndex(brokerage_data=brokerage_df)
    results[symbol] = wai.generate_whale_signals(
        symbol=symbol,
        confidence_threshold=70.0
    )

# Find strongest signals
all_signals = pd.concat(results.values())
top_signals = all_signals[
    all_signals['signal'].isin(['STRONG_BUY', 'STRONG_SELL'])
].sort_values('confidence_score', ascending=False)
```

### Custom Broker Selection

```python
# Track specific brokers instead of top N
custom_brokers = ['BRK001', 'BRK002', 'BRK005']

wai_scores = wai.calculate_wai(
    symbol='SAHOL',
    custom_brokers=custom_brokers
)
```

### Price-Flow Discrepancy Detection

```python
wai = WhaleActivityIndex(
    brokerage_data=brokerage_df,
    price_data=price_df
)

discrepancy = wai.detect_price_flow_discrepancy(
    symbol='ASELS',
    discrepancy_threshold=1.5,
    window=5
)

# Find manipulation candidates
manipulation = discrepancy[
    discrepancy['potential_manipulation'] == True
].sort_values('manipulation_score', ascending=False)

print("Potential Manipulation Events:")
for idx, row in manipulation.head(10).iterrows():
    if row['accumulation_under_pressure']:
        print(f"{row['date']}: Accumulation under pressure "
              f"(Score: {row['manipulation_score']:.2f})")
    if row['distribution_during_pump']:
        print(f"{row['date']}: Distribution during pump "
              f"(Score: {row['manipulation_score']:.2f})")
```

### Integration with Trading System

```python
# In your signal generation pipeline
def generate_signals(symbol, date):
    # Collect data
    brokerage_data = whale_collector.collect_brokerage_data(
        symbols=symbol,
        start_date=date - timedelta(days=90),
        end_date=date
    )

    price_data = price_collector.get_price_data(
        symbol=symbol,
        start_date=date - timedelta(days=90),
        end_date=date
    )

    # Generate whale signals
    wai = WhaleActivityIndex(
        brokerage_data=brokerage_data,
        price_data=price_data,
        top_n_brokers=10
    )

    signals = wai.generate_whale_signals(
        symbol=symbol,
        confidence_threshold=70.0
    )

    # Get latest signal
    latest = signals.iloc[-1]

    return {
        'signal': latest['signal'],
        'confidence': latest['confidence_score'],
        'wai_score': latest['wai_score'],
        'phase': latest['phase'],
        'pressure': latest['pressure_level']
    }
```

## Performance Considerations

### Memory Optimization

For large datasets, consider processing symbols individually:

```python
# Instead of loading all symbols at once
for symbol in symbol_list:
    symbol_data = brokerage_data[brokerage_data['symbol'] == symbol]
    wai = WhaleActivityIndex(brokerage_data=symbol_data)
    result = wai.calculate_wai(symbol)
    # Process result...
```

### Computation Time

Typical processing times on modern hardware:

- WAI calculation: ~10-50ms per symbol (60 days of data)
- Unusual activity detection: ~20-100ms per symbol
- Price-flow discrepancy: ~30-150ms per symbol
- Full signal generation: ~50-200ms per symbol

For 100 symbols: ~5-20 seconds total

## Interpretation Guide

### Signal Confidence Scores

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 90-100 | Very high confidence | Strong position |
| 80-89 | High confidence | Consider position |
| 70-79 | Moderate confidence | Small position or watchlist |
| 60-69 | Low confidence | Watchlist only |
| < 60 | Very low confidence | Ignore |

### Combining with Other Signals

WAI signals work best when combined with:

1. **Technical Analysis**: Confirm with RSI, MACD, volume patterns
2. **Price Action**: Check support/resistance levels
3. **News Sentiment**: Verify no major negative news
4. **Fundamental Data**: Ensure company fundamentals support the move

Example combined signal logic:

```python
def combined_signal(symbol):
    whale_signal = generate_whale_signals(...)
    technical_signal = calculate_technical_indicators(...)
    sentiment_signal = get_news_sentiment(...)

    if (whale_signal['signal'] == 'STRONG_BUY' and
        whale_signal['confidence'] > 75 and
        technical_signal['rsi'] < 70 and
        sentiment_signal > 0):
        return 'BUY', high_confidence
    # ... other conditions
```

## Common Patterns

### 1. Stealth Accumulation

**Characteristics:**
- Low volatility
- Consistent whale buying (5+ days)
- Price relatively stable or slightly down
- Increasing whale participation

**Signal:**
- `phase = 'ACCUMULATION'`
- `accumulation_under_pressure = True`
- `wai_momentum > 0` and increasing

**Action:** Potential early-stage accumulation; monitor for breakout

### 2. Distribution Before Drop

**Characteristics:**
- Price at highs or in uptrend
- Whale net selling increasing
- High whale participation
- Price-flow discrepancy present

**Signal:**
- `phase = 'DISTRIBUTION'`
- `distribution_during_pump = True`
- `wai_directional < 0` and decreasing

**Action:** Consider exit or short position

### 3. Panic Buying

**Characteristics:**
- Sudden spike in whale buying
- Unusual activity flag triggered
- High z-score (> 2.5)
- Often follows news or technical breakout

**Signal:**
- `unusual_accumulation = True`
- `unusual_strength > 3.0`
- `wai_score > 70`

**Action:** Momentum play; may be short-term overbought

### 4. Capitulation Selling

**Characteristics:**
- Extreme whale selling
- Very high z-score (> 3.0)
- Price falling sharply
- May indicate bottom

**Signal:**
- `unusual_distribution = True`
- `unusual_strength > 3.0`
- Watch for reversal in next 1-3 days

**Action:** Potential contrarian opportunity if fundamentals intact

## Troubleshooting

### Issue: No unusual activity detected

**Solutions:**
- Lower `z_threshold` (try 2.0 instead of 2.5)
- Reduce `min_participation` threshold
- Check if data covers sufficient history (need 20+ days)

### Issue: Too many false signals

**Solutions:**
- Increase `confidence_threshold` (try 80 instead of 70)
- Add additional filters (price action, volume, sentiment)
- Require multiple confirmations across timeframes

### Issue: WAI scores always near 0

**Solutions:**
- Verify broker data quality
- Check that top brokers are correctly identified
- Ensure sufficient trading volume in data

## API Reference

See the inline documentation in `activity_index.py` for detailed API reference.

Key classes and functions:
- `WhaleActivityIndex`: Main analysis class
- `calculate_wai()`: Quick WAI calculation
- `detect_unusual_activity()`: Convenience function for anomaly detection
- `generate_whale_signals()`: One-step signal generation

## Examples

See `activity_index_example.py` for complete working examples of all features.

## Contributing

When extending this module:
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests for new features
5. Update this documentation

## License

Part of the BIST AI Trading System.

## Support

For issues or questions:
1. Check the examples in `activity_index_example.py`
2. Review this documentation
3. Consult the main project documentation
4. Check inline code documentation

---

**Last Updated:** 2025-11-16
**Version:** 1.0.0
**Author:** BIST AI Trading System
