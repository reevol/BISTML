# Signal Prioritizer - Multi-Factor Ranking Algorithm

## Overview

The Signal Prioritizer is a sophisticated ranking system that combines multiple signal quality factors into a single, actionable priority score. It helps traders focus on the highest-probability signals with the strongest institutional backing and positive sentiment.

## Key Features

- **Multi-Factor Analysis**: Combines confidence scores, whale activity (WAI), news sentiment, and model agreement
- **Flexible Strategies**: 7 pre-configured prioritization strategies for different trading styles
- **Custom Weighting**: Support for custom factor weights
- **Risk Adjustment**: Automatic risk-adjusted scoring based on confidence and model consensus
- **Batch Processing**: Efficiently prioritize and rank multiple signals
- **Comprehensive Filtering**: Filter signals by thresholds, direction, and strength
- **Detailed Metadata**: Access to component scores and raw metrics

## Architecture

### Core Components

#### 1. SignalInput
Input data structure containing all information about a trading signal:
- Symbol and timestamp
- Signal direction (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Model confidence score (0-100)
- Whale Activity Index (WAI) score (0-100)
- News sentiment score (-1 to +1)
- Individual model predictions
- Price targets and current prices

#### 2. SignalPrioritizer
The main ranking engine that:
- Calculates component scores for each factor
- Applies weighted scoring based on strategy
- Performs risk adjustment
- Ranks signals and assigns priority scores

#### 3. PrioritizedSignal
Output structure containing:
- Original signal information
- Priority score and rank
- Component scores (confidence, WAI, sentiment, agreement)
- Risk-adjusted score
- Signal strength classification
- Expected returns

## Prioritization Factors

### 1. Confidence Score (0-100)
- Source: ML model confidence
- Measures: Model certainty in prediction
- Impact: Higher confidence = higher priority

### 2. Whale Activity Index (WAI) (0-100)
- Source: Institutional flow analysis
- Measures: Top broker net flow relative to average volume
- Impact: Non-linear scaling emphasizes high WAI scores
- Special: Signals with strong whale backing get boosted

### 3. News Sentiment (-1 to +1)
- Source: NLP sentiment analysis
- Measures: News sentiment polarity
- Impact: Alignment with signal direction boosts score
- Special: Conflicting sentiment penalizes score

### 4. Model Agreement (0-100%)
- Source: Multiple forecasting models
- Measures: Percentage of models agreeing with signal direction
- Impact: Non-linear scaling rewards high consensus
- Special: Prediction magnitude consistency considered

## Prioritization Strategies

### 1. BALANCED (Default)
Equal weight to all factors:
- Confidence: 30%
- WAI: 25%
- Sentiment: 20%
- Agreement: 25%

**Use Case**: General-purpose trading, diversified approach

### 2. CONFIDENCE_FOCUSED
Heavy weight on model confidence:
- Confidence: 50%
- WAI: 20%
- Sentiment: 10%
- Agreement: 20%

**Use Case**: Trust the ML models, minimize other noise

### 3. WHALE_FOCUSED
Emphasizes institutional activity:
- Confidence: 20%
- WAI: 50%
- Sentiment: 10%
- Agreement: 20%

**Use Case**: Follow smart money, institutional flow trading

### 4. SENTIMENT_FOCUSED
Prioritizes news sentiment:
- Confidence: 25%
- WAI: 15%
- Sentiment: 45%
- Agreement: 15%

**Use Case**: News-driven trading, event-based strategies

### 5. CONSENSUS_FOCUSED
Emphasizes model agreement:
- Confidence: 20%
- WAI: 20%
- Sentiment: 15%
- Agreement: 45%

**Use Case**: Conservative trading, require model consensus

### 6. AGGRESSIVE
High risk, high reward:
- Confidence: 40%
- WAI: 30%
- Sentiment: 15%
- Agreement: 15%

**Use Case**: Active trading, willing to take higher risk

### 7. CONSERVATIVE
Low risk, proven patterns:
- Confidence: 25%
- WAI: 20%
- Sentiment: 20%
- Agreement: 35%

**Use Case**: Risk-averse trading, require strong consensus

## Usage Examples

### Basic Usage

```python
from src.signals.prioritizer import (
    SignalPrioritizer,
    create_signal_input,
    SignalDirection,
    PrioritizationStrategy
)

# Create a signal
signal = create_signal_input(
    symbol='THYAO',
    signal_direction='STRONG_BUY',
    confidence_score=85.0,
    wai_score=78.0,
    news_sentiment=0.6,
    model_predictions={
        'LSTM': 0.035,
        'GRU': 0.028,
        'XGBoost': 0.032,
        'LightGBM': 0.030
    },
    current_price=100.0,
    target_price=103.5
)

# Prioritize with balanced strategy
prioritizer = SignalPrioritizer(strategy=PrioritizationStrategy.BALANCED)
prioritized = prioritizer.prioritize_signal(signal)

print(f"Priority Score: {prioritized.priority_score:.2f}")
print(f"Risk-Adjusted Score: {prioritized.risk_adjusted_score:.2f}")
print(f"Signal Strength: {prioritized.signal_strength}")
```

### Batch Prioritization

```python
# Create multiple signals
signals = [
    create_signal_input('THYAO', 'STRONG_BUY', 85.0, 78.0, 0.6, {...}),
    create_signal_input('AKBNK', 'BUY', 72.0, 45.0, 0.2, {...}),
    create_signal_input('GARAN', 'SELL', 68.0, 62.0, -0.4, {...})
]

# Prioritize all signals
prioritizer = SignalPrioritizer(strategy='balanced')
prioritized_signals = prioritizer.prioritize_signals(signals)

# Convert to DataFrame for easy viewing
df = prioritizer.to_dataframe(prioritized_signals)
print(df)
```

### Custom Strategy

```python
# Define custom weights (must sum to 1.0)
custom_weights = {
    'confidence': 0.50,  # High weight on confidence
    'wai': 0.30,        # Moderate weight on whale activity
    'sentiment': 0.10,  # Low weight on sentiment
    'agreement': 0.10   # Low weight on agreement
}

prioritizer = SignalPrioritizer(custom_weights=custom_weights)
prioritized = prioritizer.prioritize_signals(signals)
```

### Filtering and Thresholds

```python
prioritizer = SignalPrioritizer(
    strategy='conservative',
    min_confidence_threshold=70.0,  # Only signals with confidence >= 70
    min_wai_threshold=50.0,         # Only signals with WAI >= 50
    min_agreement_threshold=0.75,   # Only signals with 75%+ model agreement
    enable_risk_adjustment=True     # Apply risk adjustment
)

# Filter BUY signals only
buy_signals = prioritizer.get_top_signals(
    prioritized_signals,
    top_n=5,
    signal_filter=SignalDirection.BUY
)
```

### Quick Prioritization

```python
from src.signals.prioritizer import prioritize_signals

# Quick one-liner
df = prioritize_signals(
    signals,
    strategy='whale_focused',
    min_confidence=60.0,
    top_n=10
)

print(df)
```

## Integration with Trading System

### 1. With Signal Generator

```python
from src.signals import SignalGenerator, SignalPrioritizer

# Generate signals from models
generator = SignalGenerator()
raw_signals = generator.generate_signals(model_outputs)

# Convert to prioritizer input format
signal_inputs = []
for signal in raw_signals:
    signal_input = create_signal_input(
        symbol=signal.symbol,
        signal_direction=signal.signal_type,
        confidence_score=signal.confidence,
        wai_score=get_wai_score(signal.symbol),
        news_sentiment=get_sentiment(signal.symbol),
        model_predictions=signal.model_predictions
    )
    signal_inputs.append(signal_input)

# Prioritize
prioritizer = SignalPrioritizer(strategy='balanced')
prioritized = prioritizer.prioritize_signals(signal_inputs)
```

### 2. With Whale Activity Analysis

```python
from src.features.whale import WhaleActivityIndex

# Calculate WAI scores
wai_calculator = WhaleActivityIndex(brokerage_data)
wai_scores = wai_calculator.calculate_wai(symbol)

# Create signal with WAI
signal = create_signal_input(
    symbol=symbol,
    signal_direction='BUY',
    confidence_score=model_confidence,
    wai_score=wai_scores['wai_score'].iloc[-1],  # Latest WAI
    news_sentiment=sentiment_score,
    model_predictions=predictions
)
```

### 3. With News Sentiment

```python
from src.data.collectors import NewsCollectorManager

# Collect and analyze news
news_manager = NewsCollectorManager()
articles = news_manager.collect_for_stock(symbol)

# Calculate aggregate sentiment
avg_sentiment = np.mean([a.sentiment_score for a in articles if a.sentiment_score])

# Create signal with sentiment
signal = create_signal_input(
    symbol=symbol,
    signal_direction='STRONG_BUY',
    confidence_score=confidence,
    wai_score=wai,
    news_sentiment=avg_sentiment,
    model_predictions=predictions
)
```

## Scoring Algorithm

### Priority Score Calculation

```
priority_score =
    confidence_component × weight_confidence +
    wai_component × weight_wai +
    sentiment_component × weight_sentiment +
    agreement_component × weight_agreement
```

### Risk Adjustment

```
risk_factor = confidence_factor × 0.6 + agreement_factor × 0.4
risk_adjusted_score = priority_score × (0.5 + 0.5 × risk_factor)
```

### Signal Strength Classification

- **VERY_STRONG**: risk_adjusted_score >= 80
- **STRONG**: risk_adjusted_score >= 65
- **MODERATE**: risk_adjusted_score >= 50
- **WEAK**: risk_adjusted_score < 50

## Component Score Details

### 1. Confidence Component
- Direct pass-through of confidence score
- No transformation (already 0-100)

### 2. WAI Component
- Non-linear sigmoid-like transformation
- Formula: `100 × (1 / (1 + exp(-5 × (wai/100 - 0.5))))`
- Effect: Emphasizes high WAI scores (>70)

### 3. Sentiment Component
- Base: `(sentiment + 1) × 50` (converts -1..1 to 0..100)
- Alignment boost: +20 points if sentiment matches signal direction
- Conflict penalty: -15 points if sentiment opposes signal
- Neutral: 50 if no sentiment data

### 4. Agreement Component
- High agreement (80%+): 80-100 points
- Good agreement (60-80%): 60-80 points
- Moderate agreement (40-60%): 40-60 points
- Low agreement (<40%): 0-40 points
- Consistency adjustment: Based on prediction magnitude variance

## Best Practices

### 1. Strategy Selection
- **Day Trading**: Use AGGRESSIVE or CONFIDENCE_FOCUSED
- **Swing Trading**: Use BALANCED or WHALE_FOCUSED
- **Long-term**: Use CONSERVATIVE or CONSENSUS_FOCUSED
- **Event-driven**: Use SENTIMENT_FOCUSED

### 2. Threshold Setting
- **High Confidence Trading**: Set min_confidence >= 75
- **Institutional Following**: Set min_wai >= 60
- **Consensus Required**: Set min_agreement >= 0.7

### 3. Signal Filtering
- Filter HOLD signals unless rebalancing
- Focus on STRONG_BUY/STRONG_SELL for best returns
- Use expected_return for position sizing

### 4. Risk Management
- Higher priority scores don't guarantee profits
- Always use stop losses
- Consider signal strength for position sizing
- Diversify across multiple high-priority signals

## Performance Considerations

### Computational Complexity
- Per signal: O(1) - constant time
- Batch prioritization: O(n log n) - due to sorting
- Very efficient even for 1000+ signals

### Memory Usage
- Minimal - each signal ~1KB
- Component scores cached in output
- No persistent state in prioritizer

### Scalability
- Handles real-time signal streams
- Suitable for high-frequency trading (30-min, 1-hour intervals)
- Can process entire BIST 100 in milliseconds

## Troubleshooting

### Common Issues

**Issue**: All signals get similar scores
- **Solution**: Check if weights sum to 1.0, increase variance in input data

**Issue**: Low agreement scores
- **Solution**: Ensure model predictions use same direction convention (positive = up)

**Issue**: Sentiment not impacting score
- **Solution**: Increase sentiment_multiplier or use SENTIMENT_FOCUSED strategy

**Issue**: WAI component always 50
- **Solution**: Ensure WAI scores are being passed (not None)

## API Reference

### SignalPrioritizer

```python
SignalPrioritizer(
    strategy='balanced',
    custom_weights=None,
    min_confidence_threshold=50.0,
    min_wai_threshold=0.0,
    min_agreement_threshold=0.5,
    enable_risk_adjustment=True,
    sentiment_multiplier=1.0,
    verbose=True
)
```

### Methods

- `prioritize_signal(signal)`: Prioritize a single signal
- `prioritize_signals(signals, filter_by_threshold=True)`: Batch prioritization
- `to_dataframe(signals)`: Convert to pandas DataFrame
- `get_top_signals(signals, top_n=10, signal_filter=None)`: Get top N signals

## Advanced Topics

### Custom Scoring Functions

For specialized strategies, you can subclass `SignalPrioritizer` and override component calculation methods:

```python
class CustomPrioritizer(SignalPrioritizer):
    def _calculate_wai_component(self, wai_score):
        # Custom WAI scoring logic
        if wai_score is None:
            return 50.0
        # Aggressive WAI emphasis
        return wai_score ** 1.5 if wai_score > 70 else wai_score * 0.8
```

### Real-time Integration

```python
# In trading loop
while market_open:
    # Collect latest data
    signals = collect_latest_signals()

    # Prioritize
    prioritizer = SignalPrioritizer(strategy='aggressive')
    prioritized = prioritizer.prioritize_signals(signals)

    # Execute top signals
    for signal in prioritized[:5]:
        if signal.risk_adjusted_score >= 75:
            execute_trade(signal)
```

## Example Output

```
Signal Prioritizer - Balanced Strategy

rank  symbol  signal       priority_score  risk_adjusted_score  signal_strength  model_agreement_pct
1     EREGL   STRONG_BUY   87.45          84.23                VERY_STRONG      100.0
2     ASELS   STRONG_BUY   82.31          78.90                STRONG           100.0
3     GARAN   STRONG_BUY   79.88          75.43                STRONG           100.0
4     THYAO   BUY          71.23          66.45                STRONG           75.0
5     AKBNK   BUY          65.89          60.12                MODERATE         75.0
```

## See Also

- [Signal Generator](./generator.py) - Generate raw signals from models
- [Whale Activity Index](../features/whale/activity_index.py) - Calculate WAI scores
- [News Collector](../data/collectors/news_collector.py) - Collect news and sentiment
- [Example Usage](../../examples/signal_prioritizer_example.py) - Comprehensive examples

## License

Part of the BIST AI Trading System
Author: BIST AI Trading System
Date: 2025-11-16
