# Trading Signal Generator

## Overview

The Signal Generator is a sophisticated ensemble system that combines outputs from multiple models (regression, classification, and NLP) to generate comprehensive trading signals for the BIST AI Trading System.

## Features

### Multi-Model Ensemble
- **Regression Models**: LSTM, GRU, XGBoost, LightGBM for price forecasting
- **Classification Models**: Random Forest, ANN for signal classification
- **NLP Models**: Sentiment analysis from news and disclosures
- **Technical Indicators**: Integration of technical analysis signals

### Signal Aggregation Logic
- **Weighted Voting**: Combines model predictions using configurable weights
- **Confidence Scoring**: Calculates confidence based on model agreement
- **Dynamic Thresholds**: Adjusts signal thresholds based on market conditions
- **Risk Adjustment**: Modifies signals based on volatility and risk metrics

### Output Signals
- **STRONG_BUY** (4): High confidence buy signal
- **BUY** (3): Moderate buy signal
- **HOLD** (2): No action recommended
- **SELL** (1): Moderate sell signal
- **STRONG_SELL** (0): High confidence sell signal

### Confidence Levels
- **VERY_HIGH**: 80%+ confidence
- **HIGH**: 65-80% confidence
- **MEDIUM**: 45-65% confidence
- **LOW**: 30-45% confidence
- **VERY_LOW**: <30% confidence

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Outputs                            │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Regression │Classification│     NLP      │   Technical   │
│   Models     │   Models     │   Sentiment  │  Indicators   │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       │              │              │               │
       v              v              v               v
┌──────────────────────────────────────────────────────────────┐
│              Signal Aggregation Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Weighted Voting                                   │   │
│  │  • Confidence Scoring                                │   │
│  │  • Agreement Analysis                                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           v
┌──────────────────────────────────────────────────────────────┐
│              Risk Adjustment Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Volatility Analysis                               │   │
│  │  • Market Condition Assessment                       │   │
│  │  • Position Size Calculation                         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           v
┌──────────────────────────────────────────────────────────────┐
│                   Final Trading Signal                       │
│  • Signal Type (BUY/SELL/HOLD)                              │
│  • Confidence Level                                          │
│  • Position Size Recommendation                              │
│  • Stop Loss / Take Profit Levels                           │
│  • Risk Score                                                │
│  • Rationale                                                 │
└──────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from src.signals import (
    SignalGenerator,
    create_signal_generator,
    create_model_output
)

# Create signal generator
generator = create_signal_generator(
    enable_dynamic_thresholds=True,
    risk_adjustment=True,
    min_confidence=0.3
)

# Create model outputs
model_outputs = [
    # LSTM price prediction
    create_model_output(
        model_name='lstm_forecaster',
        model_type='regression',
        prediction=105.5,  # Predicted price
        confidence=0.75
    ),

    # Random Forest classification
    create_model_output(
        model_name='random_forest',
        model_type='classification',
        prediction=3,  # BUY signal
        confidence=0.68,
        probabilities=np.array([0.05, 0.10, 0.17, 0.68, 0.00])
    ),

    # Sentiment analysis
    create_model_output(
        model_name='sentiment_analyzer',
        model_type='nlp',
        prediction=0.45,  # Positive sentiment (0.45 on -1 to 1 scale)
        confidence=0.60
    )
]

# Generate signal
signal = generator.generate_signal(
    stock_code='THYAO',
    model_outputs=model_outputs,
    current_price=100.0,
    historical_prices=price_series
)

# Access signal information
print(f"Signal: {signal.signal.name}")
print(f"Confidence: {signal.confidence_score:.2%}")
print(f"Position Size: {signal.position_size:.2%}")
print(f"Expected Return: {signal.expected_return:.2%}")
print(f"Stop Loss: ${signal.stop_loss:.2f}")
print(f"Take Profit: ${signal.take_profit:.2f}")
```

### Batch Signal Generation

```python
# Generate signals for multiple stocks
stocks_data = {
    'THYAO': {
        'model_outputs': thyao_outputs,
        'current_price': 100.0,
        'historical_prices': thyao_prices
    },
    'AKBNK': {
        'model_outputs': akbnk_outputs,
        'current_price': 50.0,
        'historical_prices': akbnk_prices
    }
}

signals = generator.generate_batch_signals(stocks_data)

for stock_code, signal in signals.items():
    print(f"{stock_code}: {signal.signal.name} ({signal.confidence_score:.2%})")
```

### Custom Configuration

```python
# Create generator with custom parameters
custom_generator = SignalGenerator(
    # Model weights (must sum to ~1.0)
    regression_weight=0.40,
    classification_weight=0.40,
    sentiment_weight=0.10,
    technical_weight=0.10,

    # Signal thresholds for regression models
    signal_thresholds={
        'strong_buy': 0.05,      # 5% expected return
        'buy': 0.02,              # 2% expected return
        'sell': -0.02,
        'strong_sell': -0.05
    },

    # Confidence thresholds
    confidence_thresholds={
        'very_high': 0.85,
        'high': 0.70,
        'medium': 0.50,
        'low': 0.35,
        'very_low': 0.0
    },

    # Features
    enable_dynamic_thresholds=True,
    risk_adjustment=True,
    min_confidence=0.5,
    volatility_window=20
)
```

## Signal Generation Process

### 1. Model Output Collection
Collect predictions from all available models:
- Regression models provide price forecasts
- Classification models provide signal categories
- NLP models provide sentiment scores

### 2. Model-Type Aggregation
Each model type is aggregated separately:

**Regression Models:**
- Calculate expected returns from price predictions
- Weight by model confidence
- Convert returns to signal using thresholds
- Calculate agreement factor from variance

**Classification Models:**
- Weight signals by confidence/probability
- Average weighted signals
- Calculate agreement factor

**NLP/Sentiment:**
- Weight sentiment scores by confidence
- Convert sentiment (-1 to 1) to signal (0 to 4)

### 3. Weighted Voting
Combine all model types using configured weights:
```
final_signal = (
    regression_signal * regression_weight * regression_confidence +
    classification_signal * classification_weight * classification_confidence +
    sentiment_signal * sentiment_weight * sentiment_confidence
) / total_weight
```

### 4. Risk Adjustment
If enabled:
- Calculate market volatility from historical prices
- Compute risk score (0-1)
- Reduce confidence in high volatility
- Moderate extreme signals
- Adjust position size

### 5. Dynamic Threshold Adjustment
If enabled:
- Adjust signal thresholds based on market volatility
- Modify thresholds based on market trend
- Higher thresholds in volatile/bearish markets

### 6. Position Sizing
Calculate recommended position size:
- Base size on signal strength and confidence
- Adjust for risk score
- Cap at maximum portfolio percentage (20%)

### 7. Risk Level Calculation
Calculate stop loss and take profit:
- Use ATR-based levels if historical data available
- Fall back to percentage-based levels
- Different levels for buy vs sell signals

## Model Output Format

### ModelOutput Class

```python
@dataclass
class ModelOutput:
    model_name: str              # Name of the model
    model_type: str              # 'regression', 'classification', 'nlp'
    prediction: Union[float, int, str]  # Model prediction
    confidence: Optional[float]  # Confidence score (0-1)
    probabilities: Optional[np.ndarray]  # Class probabilities
    metadata: Dict[str, Any]     # Additional information
```

### Regression Models
```python
ModelOutput(
    model_name='lstm_forecaster',
    model_type='regression',
    prediction=105.5,  # Predicted price
    confidence=0.75,   # Model confidence
    metadata={'horizon': 5, 'mse': 0.012}
)
```

### Classification Models
```python
ModelOutput(
    model_name='random_forest',
    model_type='classification',
    prediction=3,  # Signal class (0-4)
    confidence=0.68,
    probabilities=np.array([0.05, 0.10, 0.17, 0.68, 0.00]),
    metadata={'oob_score': 0.75}
)
```

### NLP/Sentiment Models
```python
ModelOutput(
    model_name='sentiment_analyzer',
    model_type='nlp',
    prediction=0.45,  # Sentiment score (-1 to 1)
    confidence=0.60,
    metadata={'news_count': 5, 'sources': ['KAP', 'Bloomberg']}
)
```

## Trading Signal Output

### TradingSignal Class

```python
@dataclass
class TradingSignal:
    signal: SignalType              # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence: SignalConfidence    # VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW
    confidence_score: float         # Numerical confidence (0-1)
    timestamp: datetime             # Signal generation time
    stock_code: str                 # Stock symbol
    current_price: float           # Current stock price
    target_price: float            # Target price from regression
    expected_return: float         # Expected return percentage
    position_size: float           # Recommended position (0-1)
    risk_score: float              # Risk score (0-1)
    stop_loss: float               # Stop loss price
    take_profit: float             # Take profit price
    model_contributions: Dict      # Model weights in final signal
    rationale: str                 # Signal explanation
    metadata: Dict                 # Market conditions, etc.
```

## Integration Examples

### With LSTM Model

```python
from src.models.forecasting.lstm_model import LSTMPriceForecaster

# Train LSTM
lstm_model = LSTMPriceForecaster(...)
lstm_model.train(X_train, y_train, X_val, y_val)

# Get prediction
prediction = lstm_model.predict(X_latest)

# Create model output
lstm_output = create_model_output(
    model_name='lstm_forecaster',
    model_type='regression',
    prediction=prediction[0],
    confidence=0.75  # From model metrics
)
```

### With Random Forest Classifier

```python
from src.models.classification.random_forest import TradingSignalClassifier

# Train classifier
rf_classifier = TradingSignalClassifier(...)
rf_classifier.fit(X_train, y_train)

# Get prediction
prediction, probabilities = rf_classifier.predict(X_latest, return_proba=True)

# Create model output
rf_output = create_model_output(
    model_name='random_forest',
    model_type='classification',
    prediction=prediction[0],
    confidence=np.max(probabilities[0]),
    probabilities=probabilities[0]
)
```

### With Sentiment Analysis

```python
from src.data.collectors.news_collector import NewsCollector

# Collect news
news_collector = NewsCollector()
articles = news_collector.collect_stock_news('THYAO', days=7)

# Analyze sentiment (pseudocode)
sentiment_scores = [article.sentiment_score for article in articles]
avg_sentiment = np.mean(sentiment_scores)
confidence = len(sentiment_scores) / 10  # More news = higher confidence

# Create model output
sentiment_output = create_model_output(
    model_name='sentiment_analyzer',
    model_type='nlp',
    prediction=avg_sentiment,
    confidence=min(confidence, 1.0),
    metadata={'news_count': len(articles)}
)
```

## Configuration Guidelines

### Model Weights

Recommended weight distributions:

**Balanced (Default):**
- Regression: 35%
- Classification: 35%
- Sentiment: 15%
- Technical: 15%

**Regression-Heavy (Price-Focused):**
- Regression: 50%
- Classification: 30%
- Sentiment: 10%
- Technical: 10%

**Classification-Heavy (Signal-Focused):**
- Regression: 30%
- Classification: 50%
- Sentiment: 10%
- Technical: 10%

### Signal Thresholds

**Conservative (Lower Risk):**
```python
signal_thresholds={
    'strong_buy': 0.05,      # 5%
    'buy': 0.025,            # 2.5%
    'sell': -0.025,
    'strong_sell': -0.05
}
```

**Moderate (Balanced):**
```python
signal_thresholds={
    'strong_buy': 0.03,      # 3%
    'buy': 0.01,             # 1%
    'sell': -0.01,
    'strong_sell': -0.03
}
```

**Aggressive (Higher Risk):**
```python
signal_thresholds={
    'strong_buy': 0.02,      # 2%
    'buy': 0.005,            # 0.5%
    'sell': -0.005,
    'strong_sell': -0.02
}
```

## Performance Considerations

1. **Model Diversity**: Use models with different architectures and approaches
2. **Confidence Calibration**: Ensure model confidence scores are well-calibrated
3. **Historical Validation**: Backtest signal performance regularly
4. **Dynamic Updates**: Adjust thresholds based on market regime
5. **Risk Management**: Always use position sizing and stop losses

## Best Practices

1. **Minimum Confidence**: Set appropriate `min_confidence` to filter weak signals
2. **Model Agreement**: Higher weight to scenarios with strong model agreement
3. **Market Conditions**: Enable dynamic thresholds for changing markets
4. **Risk Adjustment**: Always enable risk adjustment in production
5. **Position Sizing**: Cap maximum position size (recommended: 10-20%)
6. **Regular Updates**: Retrain models and recalibrate thresholds regularly

## Troubleshooting

### Low Confidence Signals
- Check model predictions for consistency
- Verify model confidence scores are reasonable
- Increase model diversity
- Adjust confidence thresholds

### Too Many HOLD Signals
- Lower `min_confidence` parameter
- Adjust signal thresholds
- Check if models are too conservative

### Extreme Position Sizes
- Review risk adjustment settings
- Cap position sizes explicitly
- Check volatility calculations

## Future Enhancements

- [ ] Multi-timeframe signal aggregation
- [ ] Sector/market correlation adjustment
- [ ] Order flow integration
- [ ] Machine learning for optimal weight determination
- [ ] Real-time threshold optimization
- [ ] Signal quality scoring and tracking
- [ ] Advanced ensemble methods (stacking, blending)

## Dependencies

- numpy: Array operations
- pandas: Time series handling
- datetime: Timestamp management
- dataclasses: Data structures
- enum: Enumerations
- logging: Logging functionality

## License

Part of the BIST AI Trading System
