# Signal Generator Integration Guide

## Overview

This guide demonstrates how to integrate the Signal Generator with existing models in the BIST AI Trading System to produce actionable trading signals.

## Complete Integration Workflow

### Step 1: Prepare Data

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load historical price data
stock_code = 'THYAO'
historical_prices = pd.read_csv('data/historical_prices.csv', index_col='date', parse_dates=True)
prices = historical_prices[stock_code]

# Get current price
current_price = prices.iloc[-1]

# Prepare features for models
from src.features.feature_engineering import FeatureEngineering

fe = FeatureEngineering(prices)
features = fe.generate_all_features()

# Latest feature vector
X_latest = features.iloc[[-1]]
```

### Step 2: Get Predictions from Regression Models

```python
from src.models.forecasting.lstm_model import LSTMPriceForecaster
from src.models.forecasting.xgboost_model import XGBoostPricePredictor
from src.signals import create_model_output

# LSTM Model
lstm_model = LSTMPriceForecaster.load('models/lstm_thyao.keras')
lstm_prediction = lstm_model.predict(X_latest)
lstm_confidence = 0.75  # From validation metrics

lstm_output = create_model_output(
    model_name='lstm_price_forecaster',
    model_type='regression',
    prediction=float(lstm_prediction[0]),
    confidence=lstm_confidence,
    metadata={
        'forecast_horizon': 5,
        'sequence_length': 60,
        'validation_rmse': 0.012
    }
)

# XGBoost Model
xgb_model = XGBoostPricePredictor.load('models/xgboost_thyao.json')
xgb_prediction = xgb_model.predict(X_latest)
xgb_confidence = 0.70

xgb_output = create_model_output(
    model_name='xgboost_price_predictor',
    model_type='regression',
    prediction=float(xgb_prediction[0]),
    confidence=xgb_confidence,
    metadata={
        'n_estimators': 100,
        'validation_r2': 0.68
    }
)
```

### Step 3: Get Predictions from Classification Models

```python
from src.models.classification.random_forest import TradingSignalClassifier
from src.models.classification.ann_classifier import ANNSignalClassifier

# Random Forest Classifier
rf_classifier = TradingSignalClassifier.load('models/rf_classifier_thyao.pkl')
rf_prediction, rf_proba = rf_classifier.predict(X_latest, return_proba=True)

rf_output = create_model_output(
    model_name='random_forest_classifier',
    model_type='classification',
    prediction=int(rf_prediction[0]),
    confidence=float(np.max(rf_proba[0])),
    probabilities=rf_proba[0],
    metadata={
        'n_estimators': 100,
        'oob_score': 0.75,
        'feature_importances': rf_classifier.get_feature_importance(top_n=5).to_dict()
    }
)

# ANN Classifier
ann_classifier = ANNSignalClassifier(n_features=X_latest.shape[1], n_classes=5)
ann_classifier.load_model('models/ann_classifier_thyao.keras')
ann_prediction, ann_proba = ann_classifier.predict(X_latest, return_probabilities=True)

ann_output = create_model_output(
    model_name='ann_signal_classifier',
    model_type='classification',
    prediction=int(ann_prediction[0]),
    confidence=float(np.max(ann_proba[0])),
    probabilities=ann_proba[0],
    metadata={
        'hidden_layers': [256, 128, 64, 32],
        'accuracy': 0.72
    }
)
```

### Step 4: Get Sentiment Analysis

```python
from src.data.collectors.news_collector import NewsCollector

# Collect recent news
news_collector = NewsCollector()
articles = news_collector.collect_stock_news(stock_code, days=7)

# Calculate sentiment
if articles:
    # Assuming articles have sentiment_score attribute
    sentiment_scores = [
        article.sentiment_score
        for article in articles
        if article.sentiment_score is not None
    ]

    if sentiment_scores:
        avg_sentiment = np.mean(sentiment_scores)
        # Confidence based on number of articles and score agreement
        confidence = min(len(sentiment_scores) / 10, 1.0) * (1 - np.std(sentiment_scores))

        sentiment_output = create_model_output(
            model_name='news_sentiment_analyzer',
            model_type='nlp',
            prediction=float(avg_sentiment),
            confidence=float(confidence),
            metadata={
                'news_count': len(articles),
                'sources': list(set(a.source for a in articles)),
                'date_range': f"{articles[-1].published_date} to {articles[0].published_date}"
            }
        )
    else:
        # No sentiment available - create neutral output with low confidence
        sentiment_output = create_model_output(
            model_name='news_sentiment_analyzer',
            model_type='nlp',
            prediction=0.0,  # Neutral
            confidence=0.1,  # Low confidence
            metadata={'news_count': 0}
        )
else:
    sentiment_output = create_model_output(
        model_name='news_sentiment_analyzer',
        model_type='nlp',
        prediction=0.0,
        confidence=0.1,
        metadata={'news_count': 0}
    )
```

### Step 5: Generate Trading Signal

```python
from src.signals import SignalGenerator, create_signal_generator

# Create signal generator with configuration
generator = create_signal_generator(
    # Model weights
    regression_weight=0.35,
    classification_weight=0.35,
    sentiment_weight=0.15,
    technical_weight=0.15,

    # Signal thresholds
    signal_thresholds={
        'strong_buy': 0.03,      # 3% expected return
        'buy': 0.01,              # 1% expected return
        'sell': -0.01,
        'strong_sell': -0.03
    },

    # Configuration
    enable_dynamic_thresholds=True,
    risk_adjustment=True,
    min_confidence=0.3,
    volatility_window=20
)

# Combine all model outputs
model_outputs = [
    lstm_output,
    xgb_output,
    rf_output,
    ann_output,
    sentiment_output
]

# Generate signal
signal = generator.generate_signal(
    stock_code=stock_code,
    model_outputs=model_outputs,
    current_price=current_price,
    historical_prices=prices,
    market_data={
        'volume': historical_prices['volume'].iloc[-1],
        'market_cap': 1000000000  # Example
    }
)

# Display signal
print(f"\n{'='*60}")
print(f"Trading Signal for {stock_code}")
print(f"{'='*60}")
print(f"Signal: {signal.signal.name}")
print(f"Confidence: {signal.confidence.name} ({signal.confidence_score:.2%})")
print(f"Current Price: ${signal.current_price:.2f}")
print(f"Target Price: ${signal.target_price:.2f}")
print(f"Expected Return: {signal.expected_return:.2%}")
print(f"Position Size: {signal.position_size:.2%} of portfolio")
print(f"Stop Loss: ${signal.stop_loss:.2f}")
print(f"Take Profit: ${signal.take_profit:.2f}")
print(f"Risk Score: {signal.risk_score:.2f}")
print(f"\nRationale: {signal.rationale}")
```

### Step 6: Batch Processing for Multiple Stocks

```python
# Process multiple stocks
stocks = ['THYAO', 'AKBNK', 'TUPRS', 'EREGL', 'GARAN']
stocks_data = {}

for stock_code in stocks:
    try:
        # Load data and features
        prices = historical_prices[stock_code]
        current_price = prices.iloc[-1]

        # Generate features
        fe = FeatureEngineering(prices)
        features = fe.generate_all_features()
        X_latest = features.iloc[[-1]]

        # Get model predictions (simplified)
        model_outputs = []

        # LSTM
        lstm_pred = lstm_model.predict(X_latest)
        model_outputs.append(create_model_output(
            'lstm', 'regression', float(lstm_pred[0]), 0.75
        ))

        # Random Forest
        rf_pred, rf_proba = rf_classifier.predict(X_latest, return_proba=True)
        model_outputs.append(create_model_output(
            'random_forest', 'classification',
            int(rf_pred[0]), float(np.max(rf_proba[0])),
            probabilities=rf_proba[0]
        ))

        # ANN
        ann_pred, ann_proba = ann_classifier.predict(X_latest, return_probabilities=True)
        model_outputs.append(create_model_output(
            'ann', 'classification',
            int(ann_pred[0]), float(np.max(ann_proba[0])),
            probabilities=ann_proba[0]
        ))

        # Sentiment
        articles = news_collector.collect_stock_news(stock_code, days=7)
        if articles and any(a.sentiment_score for a in articles):
            sentiment = np.mean([a.sentiment_score for a in articles if a.sentiment_score])
            model_outputs.append(create_model_output(
                'sentiment', 'nlp', float(sentiment), 0.6
            ))

        stocks_data[stock_code] = {
            'model_outputs': model_outputs,
            'current_price': current_price,
            'historical_prices': prices
        }

    except Exception as e:
        print(f"Error processing {stock_code}: {e}")
        continue

# Generate signals for all stocks
signals = generator.generate_batch_signals(stocks_data)

# Display results
print(f"\n{'='*80}")
print(f"Batch Signal Generation Results")
print(f"{'='*80}\n")

summary = []
for stock_code, signal in signals.items():
    summary.append({
        'Stock': stock_code,
        'Signal': signal.signal.name,
        'Confidence': f"{signal.confidence_score:.1%}",
        'Return': f"{signal.expected_return:.1%}" if signal.expected_return else "N/A",
        'Position': f"{signal.position_size:.1%}"
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
```

## Production Pipeline

### Complete Trading System Integration

```python
class TradingSignalPipeline:
    """Production pipeline for signal generation."""

    def __init__(self, config_path='config/signal_config.yaml'):
        # Load configuration
        self.config = self.load_config(config_path)

        # Initialize models
        self.lstm_model = LSTMPriceForecaster.load(self.config['models']['lstm_path'])
        self.xgb_model = XGBoostPricePredictor.load(self.config['models']['xgb_path'])
        self.rf_classifier = TradingSignalClassifier.load(self.config['models']['rf_path'])
        self.ann_classifier = ANNSignalClassifier(...)
        self.ann_classifier.load_model(self.config['models']['ann_path'])

        # Initialize signal generator
        self.signal_generator = create_signal_generator(**self.config['signal_generator'])

        # Initialize data collectors
        self.news_collector = NewsCollector()

    def generate_signal(self, stock_code, date=None):
        """Generate signal for a single stock."""
        try:
            # Get historical data
            prices = self.get_historical_prices(stock_code, date)
            current_price = prices.iloc[-1]

            # Generate features
            features = self.generate_features(prices)
            X_latest = features.iloc[[-1]]

            # Get model predictions
            model_outputs = []

            # Regression models
            model_outputs.append(self.get_lstm_prediction(X_latest))
            model_outputs.append(self.get_xgboost_prediction(X_latest))

            # Classification models
            model_outputs.append(self.get_rf_prediction(X_latest))
            model_outputs.append(self.get_ann_prediction(X_latest))

            # Sentiment analysis
            model_outputs.append(self.get_sentiment(stock_code))

            # Generate signal
            signal = self.signal_generator.generate_signal(
                stock_code=stock_code,
                model_outputs=model_outputs,
                current_price=current_price,
                historical_prices=prices
            )

            return signal

        except Exception as e:
            logging.error(f"Error generating signal for {stock_code}: {e}")
            return None

    def get_lstm_prediction(self, X):
        """Get LSTM prediction."""
        pred = self.lstm_model.predict(X)
        return create_model_output(
            'lstm', 'regression', float(pred[0]), 0.75
        )

    def get_xgboost_prediction(self, X):
        """Get XGBoost prediction."""
        pred = self.xgb_model.predict(X)
        return create_model_output(
            'xgboost', 'regression', float(pred[0]), 0.70
        )

    def get_rf_prediction(self, X):
        """Get Random Forest prediction."""
        pred, proba = self.rf_classifier.predict(X, return_proba=True)
        return create_model_output(
            'random_forest', 'classification',
            int(pred[0]), float(np.max(proba[0])),
            probabilities=proba[0]
        )

    def get_ann_prediction(self, X):
        """Get ANN prediction."""
        pred, proba = self.ann_classifier.predict(X, return_probabilities=True)
        return create_model_output(
            'ann', 'classification',
            int(pred[0]), float(np.max(proba[0])),
            probabilities=proba[0]
        )

    def get_sentiment(self, stock_code):
        """Get sentiment analysis."""
        articles = self.news_collector.collect_stock_news(stock_code, days=7)
        if articles and any(a.sentiment_score for a in articles):
            scores = [a.sentiment_score for a in articles if a.sentiment_score]
            sentiment = np.mean(scores)
            confidence = min(len(scores) / 10, 1.0)
            return create_model_output(
                'sentiment', 'nlp', float(sentiment), float(confidence)
            )
        else:
            return create_model_output(
                'sentiment', 'nlp', 0.0, 0.1
            )

    def generate_daily_signals(self, stocks, date=None):
        """Generate signals for all stocks."""
        signals = {}
        for stock_code in stocks:
            signal = self.generate_signal(stock_code, date)
            if signal:
                signals[stock_code] = signal
        return signals

# Usage
pipeline = TradingSignalPipeline('config/signal_config.yaml')
signals = pipeline.generate_daily_signals(['THYAO', 'AKBNK', 'TUPRS'])

for stock, signal in signals.items():
    print(f"{stock}: {signal.signal.name} ({signal.confidence_score:.2%})")
```

## Configuration File Example

```yaml
# config/signal_config.yaml

models:
  lstm_path: 'models/lstm_forecaster.keras'
  xgb_path: 'models/xgboost_predictor.json'
  rf_path: 'models/rf_classifier.pkl'
  ann_path: 'models/ann_classifier.keras'

signal_generator:
  regression_weight: 0.35
  classification_weight: 0.35
  sentiment_weight: 0.15
  technical_weight: 0.15

  signal_thresholds:
    strong_buy: 0.03
    buy: 0.01
    sell: -0.01
    strong_sell: -0.03

  confidence_thresholds:
    very_high: 0.8
    high: 0.65
    medium: 0.45
    low: 0.3
    very_low: 0.0

  enable_dynamic_thresholds: true
  risk_adjustment: true
  min_confidence: 0.3
  volatility_window: 20

data:
  price_data_path: 'data/historical_prices.csv'
  feature_cache_path: 'data/features_cache/'

logging:
  level: 'INFO'
  file: 'logs/signal_generation.log'
```

## Error Handling

```python
from src.signals import SignalType, SignalConfidence, TradingSignal

def safe_generate_signal(generator, stock_code, model_outputs, current_price, prices):
    """Generate signal with error handling."""
    try:
        signal = generator.generate_signal(
            stock_code=stock_code,
            model_outputs=model_outputs,
            current_price=current_price,
            historical_prices=prices
        )
        return signal

    except ValueError as e:
        logging.error(f"Value error for {stock_code}: {e}")
        # Return HOLD signal with low confidence
        return TradingSignal(
            signal=SignalType.HOLD,
            confidence=SignalConfidence.VERY_LOW,
            confidence_score=0.1,
            timestamp=datetime.now(),
            stock_code=stock_code,
            rationale=f"Error: {str(e)}"
        )

    except Exception as e:
        logging.error(f"Unexpected error for {stock_code}: {e}")
        return TradingSignal(
            signal=SignalType.HOLD,
            confidence=SignalConfidence.VERY_LOW,
            confidence_score=0.1,
            timestamp=datetime.now(),
            stock_code=stock_code,
            rationale=f"System error: {str(e)}"
        )
```

## Testing

```python
import unittest

class TestSignalGeneration(unittest.TestCase):
    """Test signal generation pipeline."""

    def setUp(self):
        self.generator = create_signal_generator()

    def test_single_model_signal(self):
        """Test with single model output."""
        model_outputs = [
            create_model_output('lstm', 'regression', 105.0, 0.75)
        ]

        signal = self.generator.generate_signal(
            stock_code='TEST',
            model_outputs=model_outputs,
            current_price=100.0
        )

        self.assertIsNotNone(signal)
        self.assertIn(signal.signal, list(SignalType))

    def test_multiple_models_signal(self):
        """Test with multiple model outputs."""
        model_outputs = [
            create_model_output('lstm', 'regression', 105.0, 0.75),
            create_model_output('rf', 'classification', 3, 0.68,
                              probabilities=np.array([0.05, 0.10, 0.17, 0.68, 0.00])),
            create_model_output('sentiment', 'nlp', 0.45, 0.60)
        ]

        signal = self.generator.generate_signal(
            stock_code='TEST',
            model_outputs=model_outputs,
            current_price=100.0
        )

        self.assertIsNotNone(signal)
        self.assertGreater(signal.confidence_score, 0.0)

    def test_batch_generation(self):
        """Test batch signal generation."""
        stocks_data = {
            'STOCK1': {
                'model_outputs': [create_model_output('lstm', 'regression', 105.0, 0.75)],
                'current_price': 100.0
            },
            'STOCK2': {
                'model_outputs': [create_model_output('lstm', 'regression', 95.0, 0.75)],
                'current_price': 100.0
            }
        }

        signals = self.generator.generate_batch_signals(stocks_data)

        self.assertEqual(len(signals), 2)
        self.assertIn('STOCK1', signals)
        self.assertIn('STOCK2', signals)

if __name__ == '__main__':
    unittest.main()
```

## Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/signal_generation.log'),
        logging.StreamHandler()
    ]
)

def log_signal(signal):
    """Log generated signal."""
    logging.info(f"Signal generated for {signal.stock_code}")
    logging.info(f"  Signal: {signal.signal.name}")
    logging.info(f"  Confidence: {signal.confidence_score:.2%}")
    logging.info(f"  Expected Return: {signal.expected_return:.2%}" if signal.expected_return else "  Expected Return: N/A")
    logging.info(f"  Position Size: {signal.position_size:.2%}")

    # Log model contributions
    for model, contrib in signal.model_contributions.items():
        logging.debug(f"  {model}: {contrib*100:.1f}%")
```

## Next Steps

1. **Backtesting**: Test signal performance on historical data
2. **Parameter Optimization**: Tune weights and thresholds
3. **Real-time Integration**: Connect to live data feeds
4. **Alert System**: Set up notifications for high-confidence signals
5. **Performance Tracking**: Monitor signal accuracy and returns
6. **Model Updating**: Regularly retrain models and update weights

## Support

For issues or questions:
- Check the README in `/home/user/BISTML/src/signals/README.md`
- Review example code in `/home/user/BISTML/examples/signal_generation_example.py`
- Consult model documentation in respective model directories
