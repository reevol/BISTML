"""
Signal Generation Example for BIST AI Trading System

This example demonstrates how to use the SignalGenerator to combine outputs
from multiple models (LSTM, Random Forest, ANN, Sentiment Analysis) to generate
comprehensive trading signals.

Usage:
    python examples/signal_generation_example.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.signals.generator import (
    SignalGenerator,
    ModelOutput,
    create_signal_generator,
    create_model_output
)


def generate_sample_historical_data(days=100, start_price=100.0):
    """Generate sample historical price data for demonstration."""
    np.random.seed(42)

    # Generate random walk with drift
    returns = np.random.normal(0.001, 0.02, days)
    prices = start_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    return pd.Series(prices, index=dates)


def simulate_lstm_prediction(current_price, trend='bullish'):
    """Simulate LSTM model prediction."""
    if trend == 'bullish':
        # Predict 2-5% increase
        predicted_price = current_price * (1 + np.random.uniform(0.02, 0.05))
        confidence = np.random.uniform(0.6, 0.85)
    elif trend == 'bearish':
        # Predict 2-5% decrease
        predicted_price = current_price * (1 - np.random.uniform(0.02, 0.05))
        confidence = np.random.uniform(0.6, 0.85)
    else:  # neutral
        # Predict slight change
        predicted_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
        confidence = np.random.uniform(0.4, 0.6)

    return create_model_output(
        model_name='lstm_price_forecaster',
        model_type='regression',
        prediction=predicted_price,
        confidence=confidence,
        forecast_horizon=5,
        model_architecture='3-layer LSTM [128, 64, 32]'
    )


def simulate_random_forest_prediction(trend='bullish'):
    """Simulate Random Forest classifier prediction."""
    if trend == 'bullish':
        # Strong BUY or BUY
        signal = np.random.choice([3, 4], p=[0.4, 0.6])
        probabilities = np.array([0.05, 0.10, 0.15, 0.30, 0.40]) if signal == 4 else \
                       np.array([0.05, 0.10, 0.20, 0.50, 0.15])
    elif trend == 'bearish':
        # SELL or Strong SELL
        signal = np.random.choice([0, 1], p=[0.4, 0.6])
        probabilities = np.array([0.40, 0.30, 0.15, 0.10, 0.05]) if signal == 0 else \
                       np.array([0.15, 0.50, 0.20, 0.10, 0.05])
    else:  # neutral
        signal = 2  # HOLD
        probabilities = np.array([0.10, 0.15, 0.50, 0.15, 0.10])

    confidence = np.max(probabilities)

    return create_model_output(
        model_name='random_forest_classifier',
        model_type='classification',
        prediction=signal,
        confidence=confidence,
        probabilities=probabilities,
        n_estimators=100,
        oob_score=0.75
    )


def simulate_ann_prediction(trend='bullish'):
    """Simulate ANN classifier prediction."""
    if trend == 'bullish':
        signal = np.random.choice([3, 4], p=[0.5, 0.5])
        probabilities = np.array([0.03, 0.08, 0.12, 0.35, 0.42]) if signal == 4 else \
                       np.array([0.03, 0.08, 0.17, 0.60, 0.12])
    elif trend == 'bearish':
        signal = np.random.choice([0, 1], p=[0.5, 0.5])
        probabilities = np.array([0.42, 0.35, 0.12, 0.08, 0.03]) if signal == 0 else \
                       np.array([0.12, 0.60, 0.17, 0.08, 0.03])
    else:  # neutral
        signal = 2
        probabilities = np.array([0.08, 0.12, 0.60, 0.12, 0.08])

    confidence = np.max(probabilities)

    return create_model_output(
        model_name='ann_classifier',
        model_type='classification',
        prediction=signal,
        confidence=confidence,
        probabilities=probabilities,
        hidden_layers=[256, 128, 64, 32],
        accuracy=0.72
    )


def simulate_sentiment_analysis(trend='bullish'):
    """Simulate sentiment analysis from news."""
    if trend == 'bullish':
        # Positive sentiment
        sentiment = np.random.uniform(0.3, 0.8)
        confidence = np.random.uniform(0.5, 0.75)
    elif trend == 'bearish':
        # Negative sentiment
        sentiment = np.random.uniform(-0.8, -0.3)
        confidence = np.random.uniform(0.5, 0.75)
    else:  # neutral
        sentiment = np.random.uniform(-0.2, 0.2)
        confidence = np.random.uniform(0.4, 0.6)

    return create_model_output(
        model_name='sentiment_analyzer',
        model_type='nlp',
        prediction=sentiment,
        confidence=confidence,
        news_count=5,
        sources=['KAP', 'Bloomberg HT', 'Investing.com']
    )


def example_single_stock_signal():
    """Example: Generate signal for a single stock."""
    print("=" * 80)
    print("EXAMPLE 1: Single Stock Signal Generation")
    print("=" * 80)

    # Generate sample data
    stock_code = 'THYAO'
    historical_prices = generate_sample_historical_data(days=100, start_price=100.0)
    current_price = historical_prices.iloc[-1]

    print(f"\nStock: {stock_code}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Historical Data: {len(historical_prices)} days")

    # Simulate model predictions (bullish scenario)
    print("\nSimulating model predictions (Bullish Scenario)...")
    model_outputs = [
        simulate_lstm_prediction(current_price, trend='bullish'),
        simulate_random_forest_prediction(trend='bullish'),
        simulate_ann_prediction(trend='bullish'),
        simulate_sentiment_analysis(trend='bullish')
    ]

    # Display model predictions
    print("\nModel Predictions:")
    print("-" * 80)
    for output in model_outputs:
        print(f"{output.model_name}:")
        print(f"  Type: {output.model_type}")
        print(f"  Prediction: {output.prediction}")
        print(f"  Confidence: {output.confidence:.2%}")
        if output.probabilities is not None:
            print(f"  Probabilities: {output.probabilities}")
        print()

    # Create signal generator
    generator = create_signal_generator(
        enable_dynamic_thresholds=True,
        risk_adjustment=True,
        min_confidence=0.3,
        regression_weight=0.35,
        classification_weight=0.35,
        sentiment_weight=0.15,
        technical_weight=0.15
    )

    # Generate signal
    signal = generator.generate_signal(
        stock_code=stock_code,
        model_outputs=model_outputs,
        current_price=current_price,
        historical_prices=historical_prices
    )

    # Display results
    print("\n" + "=" * 80)
    print("GENERATED TRADING SIGNAL")
    print("=" * 80)
    print(f"\nStock Code: {signal.stock_code}")
    print(f"Signal: {signal.signal.name}")
    print(f"Confidence Level: {signal.confidence.name}")
    print(f"Confidence Score: {signal.confidence_score:.2%}")
    print(f"Timestamp: {signal.timestamp}")

    print(f"\nPrice Analysis:")
    print(f"  Current Price: ${signal.current_price:.2f}")
    print(f"  Target Price: ${signal.target_price:.2f}")
    print(f"  Expected Return: {signal.expected_return:.2%}")

    print(f"\nRisk Management:")
    print(f"  Position Size: {signal.position_size:.2%} of portfolio")
    print(f"  Risk Score: {signal.risk_score:.2f}" if signal.risk_score else "  Risk Score: N/A")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: N/A")
    print(f"  Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "  Take Profit: N/A")

    print(f"\nModel Contributions:")
    for model_type, contribution in sorted(signal.model_contributions.items(),
                                          key=lambda x: x[1], reverse=True):
        print(f"  {model_type}: {contribution*100:.1f}%")

    print(f"\nRationale:")
    print(f"  {signal.rationale}")

    print(f"\nMarket Conditions:")
    print(f"  Volatility: {signal.metadata['market_volatility']*100:.2f}%"
          if signal.metadata.get('market_volatility') else "  Volatility: N/A")
    print(f"  Trend: {signal.metadata['market_trend']:.4f}"
          if signal.metadata.get('market_trend') else "  Trend: N/A")

    return signal


def example_multiple_stocks_batch():
    """Example: Generate signals for multiple stocks in batch."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Batch Signal Generation for Multiple Stocks")
    print("=" * 80)

    # Generate data for multiple stocks
    stocks = {
        'THYAO': {'trend': 'bullish', 'start_price': 100.0},
        'AKBNK': {'trend': 'neutral', 'start_price': 50.0},
        'TUPRS': {'trend': 'bearish', 'start_price': 150.0},
        'EREGL': {'trend': 'bullish', 'start_price': 75.0},
        'GARAN': {'trend': 'neutral', 'start_price': 80.0}
    }

    stocks_data = {}

    for stock_code, config in stocks.items():
        historical_prices = generate_sample_historical_data(
            days=100,
            start_price=config['start_price']
        )
        current_price = historical_prices.iloc[-1]
        trend = config['trend']

        # Simulate model predictions
        model_outputs = [
            simulate_lstm_prediction(current_price, trend=trend),
            simulate_random_forest_prediction(trend=trend),
            simulate_ann_prediction(trend=trend),
            simulate_sentiment_analysis(trend=trend)
        ]

        stocks_data[stock_code] = {
            'model_outputs': model_outputs,
            'current_price': current_price,
            'historical_prices': historical_prices
        }

    # Create signal generator
    generator = create_signal_generator(
        enable_dynamic_thresholds=True,
        risk_adjustment=True,
        min_confidence=0.3
    )

    # Generate signals for all stocks
    print("\nGenerating signals for {} stocks...".format(len(stocks_data)))
    signals = generator.generate_batch_signals(stocks_data)

    # Display results
    print("\n" + "=" * 80)
    print("BATCH SIGNAL RESULTS")
    print("=" * 80)

    # Create summary table
    summary_data = []
    for stock_code, signal in signals.items():
        summary_data.append({
            'Stock': stock_code,
            'Signal': signal.signal.name,
            'Confidence': f"{signal.confidence_score:.1%}",
            'Current': f"${signal.current_price:.2f}",
            'Target': f"${signal.target_price:.2f}" if signal.target_price else "N/A",
            'Return': f"{signal.expected_return:.1%}" if signal.expected_return else "N/A",
            'Position': f"{signal.position_size:.1%}" if signal.position_size else "0.0%",
            'Risk': f"{signal.risk_score:.2f}" if signal.risk_score else "N/A"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Detailed view for each signal
    print("\n" + "=" * 80)
    print("DETAILED SIGNALS")
    print("=" * 80)

    for stock_code, signal in signals.items():
        print(f"\n{stock_code}:")
        print(f"  Signal: {signal.signal.name} ({signal.confidence.name})")
        print(f"  Expected Return: {signal.expected_return:.2%}" if signal.expected_return else "  Expected Return: N/A")
        print(f"  Position Size: {signal.position_size:.2%}")
        print(f"  Rationale: {signal.rationale}")

    return signals


def example_different_market_conditions():
    """Example: Signal generation under different market conditions."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Signal Generation Under Different Market Conditions")
    print("=" * 80)

    stock_code = 'THYAO'
    base_price = 100.0

    scenarios = {
        'Low Volatility': {
            'volatility': 0.01,
            'trend': 'bullish'
        },
        'High Volatility': {
            'volatility': 0.05,
            'trend': 'bullish'
        },
        'Downtrend': {
            'volatility': 0.02,
            'trend': 'bearish'
        }
    }

    results = []

    for scenario_name, config in scenarios.items():
        # Generate historical data with specific characteristics
        np.random.seed(42)
        days = 100
        returns = np.random.normal(0.001, config['volatility'], days)
        prices = base_price * np.exp(np.cumsum(returns))

        if config['trend'] == 'bearish':
            # Add downward trend
            trend = np.linspace(0, -0.1, days)
            prices = prices * (1 + trend)

        historical_prices = pd.Series(prices)
        current_price = historical_prices.iloc[-1]

        # Simulate predictions
        model_outputs = [
            simulate_lstm_prediction(current_price, trend=config['trend']),
            simulate_random_forest_prediction(trend=config['trend']),
            simulate_ann_prediction(trend=config['trend']),
            simulate_sentiment_analysis(trend=config['trend'])
        ]

        # Generate signal
        generator = create_signal_generator(
            enable_dynamic_thresholds=True,
            risk_adjustment=True
        )

        signal = generator.generate_signal(
            stock_code=stock_code,
            model_outputs=model_outputs,
            current_price=current_price,
            historical_prices=historical_prices
        )

        results.append({
            'Scenario': scenario_name,
            'Signal': signal.signal.name,
            'Confidence': f"{signal.confidence_score:.1%}",
            'Position': f"{signal.position_size:.1%}",
            'Risk': f"{signal.risk_score:.2f}" if signal.risk_score else "N/A",
            'Volatility': f"{signal.metadata['market_volatility']*100:.2f}%"
                         if signal.metadata.get('market_volatility') else "N/A"
        })

    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    print("\nObservations:")
    print("  - High volatility reduces position size and confidence")
    print("  - Downtrend scenarios adjust signal strength")
    print("  - Dynamic thresholds adapt to market conditions")


def example_custom_configuration():
    """Example: Custom signal generator configuration."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Signal Generator Configuration")
    print("=" * 80)

    # Create custom signal generator
    custom_generator = SignalGenerator(
        # Custom signal thresholds
        signal_thresholds={
            'strong_buy': 0.05,      # Require 5% for strong buy
            'buy': 0.02,              # Require 2% for buy
            'sell': -0.02,
            'strong_sell': -0.05
        },
        # Custom confidence thresholds
        confidence_thresholds={
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.50,
            'low': 0.35,
            'very_low': 0.0
        },
        # Higher minimum confidence
        min_confidence=0.5,
        # Custom model weights
        regression_weight=0.40,
        classification_weight=0.40,
        sentiment_weight=0.10,
        technical_weight=0.10,
        # Enable features
        enable_dynamic_thresholds=True,
        risk_adjustment=True
    )

    print("\nCustom Configuration:")
    print(f"  Signal Thresholds: {custom_generator.signal_thresholds}")
    print(f"  Minimum Confidence: {custom_generator.min_confidence}")
    print(f"  Regression Weight: {custom_generator.regression_weight}")
    print(f"  Classification Weight: {custom_generator.classification_weight}")
    print(f"  Sentiment Weight: {custom_generator.sentiment_weight}")

    # Generate sample data
    stock_code = 'THYAO'
    historical_prices = generate_sample_historical_data(days=100, start_price=100.0)
    current_price = historical_prices.iloc[-1]

    # Simulate predictions
    model_outputs = [
        simulate_lstm_prediction(current_price, trend='bullish'),
        simulate_random_forest_prediction(trend='bullish'),
        simulate_ann_prediction(trend='bullish'),
        simulate_sentiment_analysis(trend='bullish')
    ]

    # Generate signal
    signal = custom_generator.generate_signal(
        stock_code=stock_code,
        model_outputs=model_outputs,
        current_price=current_price,
        historical_prices=historical_prices
    )

    print(f"\nGenerated Signal with Custom Configuration:")
    print(f"  Signal: {signal.signal.name}")
    print(f"  Confidence: {signal.confidence.name} ({signal.confidence_score:.2%})")
    print(f"  Position Size: {signal.position_size:.2%}")
    print(f"  Expected Return: {signal.expected_return:.2%}" if signal.expected_return else "  Expected Return: N/A")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  Signal Generation Examples - BIST AI Trading System".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    # Run examples
    example_single_stock_signal()
    example_multiple_stocks_batch()
    example_different_market_conditions()
    example_custom_configuration()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Integrate with actual model predictions from LSTM, Random Forest, and ANN")
    print("  2. Connect to real-time market data feeds")
    print("  3. Implement backtesting with historical signals")
    print("  4. Set up automated signal generation pipeline")
    print("  5. Configure alerts and notifications for high-confidence signals")
    print()


if __name__ == "__main__":
    main()
