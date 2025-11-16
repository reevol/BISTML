"""
LSTM Price Forecasting Example

This example demonstrates how to use the LSTM model for price forecasting
in the BIST AI Trading System.

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from src.models.forecasting.lstm_model import (
    LSTMPriceForecaster,
    SequenceGenerator,
    train_lstm_model,
    create_lstm_forecaster
)


def example_basic_usage():
    """Example 1: Basic LSTM forecasting with synthetic data."""
    print("=" * 80)
    print("Example 1: Basic LSTM Price Forecasting")
    print("=" * 80)

    # Create synthetic price data
    np.random.seed(42)
    n_samples = 1000

    # Generate price with trend and noise
    trend = np.linspace(100, 150, n_samples)
    seasonality = 10 * np.sin(np.linspace(0, 10 * np.pi, n_samples))
    noise = np.random.randn(n_samples) * 2
    price = trend + seasonality + noise

    data = pd.DataFrame({
        'close': price,
        'volume': np.random.randint(1000, 10000, n_samples),
        'returns': pd.Series(price).pct_change().fillna(0)
    })

    print(f"\nData shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print(f"\nFirst few rows:")
    print(data.head())

    # Create sequence generator
    seq_gen = SequenceGenerator(
        data=data,
        sequence_length=60,
        forecast_horizon=1,
        target_column='close',
        feature_columns=['close', 'volume', 'returns']
    )

    # Generate sequences
    X_train, y_train, X_val, y_val, X_test, y_test = seq_gen.create_sequences(
        train_size=0.8,
        validation_size=0.1
    )

    print(f"\nSequence shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Create and train model
    model = LSTMPriceForecaster(
        sequence_length=60,
        n_features=3,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    print("\nModel Architecture:")
    print(model.get_model_summary())

    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        early_stopping_patience=10,
        verbose=1
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)

    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Make predictions
    predictions = model.predict(X_test[:10])
    print(f"\nSample predictions (first 10):")
    print(f"Predicted: {predictions}")
    print(f"Actual:    {y_test[:10]}")

    return model, metrics


def example_bidirectional_lstm():
    """Example 2: Bidirectional LSTM with batch normalization."""
    print("\n" + "=" * 80)
    print("Example 2: Bidirectional LSTM with Batch Normalization")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(123)
    n_samples = 800
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    data = pd.DataFrame({
        'close': price
    })

    # Create model with bidirectional LSTM and batch norm
    model = LSTMPriceForecaster(
        sequence_length=30,
        n_features=1,
        lstm_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        bidirectional=True,
        use_batch_norm=True,
        l1_reg=0.001,
        l2_reg=0.001
    )

    print("\nBidirectional LSTM Architecture:")
    print(model.get_model_summary())

    # Generate sequences
    seq_gen = SequenceGenerator(
        data=data,
        sequence_length=30,
        forecast_horizon=1,
        target_column='close'
    )

    X_train, y_train, X_val, y_val, X_test, y_test = seq_gen.create_sequences()

    print(f"\nTraining bidirectional LSTM...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=16,
        verbose=1
    )

    metrics = model.evaluate(X_test, y_test)
    print("\nBidirectional LSTM Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    return model


def example_complete_pipeline():
    """Example 3: Complete training pipeline with convenience function."""
    print("\n" + "=" * 80)
    print("Example 3: Complete Training Pipeline")
    print("=" * 80)

    # Create realistic stock-like data
    np.random.seed(456)
    n_samples = 1200

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Simulate OHLCV data
    returns = np.random.randn(n_samples) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_price = (high + low) / 2
    volume = np.random.randint(100000, 1000000, n_samples)

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Add some technical indicators
    data['returns'] = data['close'].pct_change().fillna(0)
    data['sma_20'] = data['close'].rolling(20).mean().fillna(method='bfill')
    data['volatility'] = data['returns'].rolling(20).std().fillna(0)

    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nData summary:")
    print(data.describe())

    # Use complete pipeline
    print("\nTraining with complete pipeline...")
    model, metrics = train_lstm_model(
        data=data,
        target_column='close',
        feature_columns=['close', 'volume', 'returns', 'sma_20', 'volatility'],
        sequence_length=60,
        forecast_horizon=1,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        epochs=50,
        batch_size=32,
        train_size=0.7,
        validation_size=0.15,
        verbose=1
    )

    # Save model
    model_path = '/tmp/lstm_price_model.keras'
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    return model, data


def example_multi_step_prediction():
    """Example 4: Multi-step ahead prediction."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Step Ahead Prediction")
    print("=" * 80)

    # Create simple trend data
    np.random.seed(789)
    n_samples = 500
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.3)

    data = pd.DataFrame({'close': price})

    # Train model
    seq_gen = SequenceGenerator(
        data=data,
        sequence_length=50,
        forecast_horizon=1,
        target_column='close'
    )

    X_train, y_train, X_val, y_val, X_test, y_test = seq_gen.create_sequences(
        train_size=0.8,
        validation_size=0.1
    )

    model = create_lstm_forecaster(
        sequence_length=50,
        n_features=1,
        lstm_units=[64, 32],
        dropout_rate=0.2
    )

    print("Training model for multi-step prediction...")
    model.train(X_train, y_train, X_val, y_val, epochs=30, verbose=1)

    # Multi-step prediction
    last_sequence = X_test[0]
    n_steps_ahead = 20

    print(f"\nPredicting {n_steps_ahead} steps ahead...")
    future_predictions = model.predict_next_steps(last_sequence, n_steps=n_steps_ahead)

    print(f"\nFuture predictions (next {n_steps_ahead} steps):")
    print(future_predictions)

    print(f"\nActual next values (first 10):")
    print(y_test[:10])

    return model, future_predictions


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LSTM PRICE FORECASTING EXAMPLES")
    print("BIST AI Trading System")
    print("=" * 80)

    try:
        # Example 1: Basic usage
        model1, metrics1 = example_basic_usage()

        # Example 2: Bidirectional LSTM
        model2 = example_bidirectional_lstm()

        # Example 3: Complete pipeline
        model3, data3 = example_complete_pipeline()

        # Example 4: Multi-step prediction
        model4, predictions4 = example_multi_step_prediction()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
