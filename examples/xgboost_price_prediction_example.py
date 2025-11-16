"""
Example: XGBoost Price Prediction for BIST Stocks

This script demonstrates how to use the XGBoostPricePredictor class
for stock price forecasting with hyperparameter tuning and feature
importance analysis.

Usage:
    python examples/xgboost_price_prediction_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.forecasting.xgboost_model import XGBoostPricePredictor


def generate_sample_stock_data(n_days=1000, n_features=25):
    """
    Generate synthetic stock data for demonstration.

    In production, replace this with actual BIST stock data from your
    data collection pipeline.

    Args:
        n_days: Number of days of historical data
        n_features: Number of technical indicators/features

    Returns:
        X: Feature DataFrame
        y: Target Series (next day's return)
    """
    np.random.seed(42)

    # Generate dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(n_days)]
    dates.reverse()

    # Feature names representing various technical indicators
    feature_names = [
        # Moving averages
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20',
        # Momentum indicators
        'rsi_14', 'rsi_21',
        'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d',
        # Volatility
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr_14',
        # Volume
        'volume_sma_20', 'volume_ratio',
        # Others
        'adx_14', 'cci_20', 'williams_r',
        # Whale Activity Index (simulated)
        'whale_activity_index',
        # Fundamental
        'pe_ratio'
    ]

    # Generate random features (in practice, these would be calculated indicators)
    X = pd.DataFrame(
        np.random.randn(n_days, n_features),
        columns=feature_names,
        index=dates
    )

    # Normalize some features to realistic ranges
    X['rsi_14'] = 50 + X['rsi_14'] * 15  # RSI typically 30-70
    X['rsi_21'] = 50 + X['rsi_21'] * 15
    X['volume_ratio'] = np.abs(X['volume_ratio'])  # Always positive
    X['whale_activity_index'] = X['whale_activity_index'] * 10
    X['pe_ratio'] = 15 + np.abs(X['pe_ratio']) * 5  # P/E around 10-25

    # Generate target: next day's price return (%)
    # Use first few features as primary drivers
    base_signal = (
        X['sma_5'] * 0.3 +
        X['rsi_14'] * 0.02 +
        X['macd'] * 0.2 +
        X['whale_activity_index'] * 0.15
    )

    # Add some noise
    noise = np.random.randn(n_days) * 0.5

    y = pd.Series(
        base_signal + noise,
        index=dates,
        name='next_day_return'
    )

    return X, y


def main():
    """Main execution function."""

    print("=" * 80)
    print("XGBoost Price Prediction Example for BIST Stocks")
    print("=" * 80)
    print()

    # Step 1: Generate sample data
    print("Step 1: Generating sample stock data...")
    X, y = generate_sample_stock_data(n_days=1000, n_features=25)
    print(f"  Data shape: {X.shape}")
    print(f"  Features: {list(X.columns)[:5]}... (showing first 5)")
    print(f"  Target: {y.name}")
    print()

    # Step 2: Split data (time series split - no shuffling!)
    print("Step 2: Splitting data into train/val/test sets...")
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print()

    # Step 3: Initialize model
    print("Step 3: Initializing XGBoost Price Predictor...")
    predictor = XGBoostPricePredictor(
        objective='reg:squarederror',
        random_state=42,
        enable_scaling=True
    )
    print("  Model initialized with feature scaling enabled")
    print()

    # Step 4: Hyperparameter tuning
    print("Step 4: Performing hyperparameter tuning with GridSearchCV...")
    print("  This may take a few minutes...")

    best_params = predictor.tune_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        param_grid=predictor.get_param_grid('quick'),  # Use 'moderate' or 'extensive' for better results
        cv_folds=3,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    print("  Hyperparameter tuning completed!")
    print(f"  Best parameters: {best_params}")
    print()

    # Step 5: Train model with early stopping
    print("Step 5: Training model with early stopping...")
    predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=20,
        verbose=False
    )
    print("  Model training completed!")
    print()

    # Step 6: Evaluate on test set
    print("Step 6: Evaluating model on test set...")
    metrics = predictor.evaluate(X_test, y_test)
    print("  Test Set Performance:")
    for metric, value in metrics.items():
        print(f"    {metric.upper():6s}: {value:.6f}")
    print()

    # Step 7: Feature importance analysis
    print("Step 7: Analyzing feature importance...")

    # Get top 10 features by gain
    importance_df = predictor.get_feature_importance(
        importance_type='gain',
        top_n=10
    )

    print("  Top 10 Most Important Features (by gain):")
    for idx, row in importance_df.iterrows():
        print(f"    {idx+1:2d}. {row['feature']:25s}: {row['importance']:.2f}")
    print()

    # Step 8: Make predictions
    print("Step 8: Making predictions on test set...")
    predictions = predictor.predict(X_test)

    # Show first 5 predictions vs actual
    print("  Sample Predictions (first 5):")
    print(f"    {'Date':<12s} {'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
    print("    " + "-" * 46)

    for i in range(min(5, len(y_test))):
        date = y_test.index[i].strftime('%Y-%m-%d')
        actual = y_test.iloc[i]
        pred = predictions[i]
        error = actual - pred
        print(f"    {date:<12s} {actual:>10.4f} {pred:>10.4f} {error:>10.4f}")
    print()

    # Step 9: Save model
    print("Step 9: Saving trained model...")
    model_path = 'models/saved/xgboost_price_predictor.pkl'
    predictor.save_model(model_path)
    print(f"  Model saved to: {model_path}")
    print()

    # Step 10: Load model (demonstration)
    print("Step 10: Loading saved model (demonstration)...")
    loaded_predictor = XGBoostPricePredictor()
    loaded_predictor.load_model(model_path)
    print("  Model loaded successfully!")

    # Verify loaded model works
    loaded_predictions = loaded_predictor.predict(X_test[:5])
    original_predictions = predictions[:5]

    print("  Verification (predictions should match):")
    print(f"    Original:  {original_predictions}")
    print(f"    Loaded:    {loaded_predictions}")
    print()

    # Step 11: Model summary
    print("Step 11: Model Summary")
    print("=" * 80)
    summary = predictor.get_model_summary()
    for key, value in summary.items():
        if key == 'feature_names':
            print(f"  {key}: {len(value)} features")
        elif key == 'best_params':
            print(f"  {key}:")
            for param, param_value in value.items():
                print(f"    - {param}: {param_value}")
        else:
            print(f"  {key}: {value}")

    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    # Optional: Plot feature importance (uncomment to display)
    # print("\nGenerating feature importance plot...")
    # predictor.plot_feature_importance(
    #     importance_type='gain',
    #     top_n=15,
    #     save_path='outputs/feature_importance.png'
    # )


def quick_prediction_example():
    """
    Quick example showing minimal code for prediction.
    """
    print("\n" + "=" * 80)
    print("Quick Prediction Example (Minimal Code)")
    print("=" * 80)
    print()

    # Generate data
    X, y = generate_sample_stock_data(n_days=500, n_features=25)

    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train
    predictor = XGBoostPricePredictor(enable_scaling=True)
    predictor.train(X_train, y_train)

    # Predict
    predictions = predictor.predict(X_test)

    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)

    print("Quick prediction completed!")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print()


if __name__ == "__main__":
    # Run full example
    main()

    # Run quick example
    quick_prediction_example()
