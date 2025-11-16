"""
Feature Engineering Example for BIST AI Trading System

This example demonstrates how to use the feature_engineering module to:
1. Combine technical, fundamental, and whale features
2. Create lag and rolling features for time series
3. Scale and encode features
4. Build a complete feature matrix for model training

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
    ScalingMethod,
    ImputationMethod,
    create_time_features,
    remove_correlated_features,
    get_feature_statistics
)


def generate_sample_technical_features(n_days=252):
    """Generate sample technical indicator features"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_days) * 2),
        'open': 100 + np.cumsum(np.random.randn(n_days) * 2),
        'high': 100 + np.cumsum(np.random.randn(n_days) * 2) + 2,
        'low': 100 + np.cumsum(np.random.randn(n_days) * 2) - 2,
        'volume': np.random.randint(1000000, 10000000, n_days),
        'RSI_14': np.random.uniform(30, 70, n_days),
        'MACD': np.random.randn(n_days) * 0.5,
        'MACD_Signal': np.random.randn(n_days) * 0.5,
        'BB_upper': 100 + np.cumsum(np.random.randn(n_days) * 2) + 5,
        'BB_lower': 100 + np.cumsum(np.random.randn(n_days) * 2) - 5,
        'ATR_14': np.random.uniform(1, 5, n_days),
    }, index=dates)

    return data


def generate_sample_fundamental_features(n_days=252):
    """Generate sample fundamental features"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Fundamental ratios don't change daily, so we'll use forward fill
    quarterly_dates = pd.date_range(end=datetime.now(), periods=n_days // 60, freq='Q')

    data = pd.DataFrame({
        'PE_ratio': np.random.uniform(10, 30, n_days),
        'PB_ratio': np.random.uniform(1, 5, n_days),
        'PS_ratio': np.random.uniform(0.5, 3, n_days),
        'EV_EBITDA': np.random.uniform(5, 15, n_days),
        'dividend_yield': np.random.uniform(0, 5, n_days),
        'ROE': np.random.uniform(5, 25, n_days),
        'debt_to_equity': np.random.uniform(0.1, 2, n_days),
    }, index=dates)

    # Forward fill to simulate quarterly updates
    return data.resample('5D').mean().reindex(dates).fillna(method='ffill')


def generate_sample_whale_features(n_days=252):
    """Generate sample whale activity features"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    data = pd.DataFrame({
        'whale_buy_volume': np.random.randint(0, 1000000, n_days),
        'whale_sell_volume': np.random.randint(0, 1000000, n_days),
        'institutional_ownership_pct': np.random.uniform(20, 80, n_days),
        'foreign_ownership_pct': np.random.uniform(10, 60, n_days),
        'top_broker_concentration': np.random.uniform(0.2, 0.8, n_days),
        'net_institutional_flow': np.random.randn(n_days) * 100000,
    }, index=dates)

    return data


def example_basic_feature_engineering():
    """Example 1: Basic feature engineering with default config"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Feature Engineering")
    print("=" * 80)

    # Generate sample data
    n_days = 252  # One trading year
    tech_features = generate_sample_technical_features(n_days)

    print(f"\nOriginal technical features shape: {tech_features.shape}")
    print("\nFirst few rows:")
    print(tech_features.head())

    # Create basic config
    config = FeatureConfig(
        scaling_method=ScalingMethod.STANDARD,
        create_lags=True,
        lag_periods=[1, 5, 10, 20],
        create_rolling=False,
        create_differences=False,
        imputation_method=ImputationMethod.FORWARD_FILL
    )

    # Initialize feature engineer
    engineer = FeatureEngineer(config=config)

    # Transform features
    engineered_features = engineer.fit_transform(tech_features)

    print(f"\nEngineered features shape: {engineered_features.shape}")
    print(f"Number of new features created: {engineered_features.shape[1] - tech_features.shape[1]}")
    print("\nFeature names (first 20):")
    print(list(engineered_features.columns[:20]))


def example_combine_all_features():
    """Example 2: Combine technical, fundamental, and whale features"""
    print("\n" + "=" * 80)
    print("Example 2: Combining Multiple Feature Types")
    print("=" * 80)

    # Generate all feature types
    n_days = 252
    tech_features = generate_sample_technical_features(n_days)
    fund_features = generate_sample_fundamental_features(n_days)
    whale_features = generate_sample_whale_features(n_days)

    print(f"\nTechnical features: {tech_features.shape[1]} columns")
    print(f"Fundamental features: {fund_features.shape[1]} columns")
    print(f"Whale features: {whale_features.shape[1]} columns")

    # Configure feature engineering
    config = FeatureConfig(
        scaling_method=ScalingMethod.ROBUST,  # Robust to outliers
        create_lags=True,
        lag_periods=[1, 5, 10],
        create_rolling=True,
        rolling_windows=[5, 20],
        create_differences=True,
        diff_periods=[1, 5],
        imputation_method=ImputationMethod.FORWARD_FILL,
        drop_na=True
    )

    # Initialize feature engineer
    engineer = FeatureEngineer(config=config)

    # Combine and transform all features
    engineered_features = engineer.fit_transform(
        X=None,
        technical_features=tech_features,
        fundamental_features=fund_features,
        whale_features=whale_features
    )

    print(f"\nCombined engineered features shape: {engineered_features.shape}")
    print(f"\nFeature prefixes:")
    prefixes = set([col.split('_')[0] for col in engineered_features.columns])
    print(prefixes)

    # Get feature statistics
    stats = get_feature_statistics(engineered_features)
    print("\nFeature statistics (first 10 features):")
    print(stats.head(10))


def example_advanced_feature_engineering():
    """Example 3: Advanced feature engineering with interactions and selection"""
    print("\n" + "=" * 80)
    print("Example 3: Advanced Feature Engineering")
    print("=" * 80)

    # Generate data
    n_days = 500
    tech_features = generate_sample_technical_features(n_days)

    # Add time features
    tech_features = create_time_features(tech_features, datetime_col='date')

    print(f"\nFeatures with time components: {tech_features.shape[1]} columns")

    # Define feature interactions
    interaction_pairs = [
        ('close', 'volume'),
        ('RSI_14', 'MACD'),
        ('BB_upper', 'BB_lower'),
        ('ATR_14', 'volume')
    ]

    # Configure with advanced options
    config = FeatureConfig(
        scaling_method=ScalingMethod.STANDARD,
        create_lags=True,
        lag_periods=[1, 5, 10, 20],
        lag_features=['close', 'volume', 'RSI_14'],  # Only lag specific features
        create_rolling=True,
        rolling_windows=[5, 10, 20, 60],
        rolling_features=['close', 'volume'],  # Only rolling for specific features
        create_differences=True,
        diff_periods=[1, 5, 20],
        create_interactions=True,
        interaction_pairs=interaction_pairs,
        imputation_method=ImputationMethod.FORWARD_FILL,
        feature_selection=False,  # Disable for this example
        drop_na=True
    )

    # Initialize and transform
    engineer = FeatureEngineer(config=config)

    # Create target variable (predict if price will go up)
    target = (tech_features['close'].shift(-1) > tech_features['close']).astype(int)

    # Fit and transform
    X = engineer.fit_transform(tech_features[:-1])
    y = target[:-1]

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # Remove highly correlated features
    X_reduced = remove_correlated_features(X, threshold=0.95)
    print(f"\nAfter removing correlated features: {X_reduced.shape}")

    # Show some interaction features
    interaction_cols = [col for col in X.columns if '_x_' in col or '_div_' in col]
    print(f"\nInteraction features created: {len(interaction_cols)}")
    print("Sample interaction features:")
    print(interaction_cols[:10])


def example_transform_new_data():
    """Example 4: Fit on training data, transform test data"""
    print("\n" + "=" * 80)
    print("Example 4: Train-Test Split with Consistent Transformation")
    print("=" * 80)

    # Generate data
    n_days = 500
    tech_features = generate_sample_technical_features(n_days)
    fund_features = generate_sample_fundamental_features(n_days)

    # Split into train and test
    train_size = int(n_days * 0.8)
    tech_train = tech_features.iloc[:train_size]
    tech_test = tech_features.iloc[train_size:]
    fund_train = fund_features.iloc[:train_size]
    fund_test = fund_features.iloc[train_size:]

    print(f"\nTraining data: {train_size} days")
    print(f"Test data: {n_days - train_size} days")

    # Configure
    config = FeatureConfig(
        scaling_method=ScalingMethod.STANDARD,
        create_lags=True,
        lag_periods=[1, 5, 10],
        create_rolling=True,
        rolling_windows=[5, 20],
        imputation_method=ImputationMethod.FORWARD_FILL,
        drop_na=True
    )

    # Initialize
    engineer = FeatureEngineer(config=config)

    # Fit on training data
    X_train = engineer.fit_transform(
        X=None,
        technical_features=tech_train,
        fundamental_features=fund_train
    )

    # Transform test data using fitted transformers
    X_test = engineer.transform(
        X=None,
        technical_features=tech_test,
        fundamental_features=fund_test
    )

    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature count matches: {X_train.shape[1] == X_test.shape[1]}")

    # Verify scaling was applied consistently
    print("\nTraining data statistics (first feature):")
    print(f"Mean: {X_train.iloc[:, 0].mean():.4f}, Std: {X_train.iloc[:, 0].std():.4f}")
    print("\nTest data statistics (first feature):")
    print(f"Mean: {X_test.iloc[:, 0].mean():.4f}, Std: {X_test.iloc[:, 0].std():.4f}")


def example_custom_pipeline():
    """Example 5: Building a custom feature engineering pipeline"""
    print("\n" + "=" * 80)
    print("Example 5: Custom Feature Engineering Pipeline")
    print("=" * 80)

    # Generate data
    tech_features = generate_sample_technical_features(300)

    # Step-by-step custom pipeline
    engineer = FeatureEngineer(config=FeatureConfig(
        create_lags=False,
        create_rolling=False,
        create_differences=False,
        create_interactions=False,
        scaling_method=ScalingMethod.NONE
    ))

    print("\nOriginal shape:", tech_features.shape)

    # Step 1: Create time features
    df = create_time_features(tech_features)
    print("After time features:", df.shape)

    # Step 2: Create specific lag features
    config = FeatureConfig(lag_periods=[1, 5, 10], lag_features=['close', 'volume'])
    engineer.config = config
    df = engineer.create_lag_features(df)
    print("After lag features:", df.shape)

    # Step 3: Create rolling features
    config.rolling_windows = [5, 20]
    config.rolling_features = ['close', 'RSI_14']
    engineer.config = config
    df = engineer.create_rolling_features(df)
    print("After rolling features:", df.shape)

    # Step 4: Create difference features
    config.diff_periods = [1]
    engineer.config = config
    df = engineer.create_difference_features(df)
    print("After difference features:", df.shape)

    # Step 5: Handle missing values
    df = engineer.handle_missing_values(df, fit=True)
    print("After imputation:", df.shape)

    # Step 6: Remove highly correlated features
    df_reduced = remove_correlated_features(df, threshold=0.9)
    print("After correlation filter:", df_reduced.shape)

    # Step 7: Scale features
    config.scaling_method = ScalingMethod.STANDARD
    engineer.config = config
    df_scaled = engineer.scale_features(df_reduced, fit=True)
    print("After scaling:", df_scaled.shape)

    print("\nFinal feature matrix created!")
    print("\nSample features:")
    print(df_scaled.iloc[:5, :10])


def main():
    """Run all examples"""
    print("=" * 80)
    print("BIST AI Trading System - Feature Engineering Examples")
    print("=" * 80)

    try:
        # Run all examples
        example_basic_feature_engineering()
        example_combine_all_features()
        example_advanced_feature_engineering()
        example_transform_new_data()
        example_custom_pipeline()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
