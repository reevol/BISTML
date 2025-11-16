"""
Features Module - BIST AI Trading System

This module provides feature extraction and engineering capabilities:
- Technical indicators (momentum, trend, volatility, advanced)
- Fundamental metrics (valuation ratios)
- Whale activity features
- Feature engineering pipeline

Author: BIST AI Trading System
"""

from .feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
    ScalingMethod,
    EncodingMethod,
    ImputationMethod,
    LagFeatureTransformer,
    RollingFeatureTransformer,
    DifferenceFeatureTransformer,
    FeatureInteractionTransformer,
    create_time_features,
    remove_correlated_features,
    get_feature_statistics
)

__all__ = [
    'FeatureEngineer',
    'FeatureConfig',
    'ScalingMethod',
    'EncodingMethod',
    'ImputationMethod',
    'LagFeatureTransformer',
    'RollingFeatureTransformer',
    'DifferenceFeatureTransformer',
    'FeatureInteractionTransformer',
    'create_time_features',
    'remove_correlated_features',
    'get_feature_statistics'
]
