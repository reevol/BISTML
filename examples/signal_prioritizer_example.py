"""
Signal Prioritizer Usage Examples

This example demonstrates how to use the SignalPrioritizer to rank trading
signals based on multiple factors including confidence scores, WAI scores,
news sentiment, and model agreement.

The prioritizer helps identify the highest-quality signals by combining
various metrics into a single, actionable priority score.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.prioritizer import (
    SignalPrioritizer,
    SignalInput,
    SignalDirection,
    PrioritizationStrategy,
    create_signal_input,
    prioritize_signals
)


def example_1_basic_prioritization():
    """Example 1: Basic signal prioritization with balanced strategy"""
    print("=" * 80)
    print("Example 1: Basic Signal Prioritization")
    print("=" * 80)

    # Create sample signals
    signals = [
        create_signal_input(
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
        ),
        create_signal_input(
            symbol='AKBNK',
            signal_direction='BUY',
            confidence_score=72.0,
            wai_score=45.0,
            news_sentiment=0.2,
            model_predictions={
                'LSTM': 0.015,
                'GRU': 0.018,
                'XGBoost': -0.002,  # One model disagrees
                'LightGBM': 0.012
            },
            current_price=50.0,
            target_price=51.0
        ),
        create_signal_input(
            symbol='GARAN',
            signal_direction='SELL',
            confidence_score=68.0,
            wai_score=62.0,
            news_sentiment=-0.4,
            model_predictions={
                'LSTM': -0.025,
                'GRU': -0.022,
                'XGBoost': -0.028,
                'LightGBM': -0.024
            },
            current_price=95.0,
            target_price=92.5
        ),
        create_signal_input(
            symbol='ISCTR',
            signal_direction='STRONG_BUY',
            confidence_score=91.0,
            wai_score=88.0,
            news_sentiment=0.8,
            model_predictions={
                'LSTM': 0.045,
                'GRU': 0.042,
                'XGBoost': 0.048,
                'LightGBM': 0.046
            },
            current_price=8.5,
            target_price=9.0
        )
    ]

    # Prioritize signals
    prioritizer = SignalPrioritizer(strategy=PrioritizationStrategy.BALANCED)
    prioritized = prioritizer.prioritize_signals(signals)

    # Display results
    df = prioritizer.to_dataframe(prioritized)
    print("\nPrioritized Signals (Balanced Strategy):")
    print(df.to_string(index=False))

    print("\n" + "=" * 80)


def example_2_strategy_comparison():
    """Example 2: Compare different prioritization strategies"""
    print("\n" + "=" * 80)
    print("Example 2: Strategy Comparison")
    print("=" * 80)

    # Create a signal with mixed metrics
    signal = create_signal_input(
        symbol='SAHOL',
        signal_direction='BUY',
        confidence_score=70.0,
        wai_score=85.0,  # High whale activity
        news_sentiment=0.3,  # Moderate positive sentiment
        model_predictions={
            'LSTM': 0.020,
            'GRU': 0.018,
            'XGBoost': 0.022,
            'LightGBM': 0.019
        },
        current_price=75.0,
        target_price=77.0
    )

    strategies = [
        PrioritizationStrategy.BALANCED,
        PrioritizationStrategy.WHALE_FOCUSED,
        PrioritizationStrategy.CONFIDENCE_FOCUSED,
        PrioritizationStrategy.SENTIMENT_FOCUSED,
        PrioritizationStrategy.CONSENSUS_FOCUSED,
        PrioritizationStrategy.AGGRESSIVE,
        PrioritizationStrategy.CONSERVATIVE
    ]

    results = []
    for strategy in strategies:
        prioritizer = SignalPrioritizer(strategy=strategy, verbose=False)
        prioritized = prioritizer.prioritize_signal(signal)

        results.append({
            'Strategy': strategy.value,
            'Priority Score': round(prioritized.priority_score, 2),
            'Risk Adjusted': round(prioritized.risk_adjusted_score, 2),
            'Signal Strength': prioritized.signal_strength
        })

    df_strategies = pd.DataFrame(results)
    print("\nSame Signal Prioritized with Different Strategies:")
    print(df_strategies.to_string(index=False))

    print("\n" + "=" * 80)


def example_3_whale_focused_trading():
    """Example 3: Whale-focused strategy for institutional flow trading"""
    print("\n" + "=" * 80)
    print("Example 3: Whale-Focused Strategy")
    print("=" * 80)

    # Create signals with varying whale activity levels
    signals = [
        create_signal_input(
            symbol='VESTL',
            signal_direction='BUY',
            confidence_score=75.0,
            wai_score=92.0,  # Very high whale activity
            news_sentiment=0.1,
            model_predictions={'LSTM': 0.020, 'GRU': 0.018},
            current_price=45.0,
            target_price=46.5
        ),
        create_signal_input(
            symbol='KRDMD',
            signal_direction='BUY',
            confidence_score=82.0,
            wai_score=35.0,  # Low whale activity
            news_sentiment=0.4,
            model_predictions={'LSTM': 0.025, 'GRU': 0.023},
            current_price=12.0,
            target_price=12.5
        ),
        create_signal_input(
            symbol='FROTO',
            signal_direction='BUY',
            confidence_score=78.0,
            wai_score=65.0,  # Moderate whale activity
            news_sentiment=0.3,
            model_predictions={'LSTM': 0.022, 'GRU': 0.021},
            current_price=85.0,
            target_price=87.0
        )
    ]

    # Prioritize with whale-focused strategy
    prioritizer = SignalPrioritizer(
        strategy=PrioritizationStrategy.WHALE_FOCUSED,
        min_wai_threshold=40.0  # Only signals with WAI > 40
    )

    prioritized = prioritizer.prioritize_signals(signals)
    df = prioritizer.to_dataframe(prioritized)

    print("\nWhale-Focused Prioritization (min WAI = 40):")
    print(df[['rank', 'symbol', 'signal', 'risk_adjusted_score',
              'wai_component', 'confidence_component']].to_string(index=False))

    print("\nNote: KRDMD filtered out due to low WAI score")
    print("=" * 80)


def example_4_model_agreement_analysis():
    """Example 4: Analyzing model agreement impact"""
    print("\n" + "=" * 80)
    print("Example 4: Model Agreement Analysis")
    print("=" * 80)

    # Create signals with different levels of model agreement
    signals = [
        create_signal_input(
            symbol='EREGL',
            signal_direction='BUY',
            confidence_score=80.0,
            wai_score=70.0,
            news_sentiment=0.3,
            model_predictions={
                'LSTM': 0.025,
                'GRU': 0.023,
                'XGBoost': 0.026,
                'LightGBM': 0.024
            }  # All models agree (100%)
        ),
        create_signal_input(
            symbol='SISE',
            signal_direction='BUY',
            confidence_score=80.0,
            wai_score=70.0,
            news_sentiment=0.3,
            model_predictions={
                'LSTM': 0.025,
                'GRU': 0.020,
                'XGBoost': -0.005,  # Disagrees
                'LightGBM': 0.022
            }  # 75% agreement
        ),
        create_signal_input(
            symbol='TUPRS',
            signal_direction='BUY',
            confidence_score=80.0,
            wai_score=70.0,
            news_sentiment=0.3,
            model_predictions={
                'LSTM': 0.025,
                'GRU': -0.010,  # Disagrees
                'XGBoost': -0.008,  # Disagrees
                'LightGBM': 0.022
            }  # 50% agreement
        )
    ]

    # Prioritize with consensus-focused strategy
    prioritizer = SignalPrioritizer(
        strategy=PrioritizationStrategy.CONSENSUS_FOCUSED,
        min_agreement_threshold=0.6  # Require 60%+ agreement
    )

    prioritized = prioritizer.prioritize_signals(signals)
    df = prioritizer.to_dataframe(prioritized)

    print("\nConsensus-Focused Prioritization (min agreement = 60%):")
    print(df[['rank', 'symbol', 'risk_adjusted_score', 'model_agreement_pct',
              'agreement_component']].to_string(index=False))

    print("\nNote: TUPRS filtered out due to low model agreement")
    print("=" * 80)


def example_5_custom_weights():
    """Example 5: Using custom factor weights"""
    print("\n" + "=" * 80)
    print("Example 5: Custom Factor Weights")
    print("=" * 80)

    signal = create_signal_input(
        symbol='PETKM',
        signal_direction='STRONG_BUY',
        confidence_score=85.0,
        wai_score=75.0,
        news_sentiment=0.5,
        model_predictions={
            'LSTM': 0.030,
            'GRU': 0.028,
            'XGBoost': 0.032,
            'LightGBM': 0.029
        }
    )

    # Define custom weights (must sum to 1.0)
    custom_weights = {
        'confidence': 0.50,  # Very high weight on confidence
        'wai': 0.10,
        'sentiment': 0.10,
        'agreement': 0.30
    }

    prioritizer = SignalPrioritizer(custom_weights=custom_weights)
    prioritized = prioritizer.prioritize_signal(signal)

    print("\nCustom Weights:")
    print(f"  Confidence: {custom_weights['confidence']*100}%")
    print(f"  WAI: {custom_weights['wai']*100}%")
    print(f"  Sentiment: {custom_weights['sentiment']*100}%")
    print(f"  Agreement: {custom_weights['agreement']*100}%")

    print(f"\nPrioritized Signal:")
    print(f"  Symbol: {prioritized.symbol}")
    print(f"  Priority Score: {prioritized.priority_score:.2f}")
    print(f"  Risk-Adjusted Score: {prioritized.risk_adjusted_score:.2f}")
    print(f"  Signal Strength: {prioritized.signal_strength}")

    print("\nComponent Breakdown:")
    print(f"  Confidence Component: {prioritized.confidence_component:.2f}")
    print(f"  WAI Component: {prioritized.wai_component:.2f}")
    print(f"  Sentiment Component: {prioritized.sentiment_component:.2f}")
    print(f"  Agreement Component: {prioritized.agreement_component:.2f}")

    print("=" * 80)


def example_6_quick_prioritization():
    """Example 6: Quick prioritization using convenience function"""
    print("\n" + "=" * 80)
    print("Example 6: Quick Prioritization (Convenience Function)")
    print("=" * 80)

    # Create multiple signals quickly
    signals = [
        create_signal_input('ASELS', 'STRONG_BUY', 88.0, 82.0, 0.7,
                          {'LSTM': 0.04, 'GRU': 0.038, 'XGBoost': 0.042}),
        create_signal_input('BIMAS', 'BUY', 75.0, 55.0, 0.2,
                          {'LSTM': 0.02, 'GRU': 0.018, 'XGBoost': 0.015}),
        create_signal_input('KOZAL', 'SELL', 70.0, 68.0, -0.3,
                          {'LSTM': -0.02, 'GRU': -0.022, 'XGBoost': -0.018}),
    ]

    # Quick prioritization with default settings
    df = prioritize_signals(signals, strategy='balanced', top_n=5)

    print("\nQuick Prioritization Results:")
    print(df[['rank', 'symbol', 'signal', 'priority_score',
              'signal_strength']].to_string(index=False))

    print("=" * 80)


def example_7_real_world_scenario():
    """Example 7: Real-world trading scenario with multiple signals"""
    print("\n" + "=" * 80)
    print("Example 7: Real-World Trading Scenario")
    print("=" * 80)

    # Simulate a real trading day with multiple signals from different sectors
    print("\nScenario: Market open at 10:00 AM")
    print("Multiple signals generated from ML models, whale analysis, and sentiment")

    signals = [
        # Banking sector
        create_signal_input(
            'AKBNK', 'BUY', 78.0, 65.0, 0.3,
            {'LSTM': 0.022, 'GRU': 0.020, 'XGBoost': 0.024, 'LightGBM': 0.021},
            current_price=50.25, target_price=51.50
        ),
        create_signal_input(
            'GARAN', 'STRONG_BUY', 85.0, 78.0, 0.6,
            {'LSTM': 0.035, 'GRU': 0.033, 'XGBoost': 0.036, 'LightGBM': 0.034},
            current_price=95.50, target_price=98.80
        ),
        create_signal_input(
            'ISCTR', 'BUY', 72.0, 48.0, 0.1,
            {'LSTM': 0.015, 'GRU': 0.012, 'XGBoost': 0.018, 'LightGBM': -0.002},
            current_price=8.45, target_price=8.65
        ),

        # Industrial sector
        create_signal_input(
            'EREGL', 'STRONG_BUY', 90.0, 85.0, 0.7,
            {'LSTM': 0.040, 'GRU': 0.038, 'XGBoost': 0.042, 'LightGBM': 0.039},
            current_price=45.30, target_price=47.20
        ),
        create_signal_input(
            'THYAO', 'BUY', 76.0, 58.0, 0.2,
            {'LSTM': 0.018, 'GRU': 0.016, 'XGBoost': 0.020, 'LightGBM': 0.017},
            current_price=100.00, target_price=102.00
        ),

        # Energy sector
        create_signal_input(
            'TUPRS', 'STRONG_SELL', 82.0, 72.0, -0.5,
            {'LSTM': -0.030, 'GRU': -0.028, 'XGBoost': -0.032, 'LightGBM': -0.029},
            current_price=125.50, target_price=121.00
        ),

        # Technology/Services
        create_signal_input(
            'ASELS', 'STRONG_BUY', 92.0, 88.0, 0.8,
            {'LSTM': 0.045, 'GRU': 0.043, 'XGBoost': 0.047, 'LightGBM': 0.044},
            current_price=68.75, target_price=72.00
        ),
    ]

    # Use conservative strategy (prefer high agreement and proven patterns)
    prioritizer = SignalPrioritizer(
        strategy=PrioritizationStrategy.CONSERVATIVE,
        min_confidence_threshold=70.0,
        min_wai_threshold=50.0,
        min_agreement_threshold=0.7
    )

    # Get BUY signals only
    all_prioritized = prioritizer.prioritize_signals(signals, return_all=True)
    buy_signals = [s for s in all_prioritized
                   if s.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]]

    df_buy = prioritizer.to_dataframe(buy_signals)

    print("\nTop BUY Signals (Conservative Strategy):")
    print(df_buy[['rank', 'symbol', 'signal', 'risk_adjusted_score',
                  'expected_return_pct', 'model_agreement_pct']].to_string(index=False))

    # Get SELL signals only
    sell_signals = [s for s in all_prioritized
                    if s.signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]]

    df_sell = prioritizer.to_dataframe(sell_signals)

    print("\nTop SELL Signals (Conservative Strategy):")
    print(df_sell[['rank', 'symbol', 'signal', 'risk_adjusted_score',
                   'expected_return_pct', 'model_agreement_pct']].to_string(index=False))

    print("\nTrading Recommendations:")
    if len(buy_signals) > 0:
        top_buy = buy_signals[0]
        print(f"  🟢 PRIMARY BUY: {top_buy.symbol}")
        print(f"     - Signal Strength: {top_buy.signal_strength}")
        print(f"     - Expected Return: {top_buy.expected_return:.2f}%")
        print(f"     - Model Agreement: {top_buy.model_agreement_pct*100:.1f}%")

    if len(sell_signals) > 0:
        top_sell = sell_signals[0]
        print(f"  🔴 PRIMARY SELL: {top_sell.symbol}")
        print(f"     - Signal Strength: {top_sell.signal_strength}")
        print(f"     - Expected Return: {top_sell.expected_return:.2f}%")
        print(f"     - Model Agreement: {top_sell.model_agreement_pct*100:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SIGNAL PRIORITIZER - COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_1_basic_prioritization()
    example_2_strategy_comparison()
    example_3_whale_focused_trading()
    example_4_model_agreement_analysis()
    example_5_custom_weights()
    example_6_quick_prioritization()
    example_7_real_world_scenario()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
