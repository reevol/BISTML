"""
Examples for Trend Indicators Module

This file demonstrates various usage patterns for the trend indicators module.
Run this file to see the indicators in action with sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import from the trend module
from trend import (
    TrendIndicators,
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_hma,
    calculate_ichimoku,
    calculate_trend_signals
)


def create_sample_data(days=200, start_price=100):
    """
    Create realistic sample OHLCV data for testing.

    Args:
        days: Number of days of data to generate
        start_price: Starting price

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(
        end=datetime.now(),
        periods=days,
        freq='D'
    )

    np.random.seed(42)

    # Create a realistic price trend with some volatility
    trend = np.linspace(start_price, start_price * 1.5, days)
    noise = np.random.randn(days).cumsum() * 2
    close = trend + noise

    # Generate OHLC from close
    open_price = close + np.random.randn(days) * 1
    high = np.maximum(open_price, close) + abs(np.random.randn(days)) * 2
    low = np.minimum(open_price, close) - abs(np.random.randn(days)) * 2
    volume = np.random.randint(1000000, 10000000, days)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


def example1_basic_moving_averages():
    """Example 1: Calculate basic moving averages."""
    print("\n" + "="*60)
    print("Example 1: Basic Moving Averages")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=100)

    # Initialize calculator
    calc = TrendIndicators(data, price_column='close')

    # Calculate different MAs
    sma_20 = calc.sma(period=20)
    ema_20 = calc.ema(period=20)
    wma_20 = calc.wma(period=20)
    hma_20 = calc.hma(period=20)

    # Display results
    print(f"\nLast 5 values:")
    results = pd.DataFrame({
        'Close': data['close'],
        'SMA_20': sma_20,
        'EMA_20': ema_20,
        'WMA_20': wma_20,
        'HMA_20': hma_20
    })
    print(results.tail())

    return results


def example2_multiple_timeframes():
    """Example 2: Calculate multiple timeframes at once."""
    print("\n" + "="*60)
    print("Example 2: Multiple Timeframes")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=250)

    # Calculate multiple SMAs
    sma_multi = calculate_sma(
        data,
        timeframes=[10, 20, 50, 100, 200]
    )

    print(f"\nCalculated SMAs for timeframes: {list(sma_multi.columns)}")
    print(f"\nLast 5 values:")
    print(sma_multi.tail())

    # Check for golden cross (50 SMA crosses above 200 SMA)
    golden_cross = (sma_multi['SMA_50'] > sma_multi['SMA_200']).iloc[-1]
    print(f"\nGolden Cross detected: {golden_cross}")

    return sma_multi


def example3_ichimoku_cloud():
    """Example 3: Ichimoku Cloud analysis."""
    print("\n" + "="*60)
    print("Example 3: Ichimoku Cloud")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=150)

    # Calculate Ichimoku
    ichimoku = calculate_ichimoku(data)

    print(f"\nIchimoku components: {list(ichimoku.columns)}")
    print(f"\nLast 5 values:")
    print(ichimoku.tail())

    # Analyze current market position
    last_row = ichimoku.iloc[-1]
    current_price = data['close'].iloc[-1]

    print(f"\n--- Market Analysis ---")
    print(f"Current Price: {current_price:.2f}")
    print(f"Tenkan-sen: {last_row['tenkan_sen']:.2f}")
    print(f"Kijun-sen: {last_row['kijun_sen']:.2f}")
    print(f"Cloud Top: {last_row['cloud_top']:.2f}")
    print(f"Cloud Bottom: {last_row['cloud_bottom']:.2f}")

    if not pd.isna(last_row['cloud_top']) and not pd.isna(last_row['cloud_bottom']):
        if current_price > last_row['cloud_top']:
            print("Position: ABOVE the cloud (Bullish)")
        elif current_price < last_row['cloud_bottom']:
            print("Position: BELOW the cloud (Bearish)")
        else:
            print("Position: INSIDE the cloud (Neutral/Consolidation)")

    if last_row['cloud_green']:
        print("Cloud Color: GREEN (Bullish)")
    else:
        print("Cloud Color: RED (Bearish)")

    return ichimoku


def example4_trend_signals():
    """Example 4: Generate trend signals."""
    print("\n" + "="*60)
    print("Example 4: Trend Signals (MA Crossovers)")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=100)

    # Calculate trend signals using EMA crossover
    signals = calculate_trend_signals(
        data,
        fast_period=12,
        slow_period=26,
        indicator='ema'
    )

    # Add MAs for visualization
    calc = TrendIndicators(data)
    ema_12 = calc.ema(period=12)
    ema_26 = calc.ema(period=26)

    results = pd.DataFrame({
        'Close': data['close'],
        'EMA_12': ema_12,
        'EMA_26': ema_26,
        'Signal': signals
    })

    print(f"\nLast 10 values:")
    print(results.tail(10))

    # Detect crossovers
    signal_changes = signals.diff()
    bullish_cross = signal_changes[signal_changes == 2].index
    bearish_cross = signal_changes[signal_changes == -2].index

    print(f"\nBullish crossovers detected: {len(bullish_cross)}")
    print(f"Bearish crossovers detected: {len(bearish_cross)}")

    if len(bullish_cross) > 0:
        print(f"Last bullish crossover: {bullish_cross[-1]}")
    if len(bearish_cross) > 0:
        print(f"Last bearish crossover: {bearish_cross[-1]}")

    current_signal = signals.iloc[-1]
    if current_signal == 1:
        print(f"\nCurrent Trend: BULLISH (EMA 12 > EMA 26)")
    elif current_signal == -1:
        print(f"\nCurrent Trend: BEARISH (EMA 12 < EMA 26)")
    else:
        print(f"\nCurrent Trend: NEUTRAL")

    return results


def example5_multi_indicator_analysis():
    """Example 5: Comprehensive multi-indicator analysis."""
    print("\n" + "="*60)
    print("Example 5: Multi-Indicator Analysis")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=150)

    # Initialize calculator
    calc = TrendIndicators(data)

    # Calculate multiple indicators across timeframes
    analysis = calc.multi_timeframe_analysis(
        indicators=['sma', 'ema', 'wma', 'hma'],
        timeframes={
            'sma': [20, 50, 200],
            'ema': [12, 26, 50],
            'wma': [10, 20],
            'hma': [9, 16]
        }
    )

    print(f"\nCalculated {len(analysis.columns)} indicators")
    print(f"Indicators: {list(analysis.columns)}")
    print(f"\nLast 5 values:")
    print(analysis.tail())

    # Analyze trend strength
    last_price = data['close'].iloc[-1]
    above_ma_count = 0
    total_mas = len(analysis.columns)

    for col in analysis.columns:
        if not pd.isna(analysis[col].iloc[-1]) and last_price > analysis[col].iloc[-1]:
            above_ma_count += 1

    trend_strength = (above_ma_count / total_mas) * 100

    print(f"\n--- Trend Strength Analysis ---")
    print(f"Current Price: {last_price:.2f}")
    print(f"Price above {above_ma_count}/{total_mas} MAs")
    print(f"Trend Strength: {trend_strength:.1f}%")

    if trend_strength > 70:
        print("Assessment: STRONG UPTREND")
    elif trend_strength > 50:
        print("Assessment: MODERATE UPTREND")
    elif trend_strength > 30:
        print("Assessment: WEAK/NEUTRAL")
    else:
        print("Assessment: DOWNTREND")

    return analysis


def example6_convenience_functions():
    """Example 6: Using convenience functions."""
    print("\n" + "="*60)
    print("Example 6: Convenience Functions")
    print("="*60)

    # Create sample data
    data = create_sample_data(days=100)

    # Use standalone functions
    print("\nUsing standalone convenience functions:")

    sma = calculate_sma(data, period=20)
    print(f"SMA(20) last value: {sma.iloc[-1]:.2f}")

    ema = calculate_ema(data, period=20)
    print(f"EMA(20) last value: {ema.iloc[-1]:.2f}")

    wma = calculate_wma(data, period=20)
    print(f"WMA(20) last value: {wma.iloc[-1]:.2f}")

    hma = calculate_hma(data, period=20)
    print(f"HMA(20) last value: {hma.iloc[-1]:.2f}")

    # Multi-timeframe with convenience function
    print("\nMultiple timeframes with convenience function:")
    sma_multi = calculate_sma(data, timeframes=[10, 20, 50])
    print(sma_multi.tail())

    return {
        'sma': sma,
        'ema': ema,
        'wma': wma,
        'hma': hma
    }


def run_all_examples():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Trend Indicators Module - Complete Examples")
    print("#"*60)

    try:
        example1_basic_moving_averages()
        example2_multiple_timeframes()
        example3_ichimoku_cloud()
        example4_trend_signals()
        example5_multi_indicator_analysis()
        example6_convenience_functions()

        print("\n" + "#"*60)
        print("# All examples completed successfully!")
        print("#"*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
