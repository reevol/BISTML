"""
Example Usage of Whale Activity Index (WAI) Module

This script demonstrates how to use the WhaleActivityIndex class to analyze
institutional trading activity on BIST stocks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the WhaleActivityIndex class
from activity_index import WhaleActivityIndex, calculate_wai, generate_whale_signals


def generate_sample_brokerage_data(symbol='THYAO', days=60):
    """
    Generate sample brokerage data for testing.

    In production, this would come from the WhaleCollector data source.
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    brokers = ['BRK001', 'BRK002', 'BRK003', 'BRK004', 'BRK005',
               'BRK006', 'BRK007', 'BRK008', 'BRK009', 'BRK010',
               'BRK011', 'BRK012', 'BRK013', 'BRK014', 'BRK015']

    data = []

    for date in dates:
        for broker in brokers:
            # Simulate realistic broker trading data
            base_volume = np.random.randint(100000, 2000000)

            # Top brokers have larger volumes
            if broker in ['BRK001', 'BRK002', 'BRK003']:
                base_volume *= 3

            # Simulate buying and selling with some net flow
            buy_volume = base_volume + np.random.randint(-50000, 100000)
            sell_volume = base_volume + np.random.randint(-100000, 50000)

            # Occasionally simulate whale activity
            if np.random.random() > 0.9 and broker in ['BRK001', 'BRK002']:
                buy_volume *= 2  # Unusual buying

            data.append({
                'date': date,
                'symbol': symbol,
                'broker_code': broker,
                'buy_volume': max(0, buy_volume),
                'sell_volume': max(0, sell_volume),
                'net_volume': buy_volume - sell_volume,
                'buy_value': buy_volume * 10.5,  # Avg price
                'sell_value': sell_volume * 10.5
            })

    return pd.DataFrame(data)


def generate_sample_price_data(symbol='THYAO', days=60):
    """
    Generate sample price data for testing.

    In production, this would come from the BIST data collector.
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Simulate price movement
    base_price = 10.0
    prices = []

    for i in range(len(dates)):
        change = np.random.randn() * 0.02  # 2% daily volatility
        base_price *= (1 + change)

        daily_high = base_price * (1 + abs(np.random.randn()) * 0.01)
        daily_low = base_price * (1 - abs(np.random.randn()) * 0.01)
        daily_open = base_price * (1 + np.random.randn() * 0.005)

        prices.append({
            'date': dates[i],
            'symbol': symbol,
            'open': daily_open,
            'high': daily_high,
            'low': daily_low,
            'close': base_price,
            'volume': np.random.randint(5000000, 20000000)
        })

    return pd.DataFrame(prices)


def example_1_basic_wai_calculation():
    """Example 1: Basic WAI Calculation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Whale Activity Index Calculation")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('THYAO', days=60)

    # Initialize WhaleActivityIndex
    wai = WhaleActivityIndex(
        brokerage_data=brokerage_data,
        top_n_brokers=10,
        lookback_period=20
    )

    # Calculate WAI for THYAO
    wai_scores = wai.calculate_wai(
        symbol='THYAO',
        normalize=True,
        include_components=True
    )

    print("\nWAI Scores (Last 10 days):")
    print(wai_scores[['date', 'wai_score', 'wai_directional',
                      'whale_participation_pct', 'net_flow_ratio']].tail(10))

    print("\nLatest WAI Metrics:")
    latest = wai_scores.iloc[-1]
    print(f"  WAI Score: {latest['wai_score']:.2f}")
    print(f"  Directional WAI: {latest['wai_directional']:.2f}")
    print(f"  Whale Participation: {latest['whale_participation_pct']:.2f}%")
    print(f"  Net Flow Ratio: {latest['net_flow_ratio']:.4f}")


def example_2_unusual_activity_detection():
    """Example 2: Detect Unusual Whale Activity"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Unusual Whale Activity Detection")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('AKBNK', days=90)

    # Initialize WAI
    wai = WhaleActivityIndex(
        brokerage_data=brokerage_data,
        top_n_brokers=10,
        lookback_period=20
    )

    # Detect unusual activity
    unusual = wai.detect_unusual_activity(
        symbol='AKBNK',
        z_threshold=2.0,
        min_participation=15.0
    )

    # Filter for unusual activity days
    unusual_days = unusual[unusual['unusual_activity'] == True]

    print(f"\nFound {len(unusual_days)} days with unusual whale activity")

    if len(unusual_days) > 0:
        print("\nUnusual Activity Days:")
        print(unusual_days[['date', 'activity_type', 'unusual_strength',
                           'net_flow_zscore', 'whale_participation_pct']].head(10))


def example_3_price_flow_discrepancy():
    """Example 3: Price-Flow Discrepancy Analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Price-Flow Discrepancy Detection")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('GARAN', days=60)
    price_data = generate_sample_price_data('GARAN', days=60)

    # Initialize WAI with price data
    wai = WhaleActivityIndex(
        brokerage_data=brokerage_data,
        price_data=price_data,
        top_n_brokers=10
    )

    # Detect price-flow discrepancies
    discrepancy = wai.detect_price_flow_discrepancy(
        symbol='GARAN',
        discrepancy_threshold=1.5,
        window=5
    )

    print("\nPrice-Flow Analysis (Last 10 days):")
    print(discrepancy[['date', 'price_change', 'net_flow_ratio',
                       'discrepancy_zscore', 'potential_manipulation']].tail(10))

    # Find potential manipulation cases
    manipulation = discrepancy[discrepancy['potential_manipulation'] == True]

    if len(manipulation) > 0:
        print(f"\nFound {len(manipulation)} potential manipulation events:")
        print(manipulation[['date', 'accumulation_under_pressure',
                           'distribution_during_pump', 'manipulation_score']].head(5))


def example_4_accumulation_distribution():
    """Example 4: Accumulation/Distribution Analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Accumulation/Distribution Phase Detection")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('EREGL', days=60)

    # Initialize WAI
    wai = WhaleActivityIndex(brokerage_data=brokerage_data)

    # Calculate accumulation/distribution scores
    ad_scores = wai.calculate_accumulation_distribution_score(
        symbol='EREGL',
        short_window=5,
        long_window=20
    )

    print("\nAccumulation/Distribution Phases (Last 10 days):")
    print(ad_scores[['date', 'phase', 'phase_strength',
                     'ad_score_short', 'ad_score_long']].tail(10))

    # Current phase
    current = ad_scores.iloc[-1]
    print(f"\nCurrent Phase: {current['phase']}")
    print(f"Phase Strength: {current['phase_strength']:.2f}")


def example_5_institutional_pressure():
    """Example 5: Institutional Pressure Analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Institutional Pressure Metrics")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('TUPRS', days=60)

    # Initialize WAI
    wai = WhaleActivityIndex(brokerage_data=brokerage_data)

    # Calculate institutional pressure
    pressure = wai.calculate_institutional_pressure(
        symbol='TUPRS',
        window=10
    )

    print("\nInstitutional Pressure (Last 10 days):")
    print(pressure[['date', 'pressure_level', 'institutional_pressure_score',
                   'buy_pressure_pct', 'sell_pressure_pct']].tail(10))

    # Current pressure
    current = pressure.iloc[-1]
    print(f"\nCurrent Pressure Level: {current['pressure_level']}")
    print(f"Pressure Score: {current['institutional_pressure_score']:.2f}")


def example_6_signal_generation():
    """Example 6: Generate Trading Signals"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Whale-Based Trading Signal Generation")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('SAHOL', days=90)

    # Initialize WAI
    wai = WhaleActivityIndex(
        brokerage_data=brokerage_data,
        top_n_brokers=10
    )

    # Generate signals
    signals = wai.generate_whale_signals(
        symbol='SAHOL',
        confidence_threshold=60.0
    )

    print("\nRecent Trading Signals:")
    print(signals[['date', 'signal', 'confidence_score',
                  'wai_score', 'phase', 'pressure_level']].tail(10))

    # Show only actionable signals (not HOLD)
    actionable = signals[signals['signal'] != 'HOLD']

    if len(actionable) > 0:
        print(f"\nActionable Signals: {len(actionable)}")
        print(actionable[['date', 'signal', 'confidence_score']].tail(5))


def example_7_convenience_functions():
    """Example 7: Using Convenience Functions"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Quick Analysis with Convenience Functions")
    print("="*70)

    # Generate sample data
    brokerage_data = generate_sample_brokerage_data('ASELS', days=60)

    # Quick WAI calculation
    wai_scores = calculate_wai(
        brokerage_data=brokerage_data,
        symbol='ASELS',
        top_n_brokers=10
    )

    print("\nQuick WAI Calculation (Last 5 days):")
    print(wai_scores[['date', 'wai_score', 'wai_directional']].tail(5))

    # Quick signal generation
    signals = generate_whale_signals(
        brokerage_data=brokerage_data,
        symbol='ASELS',
        confidence_threshold=65.0
    )

    latest_signal = signals.iloc[-1]
    print(f"\nLatest Signal: {latest_signal['signal']}")
    print(f"Confidence: {latest_signal['confidence_score']:.2f}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("WHALE ACTIVITY INDEX (WAI) - USAGE EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate various capabilities of the WAI module")
    print("for analyzing institutional trading activity on BIST stocks.")

    try:
        example_1_basic_wai_calculation()
        example_2_unusual_activity_detection()
        example_3_price_flow_discrepancy()
        example_4_accumulation_distribution()
        example_5_institutional_pressure()
        example_6_signal_generation()
        example_7_convenience_functions()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
