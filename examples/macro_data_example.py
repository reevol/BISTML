#!/usr/bin/env python3
"""
Example Usage of MacroCollector

This script demonstrates how to use the MacroCollector to fetch
Turkish and global macroeconomic indicators for the BIST trading system.
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.collectors.macro_collector import MacroCollector


def example_1_single_indicator():
    """Example 1: Fetch a single indicator"""
    print("\n" + "="*80)
    print("Example 1: Fetching S&P 500 Index (Last 30 Days)")
    print("="*80 + "\n")

    collector = MacroCollector()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    try:
        sp500 = collector.get_indicator('SP500', start_date, end_date)
        print(f"Successfully fetched {len(sp500)} records\n")
        print(sp500.head(10))
        print(f"\n{'.'*80}")
        print(f"Latest S&P 500: {sp500['value'].iloc[-1]:.2f}")
        print(f"30-day change: {((sp500['value'].iloc[-1] / sp500['value'].iloc[0]) - 1) * 100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")


def example_2_multiple_indicators():
    """Example 2: Fetch multiple global indicators"""
    print("\n" + "="*80)
    print("Example 2: Fetching Multiple Global Indicators (Last 90 Days)")
    print("="*80 + "\n")

    collector = MacroCollector()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    try:
        indicators = ['SP500', 'DAX', 'VIX']
        data = collector.get_multiple_indicators(indicators, start_date, end_date)

        print(f"Successfully fetched {len(data)} days of data for {len(data.columns)} indicators\n")
        print(data.tail(10))

        print(f"\n{'.'*80}")
        print("Latest Values:")
        for col in data.columns:
            latest_value = data[col].iloc[-1]
            if not pd.isna(latest_value):
                print(f"  {col}: {latest_value:.2f}")
    except Exception as e:
        print(f"Error: {e}")


def example_3_turkish_indicators():
    """Example 3: Fetch Turkish macroeconomic indicators"""
    print("\n" + "="*80)
    print("Example 3: Fetching Turkish Macroeconomic Indicators (Last 180 Days)")
    print("="*80 + "\n")

    collector = MacroCollector()

    # Note: Requires EVDS API key to be set
    if not os.getenv('EVDS_API_KEY'):
        print("⚠ EVDS_API_KEY not set. Skipping Turkish indicators example.")
        print("To fetch Turkish data, get an API key from: https://evds2.tcmb.gov.tr/")
        print("Then set: export EVDS_API_KEY='your_key'")
        return

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    try:
        turkish_data = collector.get_all_turkish_indicators(start_date, end_date)

        print(f"Successfully fetched {len(turkish_data)} days of Turkish macro data\n")
        print(turkish_data.tail(10))

        print(f"\n{'.'*80}")
        print("Latest Turkish Indicators:")
        for col in turkish_data.columns:
            latest_value = turkish_data[col].iloc[-1]
            if not pd.isna(latest_value):
                print(f"  {col}: {latest_value:.2f}")
    except Exception as e:
        print(f"Error: {e}")


def example_4_ml_features():
    """Example 4: Generate ML-ready macro features"""
    print("\n" + "="*80)
    print("Example 4: Generating ML-Ready Macro Features (Last 180 Days)")
    print("="*80 + "\n")

    collector = MacroCollector()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    try:
        features = collector.get_macro_features(
            start_date,
            end_date,
            include_derived=True
        )

        print(f"Generated {len(features.columns)} features from {len(features)} days of data\n")
        print("Feature columns (first 20):")
        for i, col in enumerate(list(features.columns)[:20], 1):
            print(f"  {i:2d}. {col}")

        print(f"\n{'.'*80}")
        print("Sample feature data (last 5 days):")
        print(features.tail(5).iloc[:, :10])  # Show first 10 columns

        # Check for missing values
        missing_pct = (features.isna().sum() / len(features) * 100).sort_values(ascending=False)
        if missing_pct.max() > 0:
            print(f"\n{'.'*80}")
            print("Missing data analysis (top 5):")
            for col, pct in missing_pct.head().items():
                print(f"  {col}: {pct:.1f}% missing")
    except Exception as e:
        print(f"Error: {e}")


def example_5_correlation_analysis():
    """Example 5: Correlation analysis between indicators"""
    print("\n" + "="*80)
    print("Example 5: Correlation Analysis Between Macro Indicators")
    print("="*80 + "\n")

    collector = MacroCollector()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    try:
        indicators = ['SP500', 'DAX', 'VIX']
        data = collector.get_multiple_indicators(indicators, start_date, end_date)

        # Calculate daily returns
        returns = data.pct_change().dropna()

        print(f"Analyzing correlations based on {len(returns)} days of data\n")
        print("Correlation Matrix (Daily Returns):")
        print(returns.corr().round(3))

        print(f"\n{'.'*80}")
        print("Key Insights:")

        # SP500 vs DAX
        sp_dax_corr = returns['SP500'].corr(returns['DAX'])
        print(f"  S&P 500 vs DAX correlation: {sp_dax_corr:.3f} (Market indices move together)")

        # VIX vs SP500
        vix_sp_corr = returns['VIX'].corr(returns['SP500'])
        print(f"  VIX vs S&P 500 correlation: {vix_sp_corr:.3f} (Negative = fear index)")

        # Volatility
        print(f"\nVolatility (Annualized Standard Deviation):")
        for col in returns.columns:
            vol = returns[col].std() * (252 ** 0.5) * 100
            print(f"  {col}: {vol:.2f}%")
    except Exception as e:
        print(f"Error: {e}")


def list_all_indicators():
    """List all available indicators"""
    print("\n" + "="*80)
    print("All Available Macroeconomic Indicators")
    print("="*80 + "\n")

    collector = MacroCollector()
    indicators_df = collector.list_available_indicators()

    print(indicators_df.to_string(index=False))


def main():
    """Run all examples"""
    # Import pandas here to avoid circular import
    global pd
    import pandas as pd

    print("\n" + "#"*80)
    print("# BIST AI Trading System - Macro Data Collector Examples")
    print("#"*80)

    # Check API keys
    print("\nAPI Key Status:")
    print(f"  FRED_API_KEY: {'✓ Set' if os.getenv('FRED_API_KEY') else '✗ Not Set'}")
    print(f"  EVDS_API_KEY: {'✓ Set' if os.getenv('EVDS_API_KEY') else '✗ Not Set'}")

    if not os.getenv('FRED_API_KEY'):
        print("\n⚠ Warning: FRED_API_KEY not set.")
        print("Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then set: export FRED_API_KEY='your_key'")

    # List all indicators
    list_all_indicators()

    # Run examples
    try:
        example_1_single_indicator()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_multiple_indicators()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_turkish_indicators()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_ml_features()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_correlation_analysis()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    # Setup instructions
    print("\n" + "#"*80)
    print("# Setup Instructions")
    print("#"*80)
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Get API keys:")
    print("   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("   - EVDS: https://evds2.tcmb.gov.tr/")
    print("\n3. Set environment variables:")
    print("   export FRED_API_KEY='your_fred_api_key'")
    print("   export EVDS_API_KEY='your_evds_api_key'")
    print("\n4. Run this example:")
    print("   python examples/macro_data_example.py")
    print("\n" + "#"*80 + "\n")


if __name__ == '__main__':
    main()
