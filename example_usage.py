#!/usr/bin/env python3
"""
Simple example demonstrating BIST Collector usage.

This script shows basic usage patterns for collecting BIST stock data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.collectors.bist_collector import BISTCollector


def main():
    print("BIST Data Collector - Simple Example\n")
    print("="*60)
    
    # Initialize the collector
    collector = BISTCollector(rate_limit_delay=0.5)
    
    # Example 1: Get daily data for THYAO (Turkish Airlines)
    print("\n1. Fetching daily data for THYAO (Turkish Airlines)...")
    try:
        df_daily = collector.get_historical_data(
            symbol='THYAO',
            timeframe='daily',
            period='1mo'
        )
        print(f"   ✓ Retrieved {len(df_daily)} daily records")
        print(f"   Latest Close: {df_daily['Close'].iloc[-1]:.2f} TL")
        print(f"   Volume: {df_daily['Volume'].iloc[-1]:,.0f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 2: Get hourly data for GARAN (Garanti Bank)
    print("\n2. Fetching hourly data for GARAN (Garanti Bank)...")
    try:
        df_hourly = collector.get_historical_data(
            symbol='GARAN',
            timeframe='1h',
            period='5d'
        )
        print(f"   ✓ Retrieved {len(df_hourly)} hourly records")
        print(f"   Latest Close: {df_hourly['Close'].iloc[-1]:.2f} TL")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 3: Get 30-minute data for AKBNK (Akbank)
    print("\n3. Fetching 30-min data for AKBNK (Akbank)...")
    try:
        df_30min = collector.get_historical_data(
            symbol='AKBNK',
            timeframe='30min',
            period='5d'
        )
        print(f"   ✓ Retrieved {len(df_30min)} 30-min records")
        print(f"   Latest Close: {df_30min['Close'].iloc[-1]:.2f} TL")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 4: Get multiple stocks
    print("\n4. Fetching data for multiple stocks...")
    symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'TUPRS']
    try:
        data_dict = collector.get_multiple_stocks(
            symbols=symbols,
            timeframe='daily',
            period='1mo',
            continue_on_error=True
        )
        print(f"   ✓ Retrieved data for {len(data_dict)} stocks:")
        for symbol, df in data_dict.items():
            clean_symbol = symbol.replace('.IS', '')
            latest_close = df['Close'].iloc[-1]
            print(f"     - {clean_symbol}: {latest_close:.2f} TL ({len(df)} records)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 5: Get BIST 100 index
    print("\n5. Fetching BIST 100 Index (XU100)...")
    try:
        xu100 = collector.get_bist_index(
            index_name='XU100',
            timeframe='daily',
            period='1mo'
        )
        print(f"   ✓ Retrieved {len(xu100)} records")
        print(f"   Latest Close: {xu100['Close'].iloc[-1]:,.2f}")
        
        # Calculate monthly return
        monthly_return = ((xu100['Close'].iloc[-1] / xu100['Close'].iloc[0]) - 1) * 100
        print(f"   Monthly Return: {monthly_return:+.2f}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("\nFor more examples, see:")
    print("  - test_collector.py (comprehensive tests)")
    print("  - README_COLLECTOR.md (documentation)")


if __name__ == "__main__":
    main()
