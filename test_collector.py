#!/usr/bin/env python3
"""
Test script for BIST Collector

This script demonstrates the usage of the BISTCollector class
and tests its various features.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.collectors.bist_collector import BISTCollector
import pandas as pd


def test_single_stock_daily():
    """Test collecting daily data for a single stock."""
    print("\n" + "="*70)
    print("TEST 1: Single Stock - Daily Data")
    print("="*70)
    
    collector = BISTCollector()
    
    try:
        df = collector.get_historical_data(
            symbol='THYAO',
            timeframe='daily',
            period='1mo'
        )
        
        print(f"\nSuccessfully fetched {len(df)} records for THYAO")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print("\nLast 5 records:")
        print(df.tail())
        print("\nData info:")
        print(df.info())
        
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_single_stock_intraday():
    """Test collecting intraday data for multiple timeframes."""
    print("\n" + "="*70)
    print("TEST 2: Single Stock - Intraday Data (Multiple Timeframes)")
    print("="*70)
    
    collector = BISTCollector()
    timeframes = ['1h', '30min', '15min']
    
    for tf in timeframes:
        try:
            print(f"\n--- Testing {tf} timeframe ---")
            df = collector.get_historical_data(
                symbol='GARAN',
                timeframe=tf,
                period='5d'
            )
            
            print(f"Fetched {len(df)} records")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print("\nLast 3 records:")
            print(df.tail(3))
            
        except Exception as e:
            print(f"ERROR for {tf}: {str(e)}")
            return False
    
    return True


def test_multiple_stocks():
    """Test collecting data for multiple stocks."""
    print("\n" + "="*70)
    print("TEST 3: Multiple Stocks - Daily Data")
    print("="*70)
    
    collector = BISTCollector()
    
    # BIST 30 sample stocks
    symbols = [
        'THYAO',   # Türk Hava Yolları
        'GARAN',   # Garanti Bankası
        'AKBNK',   # Akbank
        'ISCTR',   # İş Bankası (C)
        'EREGL',   # Ereğli Demir Çelik
        'TUPRS',   # Tüpraş
        'SAHOL',   # Sabancı Holding
        'KCHOL',   # Koç Holding
        'PETKM',   # Petkim
        'ASELS'    # Aselsan
    ]
    
    try:
        results = collector.get_multiple_stocks(
            symbols=symbols,
            timeframe='daily',
            period='1mo',
            continue_on_error=True
        )
        
        print(f"\nSuccessfully fetched data for {len(results)}/{len(symbols)} stocks")
        
        for symbol, df in results.items():
            symbol_clean = symbol.replace('.IS', '')
            print(f"\n{symbol_clean}:")
            print(f"  Records: {len(df)}")
            print(f"  Latest Close: {df['Close'].iloc[-1]:.2f}")
            print(f"  Volume: {df['Volume'].iloc[-1]:,.0f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_bist_index():
    """Test collecting BIST index data."""
    print("\n" + "="*70)
    print("TEST 4: BIST Indices")
    print("="*70)
    
    collector = BISTCollector()
    indices = ['XU100', 'XU030', 'XU050']
    
    for index in indices:
        try:
            print(f"\n--- Testing {index} ---")
            df = collector.get_bist_index(
                index_name=index,
                timeframe='daily',
                period='1mo'
            )
            
            print(f"Fetched {len(df)} records")
            print(f"Latest Close: {df['Close'].iloc[-1]:,.2f}")
            print(f"30-day Change: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
            
        except Exception as e:
            print(f"ERROR for {index}: {str(e)}")
            return False
    
    return True


def test_latest_price():
    """Test getting latest price information."""
    print("\n" + "="*70)
    print("TEST 5: Latest Price Information")
    print("="*70)
    
    collector = BISTCollector()
    
    try:
        symbols = ['THYAO', 'GARAN', 'AKBNK']
        
        for symbol in symbols:
            print(f"\n--- {symbol} ---")
            latest = collector.get_latest_price(symbol)
            
            for key, value in latest.items():
                if isinstance(value, float) and value is not None:
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_date_range():
    """Test with specific date range."""
    print("\n" + "="*70)
    print("TEST 6: Specific Date Range")
    print("="*70)
    
    collector = BISTCollector()
    
    try:
        df = collector.get_historical_data(
            symbol='EREGL',
            timeframe='daily',
            start_date='2024-01-01',
            end_date='2024-11-16'
        )
        
        print(f"\nFetched {len(df)} records for EREGL")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print("\nStatistics:")
        print(f"  High: {df['High'].max():,.2f}")
        print(f"  Low: {df['Low'].min():,.2f}")
        print(f"  Avg Volume: {df['Volume'].mean():,.0f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "="*70)
    print("TEST 7: Error Handling")
    print("="*70)
    
    collector = BISTCollector()
    
    # Test 1: Invalid symbol
    print("\n--- Test 7.1: Invalid Symbol ---")
    try:
        df = collector.get_historical_data(
            symbol='INVALIDSTOCK',
            timeframe='daily',
            period='1mo'
        )
        print("WARNING: Should have raised an error for invalid symbol")
        return False
    except Exception as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    # Test 2: Invalid timeframe
    print("\n--- Test 7.2: Invalid Timeframe ---")
    try:
        df = collector.get_historical_data(
            symbol='THYAO',
            timeframe='invalid_timeframe',
            period='1mo'
        )
        print("WARNING: Should have raised an error for invalid timeframe")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught error: {type(e).__name__}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("BIST COLLECTOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Single Stock Daily", test_single_stock_daily),
        ("Single Stock Intraday", test_single_stock_intraday),
        ("Multiple Stocks", test_multiple_stocks),
        ("BIST Indices", test_bist_index),
        ("Latest Price", test_latest_price),
        ("Date Range", test_date_range),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {test_name}: {str(e)}")
            results[test_name] = "ERROR"
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BIST Collector')
    parser.add_argument('--test', type=str, help='Run specific test (1-7)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test:
        test_num = int(args.test)
        tests = [
            test_single_stock_daily,
            test_single_stock_intraday,
            test_multiple_stocks,
            test_bist_index,
            test_latest_price,
            test_date_range,
            test_error_handling
        ]
        
        if 1 <= test_num <= len(tests):
            tests[test_num - 1]()
        else:
            print(f"Invalid test number. Choose 1-{len(tests)}")
    
    elif args.all:
        run_all_tests()
    
    else:
        print("Usage:")
        print("  python test_collector.py --all          # Run all tests")
        print("  python test_collector.py --test 1       # Run specific test")
        print("\nAvailable tests:")
        print("  1. Single Stock Daily")
        print("  2. Single Stock Intraday")
        print("  3. Multiple Stocks")
        print("  4. BIST Indices")
        print("  5. Latest Price")
        print("  6. Date Range")
        print("  7. Error Handling")
