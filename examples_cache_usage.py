"""
Example usage of the Redis-based caching system for BISTML trading system.

This file demonstrates various caching patterns and use cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.storage import TradingDataCache, get_cache, CacheInvalidationStrategy


def example_basic_caching():
    """Example 1: Basic cache operations."""
    print("\n=== Example 1: Basic Caching ===")

    # Initialize cache
    cache = TradingDataCache(
        host='localhost',
        port=6379,
        namespace='bistml',
        default_ttl=3600  # 1 hour default
    )

    # Set and get simple values
    cache.set("user_setting", {"theme": "dark", "language": "tr"}, ttl=300)
    settings = cache.get("user_setting")
    print(f"User settings: {settings}")

    # Check if key exists
    exists = cache.exists("user_setting")
    print(f"Key exists: {exists}")

    # Get TTL
    remaining_ttl = cache.ttl("user_setting")
    print(f"Remaining TTL: {remaining_ttl} seconds")

    # Delete key
    cache.delete("user_setting")
    print("Key deleted")


def example_dataframe_caching():
    """Example 2: Caching pandas DataFrames."""
    print("\n=== Example 2: DataFrame Caching ===")

    cache = get_cache()  # Use singleton

    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Cache OHLCV data
    symbol = "THYAO"
    timeframe = "1h"
    cache.cache_ohlcv(symbol, timeframe, ohlcv_data)
    print(f"Cached OHLCV data for {symbol} ({timeframe})")

    # Retrieve OHLCV data
    cached_df = cache.get_ohlcv(symbol, timeframe)
    print(f"Retrieved DataFrame shape: {cached_df.shape}")
    print(f"First few rows:\n{cached_df.head()}")


def example_technical_indicators_caching():
    """Example 3: Caching technical indicators."""
    print("\n=== Example 3: Technical Indicators Caching ===")

    cache = get_cache()

    # Create sample indicators data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
    indicators_data = pd.DataFrame({
        'timestamp': dates,
        'sma_20': np.random.uniform(100, 110, 100),
        'ema_50': np.random.uniform(100, 110, 100),
        'rsi_14': np.random.uniform(30, 70, 100),
        'macd': np.random.uniform(-2, 2, 100),
        'bb_upper': np.random.uniform(110, 120, 100),
        'bb_lower': np.random.uniform(90, 100, 100),
    })

    symbol = "GARAN"
    timeframe = "30min"
    cache.cache_indicators(symbol, timeframe, indicators_data)
    print(f"Cached indicators for {symbol} ({timeframe})")

    # Retrieve indicators
    cached_indicators = cache.get_indicators(symbol, timeframe)
    print(f"Retrieved indicators shape: {cached_indicators.shape}")


def example_signals_caching():
    """Example 4: Caching trading signals."""
    print("\n=== Example 4: Trading Signals Caching ===")

    cache = get_cache()

    # Create sample signals
    signals_data = pd.DataFrame({
        'symbol': ['THYAO', 'GARAN', 'EREGL', 'AKBNK', 'TUPRS'],
        'signal': ['BUY', 'STRONG_BUY', 'HOLD', 'SELL', 'BUY'],
        'target_price': [120.5, 85.3, 45.2, 32.1, 150.8],
        'confidence': [0.85, 0.92, 0.65, 0.78, 0.88],
        'whale_score': [0.7, 0.9, 0.5, 0.6, 0.8],
        'sentiment': [0.3, 0.5, 0.0, -0.2, 0.4]
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache.cache_signals(timestamp, signals_data, ttl=60)
    print(f"Cached signals for timestamp: {timestamp}")

    # Retrieve signals
    cached_signals = cache.get_signals(timestamp)
    print(f"Retrieved signals:\n{cached_signals}")


def example_whale_data_caching():
    """Example 5: Caching whale/brokerage data."""
    print("\n=== Example 5: Whale Data Caching ===")

    cache = get_cache()

    # Create sample whale data
    whale_data = pd.DataFrame({
        'broker_code': ['IS01', 'IS02', 'IS03', 'IS04', 'IS05'],
        'broker_name': ['Broker A', 'Broker B', 'Broker C', 'Broker D', 'Broker E'],
        'buy_volume': [50000, 30000, 25000, 20000, 15000],
        'sell_volume': [10000, 20000, 15000, 30000, 5000],
        'net_volume': [40000, 10000, 10000, -10000, 10000],
        'buy_value': [5000000, 3000000, 2500000, 2000000, 1500000],
        'sell_value': [1000000, 2000000, 1500000, 3000000, 500000]
    })

    symbol = "THYAO"
    cache.cache_whale_data(symbol, whale_data)
    print(f"Cached whale data for {symbol}")

    # Retrieve whale data
    cached_whale = cache.get_whale_data(symbol)
    print(f"Retrieved whale data:\n{cached_whale}")


def example_batch_operations():
    """Example 6: Batch cache operations."""
    print("\n=== Example 6: Batch Operations ===")

    cache = get_cache()

    # Set multiple values at once
    batch_data = {
        "config:max_positions": 10,
        "config:risk_per_trade": 0.02,
        "config:stop_loss": 0.05,
        "config:take_profit": 0.10,
    }
    cache.mset(batch_data, ttl=3600)
    print("Cached batch configuration")

    # Get multiple values at once
    keys = list(batch_data.keys())
    retrieved = cache.mget(keys)
    print(f"Retrieved batch data: {retrieved}")


def example_cache_decorator():
    """Example 7: Using cache decorator."""
    print("\n=== Example 7: Cache Decorator ===")

    cache = get_cache()

    # Expensive function with caching
    @cache.cached(key_prefix="expensive_calc", ttl=300)
    def expensive_calculation(symbol: str, days: int):
        """Simulate an expensive calculation."""
        print(f"  Performing expensive calculation for {symbol} ({days} days)...")
        import time
        time.sleep(0.5)  # Simulate work
        return {"symbol": symbol, "result": np.random.random(), "days": days}

    # First call - will execute function
    print("First call:")
    result1 = expensive_calculation("THYAO", 30)
    print(f"  Result: {result1}")

    # Second call - will use cache
    print("Second call (cached):")
    result2 = expensive_calculation("THYAO", 30)
    print(f"  Result: {result2}")

    # Different parameters - will execute function
    print("Third call (different params):")
    result3 = expensive_calculation("GARAN", 30)
    print(f"  Result: {result3}")


def example_pattern_invalidation():
    """Example 8: Pattern-based cache invalidation."""
    print("\n=== Example 8: Pattern-based Invalidation ===")

    cache = get_cache()

    # Cache some data
    cache.set("ohlcv:THYAO:1h", "data1", ttl=3600)
    cache.set("ohlcv:THYAO:30min", "data2", ttl=3600)
    cache.set("ohlcv:GARAN:1h", "data3", ttl=3600)
    cache.set("indicators:THYAO:1h", "data4", ttl=3600)
    print("Cached multiple keys")

    # Invalidate all OHLCV data for THYAO
    count = cache.invalidate_pattern("ohlcv:THYAO:*")
    print(f"Invalidated {count} keys matching pattern 'ohlcv:THYAO:*'")

    # Invalidate all data for a symbol
    count = cache.invalidate_symbol("GARAN")
    print(f"Invalidated {count} keys for symbol GARAN")


def example_cache_statistics():
    """Example 9: Cache statistics."""
    print("\n=== Example 9: Cache Statistics ===")

    cache = get_cache()

    # Reset stats
    cache.reset_stats()

    # Perform some operations
    cache.set("test1", "value1")
    cache.set("test2", "value2")
    cache.get("test1")  # Hit
    cache.get("test1")  # Hit
    cache.get("test3")  # Miss
    cache.delete("test1")

    # Get statistics
    stats = cache.get_stats()
    print(f"Cache statistics:")
    print(f"  Hits: {stats.get('hits', 0)}")
    print(f"  Misses: {stats.get('misses', 0)}")
    print(f"  Sets: {stats.get('sets', 0)}")
    print(f"  Deletes: {stats.get('deletes', 0)}")
    print(f"  Hit Rate: {stats.get('hit_rate', 0):.2%}")


def example_ttl_management():
    """Example 10: TTL management."""
    print("\n=== Example 10: TTL Management ===")

    cache = get_cache()

    # Set with different TTLs
    cache.set("short_lived", "data1", ttl=60)  # 1 minute
    cache.set("medium_lived", "data2", ttl=3600)  # 1 hour
    cache.set("long_lived", "data3", ttl=86400)  # 24 hours
    cache.set("permanent", "data4", ttl=0)  # No expiration

    # Check TTLs
    print(f"short_lived TTL: {cache.ttl('short_lived')} seconds")
    print(f"medium_lived TTL: {cache.ttl('medium_lived')} seconds")
    print(f"permanent TTL: {cache.ttl('permanent')} seconds (-1 means no expiry)")

    # Update TTL
    cache.expire("short_lived", 300)  # Extend to 5 minutes
    print(f"Updated short_lived TTL: {cache.ttl('short_lived')} seconds")


def example_real_world_workflow():
    """Example 11: Real-world trading workflow."""
    print("\n=== Example 11: Real-world Trading Workflow ===")

    cache = get_cache()

    symbols = ["THYAO", "GARAN", "EREGL"]
    timeframe = "30min"

    # Step 1: Check cache for OHLCV data
    print("Step 1: Fetching OHLCV data...")
    for symbol in symbols:
        ohlcv = cache.get_ohlcv(symbol, timeframe)
        if ohlcv is None:
            print(f"  {symbol}: Cache MISS - would fetch from API")
            # Simulate fetching from API
            dates = pd.date_range(start='2024-01-01', periods=50, freq='30min')
            ohlcv = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(100, 110, 50),
                'high': np.random.uniform(110, 120, 50),
                'low': np.random.uniform(90, 100, 50),
                'close': np.random.uniform(100, 110, 50),
                'volume': np.random.randint(1000, 10000, 50)
            })
            cache.cache_ohlcv(symbol, timeframe, ohlcv, ttl=300)
        else:
            print(f"  {symbol}: Cache HIT - using cached data")

    # Step 2: Check cache for indicators
    print("\nStep 2: Fetching technical indicators...")
    for symbol in symbols:
        indicators = cache.get_indicators(symbol, timeframe)
        if indicators is None:
            print(f"  {symbol}: Cache MISS - would calculate indicators")
            # Simulate calculating indicators
            dates = pd.date_range(start='2024-01-01', periods=50, freq='30min')
            indicators = pd.DataFrame({
                'timestamp': dates,
                'rsi': np.random.uniform(30, 70, 50),
                'macd': np.random.uniform(-2, 2, 50),
            })
            cache.cache_indicators(symbol, timeframe, indicators, ttl=300)
        else:
            print(f"  {symbol}: Cache HIT - using cached indicators")

    # Step 3: Generate and cache signals
    print("\nStep 3: Generating signals...")
    signals = pd.DataFrame({
        'symbol': symbols,
        'signal': ['BUY', 'HOLD', 'SELL'],
        'confidence': [0.85, 0.60, 0.75]
    })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache.cache_signals(timestamp, signals, ttl=60)
    print(f"  Cached signals for timestamp: {timestamp}")

    # Step 4: Get cache statistics
    print("\nStep 4: Cache performance:")
    stats = cache.get_stats()
    print(f"  Hit Rate: {stats.get('hit_rate', 0):.2%}")
    print(f"  Total Hits: {stats.get('hits', 0)}")
    print(f"  Total Misses: {stats.get('misses', 0)}")


if __name__ == "__main__":
    """Run all examples."""
    print("=" * 80)
    print("BISTML Trading System - Redis Cache Usage Examples")
    print("=" * 80)

    try:
        # Run examples
        example_basic_caching()
        example_dataframe_caching()
        example_technical_indicators_caching()
        example_signals_caching()
        example_whale_data_caching()
        example_batch_operations()
        example_cache_decorator()
        example_pattern_invalidation()
        example_cache_statistics()
        example_ttl_management()
        example_real_world_workflow()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure Redis is running on localhost:6379")
        import traceback
        traceback.print_exc()
