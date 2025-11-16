# Redis Cache - Quick Start Guide

## Prerequisites

1. **Install Redis**:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

2. **Install Python dependencies**:
```bash
pip install redis pandas numpy
```

## Basic Usage

### Initialize Cache
```python
from src.data.storage import get_cache

# Get singleton cache instance
cache = get_cache()
```

### Cache OHLCV Data
```python
import pandas as pd

# Create or fetch OHLCV data
ohlcv_df = pd.DataFrame({...})

# Cache it (TTL: 5 minutes)
cache.cache_ohlcv("THYAO", "1h", ohlcv_df)

# Retrieve it
df = cache.get_ohlcv("THYAO", "1h")
```

### Cache Technical Indicators
```python
# Cache indicators (TTL: 5 minutes)
cache.cache_indicators("THYAO", "30min", indicators_df)

# Retrieve indicators
indicators = cache.get_indicators("THYAO", "30min")
```

### Cache Trading Signals
```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Cache signals (TTL: 1 minute)
cache.cache_signals(timestamp, signals_df)

# Retrieve signals
signals = cache.get_signals(timestamp)
```

### Use Cache Decorator
```python
@cache.cached(key_prefix="expensive_op", ttl=300)
def expensive_operation(symbol, timeframe):
    # Expensive calculation
    return result

# First call - executes function
result1 = expensive_operation("THYAO", "1h")

# Second call - returns cached result
result2 = expensive_operation("THYAO", "1h")
```

### Cache Invalidation
```python
# Delete specific key
cache.delete("ohlcv:THYAO:1h")

# Delete by pattern
cache.invalidate_pattern("ohlcv:THYAO:*")

# Delete all data for symbol
cache.invalidate_symbol("THYAO")

# Delete all data for timeframe
cache.invalidate_timeframe("30min")
```

### Check Cache Statistics
```python
stats = cache.get_stats()
print(f"Hit Rate: {stats['hit_rate']:.2%}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
```

## Default TTLs

| Data Type | TTL | Description |
|-----------|-----|-------------|
| OHLCV | 5 minutes | Price/volume data |
| Indicators | 5 minutes | Technical indicators |
| Signals | 1 minute | Trading signals |
| Whale Data | 1 hour | Brokerage distribution |
| Fundamentals | 24 hours | Financial statements |
| News | 30 minutes | News sentiment |

## Examples

Run the comprehensive examples:
```bash
python examples_cache_usage.py
```

## Documentation

- **Full Documentation**: `/home/user/BISTML/src/data/storage/CACHE_README.md`
- **Source Code**: `/home/user/BISTML/src/data/storage/cache.py`
- **Examples**: `/home/user/BISTML/examples_cache_usage.py`

## Key Features

- Multiple serialization methods (Pickle, JSON, MessagePack)
- TTL management with sensible defaults
- Cache statistics and monitoring
- Batch operations (mset/mget)
- Pattern-based invalidation
- DataFrame compression
- Cache decorators
- Connection pooling
- Error handling with fallbacks

## Common Patterns

### 1. Cache-Aside Pattern
```python
def get_data_with_cache(symbol, timeframe):
    # Try cache first
    data = cache.get_ohlcv(symbol, timeframe)
    if data is None:
        # Cache miss - fetch from source
        data = fetch_from_api(symbol, timeframe)
        cache.cache_ohlcv(symbol, timeframe, data)
    return data
```

### 2. Write-Through Pattern
```python
def update_data(symbol, timeframe, new_data):
    # Update database
    save_to_database(new_data)
    # Update cache
    cache.cache_ohlcv(symbol, timeframe, new_data)
```

### 3. Refresh-Ahead Pattern
```python
def refresh_cache_before_expiry():
    # Refresh popular data before TTL expires
    for symbol in popular_symbols:
        ttl = cache.ttl(f"ohlcv:{symbol}:1h")
        if ttl < 60:  # Less than 1 minute remaining
            data = fetch_from_api(symbol, "1h")
            cache.cache_ohlcv(symbol, "1h", data)
```

## Troubleshooting

**Connection Error**: Make sure Redis is running
```bash
redis-cli ping  # Should return PONG
```

**Low Hit Rate**: Increase TTLs or check cache patterns
```python
stats = cache.get_stats()
if stats['hit_rate'] < 0.5:
    # Consider increasing TTLs
    pass
```

**Memory Issues**: Configure Redis maxmemory policy
```bash
# In redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```
