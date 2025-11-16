# Configuration System Quick Reference

## Overview

A comprehensive configuration management system for BIST AI Trading System with:
- YAML file loading
- Environment-specific configs (dev, staging, prod)
- Environment variable substitution
- Configuration validation
- Singleton pattern support

## Quick Start

### 1. Basic Usage

```python
from src.utils.config import Config

# Load config
config = Config.load('configs/data_sources.yaml')

# Get values
api_key = config.get('macro_data.fred.api_key')
rate_limit = config.get('macro_data.fred.rate_limit', default=100)
```

### 2. Environment-Specific Config

```python
# Load with environment (dev, staging, prod)
dev_config = Config.load('configs/data_sources.yaml', env='development')
prod_config = Config.load('configs/data_sources.yaml', env='production')

# Or use ENVIRONMENT env var
import os
os.environ['ENVIRONMENT'] = 'production'
config = Config.load('configs/data_sources.yaml')
```

### 3. Global Configuration

```python
from src.utils.config import init_config, get_config

# Initialize once
init_config('configs/data_sources.yaml', env='production')

# Use anywhere in your app
config = get_config()
value = config.get('some.config.path')
```

### 4. Configuration Validation

```python
# Validate required fields
config.validate_required_fields([
    'macro_data.fred.api_key',
    'collection.cache.enabled'
])

# Validate types
config.validate_types({
    'macro_data.fred.rate_limit': int,
    'collection.cache.enabled': bool
})
```

## Environment Variable Substitution

In your YAML files:
```yaml
api:
  key: "${API_KEY}"  # Required
  secret: "${API_SECRET:default_value}"  # With default
  url: "https://${HOST:localhost}:${PORT:8080}/api"  # In strings
```

## File Structure

Created files:
```
/home/user/BISTML/
├── src/utils/config.py                          # Main config module
├── configs/
│   ├── data_sources.yaml                        # Base config
│   ├── data_sources.development.yaml            # Dev overrides
│   ├── data_sources.staging.yaml                # Staging overrides
│   └── data_sources.production.yaml             # Prod overrides
├── examples/config_usage_example.py             # Usage examples
├── tests/test_config.py                         # Unit tests
└── docs/CONFIG_USAGE.md                         # Full documentation
```

## API Reference

### Loading
- `Config.load(path, env=None, validate=True)` - Load from file
- `Config.load_multiple(paths, env=None)` - Load and merge multiple files
- `Config.from_dict(dict, env='development')` - Create from dictionary

### Accessing
- `config.get(path, default=None)` - Get value with optional default
- `config.get_required(path)` - Get required value (raises if missing)
- `config.has(path)` - Check if path exists
- `config.set(path, value)` - Set value

### Environment
- `config.env` - Current environment name
- `config.is_production` - True if production
- `config.is_development` - True if development
- `config.is_staging` - True if staging

### Global Config
- `init_config(paths, env=None)` - Initialize global config
- `get_config()` - Get global config instance
- `set_config(config)` - Set global config instance
- `clear_cache()` - Clear config cache

## Examples

Run comprehensive examples:
```bash
python examples/config_usage_example.py
```

Run tests:
```bash
pytest tests/test_config.py -v
```

## Best Practices

1. **Never commit secrets** - Use environment variables
2. **Validate on startup** - Check required fields early
3. **Use environment-specific configs** - Separate dev/staging/prod
4. **Document your configs** - Add comments to YAML files
5. **Test your configs** - Validate in CI/CD pipeline

## Common Patterns

### Application Initialization
```python
from src.utils.config import init_config

# At application startup
config = init_config([
    'configs/data_sources.yaml',
    'configs/scheduler_config.yaml'
], env='production')

# Validate critical settings
config.validate_required_fields([
    'database.host',
    'api.key'
])
```

### Module-Level Config
```python
from src.utils.config import get_config

def fetch_data():
    config = get_config()
    api_key = config.get_required('macro_data.fred.api_key')
    rate_limit = config.get('macro_data.fred.rate_limit', default=100)
    # ... use config
```

### Testing with Custom Config
```python
import pytest
from src.utils.config import Config, clear_cache

@pytest.fixture
def test_config():
    clear_cache()
    config = Config.from_dict({
        'api': {'timeout': 5},
        'database': {'host': 'testdb'}
    })
    return config

def test_something(test_config):
    timeout = test_config.get('api.timeout')
    assert timeout == 5
```

## Full Documentation

See `/home/user/BISTML/docs/CONFIG_USAGE.md` for complete documentation.
