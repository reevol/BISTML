# Configuration Management System

Comprehensive configuration management for the BIST AI Trading System with support for environment-specific configs, validation, and YAML file loading.

## Features

- **YAML Configuration Files**: Load and parse YAML configuration files
- **Environment-Specific Configs**: Support for development, staging, and production environments
- **Environment Variable Substitution**: Automatic substitution of `${VAR_NAME}` placeholders
- **Configuration Validation**: Built-in validators for required fields, types, ranges, and choices
- **Configuration Merging**: Deep merge multiple configuration files
- **Singleton Pattern**: Global configuration access throughout your application
- **Type-Safe Access**: Dot notation for accessing nested configuration values
- **Caching**: Automatic caching of loaded configurations for performance
- **Extensible**: Easy to add custom validators and configuration schemas

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Environment-Specific Configurations](#environment-specific-configurations)
- [Environment Variables](#environment-variables)
- [Configuration Validation](#configuration-validation)
- [Multiple Configuration Files](#multiple-configuration-files)
- [Global Configuration](#global-configuration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Quick Start

### 1. Install Dependencies

```bash
pip install pyyaml python-dotenv
```

### 2. Create Configuration File

Create `configs/myapp.yaml`:

```yaml
# Application Configuration
app:
  name: "My App"
  version: "1.0.0"
  debug: false

database:
  host: "${DB_HOST:localhost}"
  port: 5432
  name: "${DB_NAME}"

api:
  timeout: 30
  rate_limit: 100
```

### 3. Load and Use Configuration

```python
from src.utils.config import Config

# Load configuration
config = Config.load('configs/myapp.yaml')

# Access values
app_name = config.get('app.name')
db_host = config.get('database.host')
timeout = config.get('api.timeout', default=30)
```

## Basic Usage

### Loading Configuration

```python
from src.utils.config import Config

# Load from file
config = Config.load('configs/data_sources.yaml')

# Access nested values using dot notation
api_key = config.get('macro_data.fred.api_key')
rate_limit = config.get('macro_data.fred.rate_limit')

# Get with default value
cache_ttl = config.get('collection.cache.ttl', default=3600)

# Get required value (raises error if not found)
api_key = config.get_required('macro_data.fred.api_key')

# Check if value exists
if config.has('collection.cache.enabled'):
    cache_enabled = config.get('collection.cache.enabled')
```

### Setting Values

```python
# Set configuration values
config.set('api.timeout', 60)
config.set('custom.nested.value', 'my_value')

# Save modified configuration
config.save('configs/modified_config.yaml')
```

### Converting to Dictionary

```python
# Get entire configuration as dictionary
config_dict = config.to_dict()

# Create config from dictionary
config = Config.from_dict(config_dict, env='development')
```

## Environment-Specific Configurations

The configuration system supports environment-specific overrides:

### File Naming Convention

- Base config: `config.yaml`
- Development: `config.development.yaml`
- Staging: `config.staging.yaml`
- Production: `config.production.yaml`

### Creating Environment Configs

**Base config** (`configs/data_sources.yaml`):
```yaml
macro_data:
  fred:
    rate_limit: 120

logging:
  level: "INFO"
```

**Development override** (`configs/data_sources.development.yaml`):
```yaml
macro_data:
  fred:
    rate_limit: 50  # Lower limit for dev

logging:
  level: "DEBUG"  # Verbose logging
```

**Production override** (`configs/data_sources.production.yaml`):
```yaml
macro_data:
  fred:
    rate_limit: 100  # Conservative limit

logging:
  level: "WARNING"  # Less verbose
```

### Loading Environment-Specific Config

```python
# Method 1: Specify environment explicitly
dev_config = Config.load('configs/data_sources.yaml', env='development')
prod_config = Config.load('configs/data_sources.yaml', env='production')

# Method 2: Use ENVIRONMENT environment variable
import os
os.environ['ENVIRONMENT'] = 'production'
config = Config.load('configs/data_sources.yaml')  # Loads production config
```

### Checking Environment

```python
config = Config.load('configs/data_sources.yaml', env='production')

# Check current environment
print(config.env)  # 'production'

# Environment checks
if config.is_production:
    # Production-specific code
    pass

if config.is_development:
    # Development-specific code
    pass

if config.is_staging:
    # Staging-specific code
    pass
```

## Environment Variables

### Syntax

The configuration system supports environment variable substitution:

```yaml
# Basic substitution
api_key: "${API_KEY}"

# With default value
api_key: "${API_KEY:default_key}"

# In complex values
database:
  url: "postgresql://${DB_USER:admin}:${DB_PASSWORD}@${DB_HOST:localhost}/${DB_NAME}"
```

### Example

**config.yaml**:
```yaml
database:
  host: "${DB_HOST:localhost}"
  port: "${DB_PORT:5432}"
  name: "${DB_NAME}"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
```

**.env**:
```bash
DB_HOST=prod-db.example.com
DB_PORT=5432
DB_NAME=bistml_prod
DB_USER=admin
DB_PASSWORD=secure_password
```

**Usage**:
```python
from dotenv import load_dotenv
from src.utils.config import Config

# Load environment variables
load_dotenv()

# Load configuration (environment variables will be substituted)
config = Config.load('configs/database.yaml')

# Access substituted values
db_host = config.get('database.host')  # 'prod-db.example.com'
db_port = config.get('database.port')  # '5432'
```

## Configuration Validation

### Built-in Validators

#### Required Fields

```python
config = Config.load('configs/data_sources.yaml')

# Validate required fields
config.validate_required_fields([
    'macro_data.fred.api_key',
    'macro_data.evds.api_key',
    'collection.cache.enabled'
])
```

#### Type Validation

```python
# Validate types
config.validate_types({
    'macro_data.fred.rate_limit': int,
    'collection.cache.enabled': bool,
    'logging.level': str
})
```

#### Range Validation

```python
from src.utils.config import ConfigValidator

validator = ConfigValidator()

# Validate numeric range
validator.validate_range(
    config._config,
    'macro_data.fred.rate_limit',
    min_val=1,
    max_val=10000
)
```

#### Choice Validation

```python
# Validate value is in allowed choices
validator.validate_choices(
    config._config,
    'logging.level',
    ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
)
```

### Pre-configured Validators

```python
from src.utils.config import (
    Config,
    DataSourceConfigValidator,
    SchedulerConfigValidator
)

# Load and validate data source config
config = Config.load('configs/data_sources.yaml')
DataSourceConfigValidator.validate(config)

# Load and validate scheduler config
scheduler_config = Config.load('configs/scheduler_config.yaml')
SchedulerConfigValidator.validate(scheduler_config)
```

### Custom Validators

```python
from src.utils.config import Config, ConfigValidationError

class MyCustomValidator:
    @staticmethod
    def validate(config: Config) -> None:
        # Custom validation logic
        if config.get('api.timeout') > 300:
            raise ConfigValidationError("API timeout too high")

        if not config.has('api.endpoint'):
            raise ConfigValidationError("API endpoint required")

# Use custom validator
config = Config.load('configs/api.yaml')
MyCustomValidator.validate(config)
```

## Multiple Configuration Files

### Loading Multiple Configs

```python
from src.utils.config import Config

# Load and merge multiple configuration files
config = Config.load_multiple([
    'configs/data_sources.yaml',
    'configs/scheduler_config.yaml',
    'configs/features.yaml'
], env='production')

# Access values from any of the configs
rate_limit = config.get('macro_data.fred.rate_limit')
scheduler_tz = config.get('scheduler.timezone')
```

### Merge Behavior

Later configurations override earlier ones:

```python
# config1.yaml
api:
  timeout: 30
  retries: 3

# config2.yaml
api:
  timeout: 60  # Overrides config1

# Merged result
api:
  timeout: 60  # From config2
  retries: 3   # From config1
```

## Global Configuration

### Initialize Global Config

```python
from src.utils.config import init_config, get_config

# Initialize global configuration
init_config('configs/data_sources.yaml', env='production')

# Access from anywhere in your application
config = get_config()
api_key = config.get('macro_data.fred.api_key')
```

### Set Custom Global Config

```python
from src.utils.config import Config, set_config, get_config

# Load custom config
custom_config = Config.load('configs/custom.yaml')

# Set as global
set_config(custom_config)

# Access global config
config = get_config()
```

## API Reference

### Config Class

#### Methods

- `Config.load(config_path, env=None, validate=True)` - Load configuration from file
- `Config.load_multiple(config_paths, env=None)` - Load and merge multiple configs
- `Config.from_dict(config_dict, env='development')` - Create from dictionary
- `config.get(path, default=None)` - Get configuration value
- `config.get_required(path)` - Get required value (raises error if missing)
- `config.set(path, value)` - Set configuration value
- `config.has(path)` - Check if path exists
- `config.to_dict()` - Convert to dictionary
- `config.save(output_path)` - Save to YAML file
- `config.validate_required_fields(required_fields)` - Validate required fields
- `config.validate_types(type_specs)` - Validate types

#### Properties

- `config.env` - Current environment name
- `config.is_production` - True if production environment
- `config.is_development` - True if development environment
- `config.is_staging` - True if staging environment

### Global Functions

- `init_config(config_paths, env=None)` - Initialize global configuration
- `get_config(reload=False)` - Get global configuration instance
- `set_config(config)` - Set global configuration instance
- `clear_cache()` - Clear configuration cache

### Validators

- `ConfigValidator.validate_required_fields(config, required_fields)`
- `ConfigValidator.validate_types(config, type_specs)`
- `ConfigValidator.validate_range(config, field_path, min_val, max_val)`
- `ConfigValidator.validate_choices(config, field_path, choices)`

### Exceptions

- `ConfigError` - Base exception for configuration errors
- `ConfigValidationError` - Configuration validation failed
- `ConfigNotFoundError` - Configuration file not found

## Best Practices

### 1. Use Environment Variables for Secrets

**Never** commit secrets to configuration files. Use environment variables:

```yaml
# Good
api:
  key: "${API_KEY}"
  secret: "${API_SECRET}"

# Bad
api:
  key: "hardcoded_secret_key"
  secret: "hardcoded_secret"
```

### 2. Provide Sensible Defaults

```yaml
# Provide defaults for non-critical values
cache:
  ttl: "${CACHE_TTL:3600}"
  enabled: "${CACHE_ENABLED:true}"
```

### 3. Use Environment-Specific Configs

Keep environment-specific settings in separate files:

```
configs/
  ├── app.yaml              # Base config
  ├── app.development.yaml  # Dev overrides
  ├── app.staging.yaml      # Staging overrides
  └── app.production.yaml   # Production overrides
```

### 4. Validate Critical Configurations

```python
# Validate critical configurations on startup
config = Config.load('configs/app.yaml', env='production')
config.validate_required_fields([
    'database.host',
    'api.key',
    'redis.url'
])
```

### 5. Use Type Hints

```python
from typing import Optional

def get_api_timeout(config: Config) -> int:
    return config.get('api.timeout', default=30)

def get_api_key(config: Config) -> str:
    return config.get_required('api.key')
```

### 6. Document Your Configuration

Add comments to YAML files:

```yaml
# API Configuration
api:
  # Maximum request timeout in seconds
  timeout: 30

  # Maximum requests per minute
  rate_limit: 100

  # Retry configuration
  retry:
    max_attempts: 3  # Maximum number of retry attempts
    backoff_factor: 2  # Exponential backoff multiplier
```

### 7. Use Global Config Sparingly

Only use global config for truly application-wide settings. For component-specific configs, pass Config instances explicitly:

```python
# Good - explicit dependency
def my_function(config: Config):
    timeout = config.get('api.timeout')
    # ...

# Less good - hidden dependency
def my_function():
    config = get_config()
    timeout = config.get('api.timeout')
    # ...
```

### 8. Clear Cache During Testing

```python
import pytest
from src.utils.config import clear_cache

@pytest.fixture(autouse=True)
def clear_config_cache():
    clear_cache()
    yield
    clear_cache()
```

## Examples

See `/home/user/BISTML/examples/config_usage_example.py` for comprehensive examples of all features.

Run examples:
```bash
python examples/config_usage_example.py
```

## Related Documentation

- [Data Sources Configuration](../configs/data_sources.yaml)
- [Scheduler Configuration](../configs/scheduler_config.yaml)
- [Environment Variables](.env.example)
