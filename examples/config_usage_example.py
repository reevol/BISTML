#!/usr/bin/env python3
"""
Configuration System Usage Examples

This script demonstrates how to use the configuration management system
in the BIST AI Trading System.

Run with:
    python examples/config_usage_example.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import (
    Config,
    get_config,
    set_config,
    init_config,
    clear_cache,
    ConfigError,
    ConfigValidationError,
    DataSourceConfigValidator,
    SchedulerConfigValidator
)


def example_1_basic_usage():
    """Example 1: Basic configuration loading"""
    print("\n" + "=" * 70)
    print("Example 1: Basic Configuration Loading")
    print("=" * 70)

    # Load configuration from file
    config = Config.load('configs/data_sources.yaml')

    # Access configuration values using dot notation
    fred_api_key = config.get('macro_data.fred.api_key')
    print(f"FRED API Key: {fred_api_key}")

    # Get with default value
    rate_limit = config.get('macro_data.fred.rate_limit', default=100)
    print(f"FRED Rate Limit: {rate_limit}")

    # Get nested values
    cache_enabled = config.get('collection.cache.enabled')
    print(f"Cache Enabled: {cache_enabled}")

    # Check if value exists
    has_redis = config.has('collection.cache.redis_url')
    print(f"Has Redis URL: {has_redis}")


def example_2_environment_specific():
    """Example 2: Environment-specific configurations"""
    print("\n" + "=" * 70)
    print("Example 2: Environment-Specific Configurations")
    print("=" * 70)

    # Load development configuration
    dev_config = Config.load('configs/data_sources.yaml', env='development')
    print(f"\nDevelopment Environment:")
    print(f"  Log Level: {dev_config.get('logging.level')}")
    print(f"  Cache TTL: {dev_config.get('collection.cache.ttl')}")
    print(f"  Rate Limit: {dev_config.get('macro_data.fred.rate_limit')}")

    # Load production configuration
    prod_config = Config.load('configs/data_sources.yaml', env='production')
    print(f"\nProduction Environment:")
    print(f"  Log Level: {prod_config.get('logging.level')}")
    print(f"  Cache TTL: {prod_config.get('collection.cache.ttl')}")
    print(f"  Rate Limit: {prod_config.get('macro_data.fred.rate_limit')}")

    # Load staging configuration
    staging_config = Config.load('configs/data_sources.yaml', env='staging')
    print(f"\nStaging Environment:")
    print(f"  Log Level: {staging_config.get('logging.level')}")
    print(f"  Cache TTL: {staging_config.get('collection.cache.ttl')}")
    print(f"  Rate Limit: {staging_config.get('macro_data.fred.rate_limit')}")


def example_3_environment_variables():
    """Example 3: Environment variable substitution"""
    print("\n" + "=" * 70)
    print("Example 3: Environment Variable Substitution")
    print("=" * 70)

    # Set some environment variables for demonstration
    os.environ['FRED_API_KEY'] = 'demo_fred_key_12345'
    os.environ['EVDS_API_KEY'] = 'demo_evds_key_67890'
    os.environ['REDIS_URL'] = 'redis://custom-redis:6379/5'

    # Load configuration (environment variables will be substituted)
    config = Config.load('configs/data_sources.yaml')

    # Environment variables are automatically substituted
    print(f"FRED API Key (from env): {config.get('macro_data.fred.api_key')}")
    print(f"EVDS API Key (from env): {config.get('macro_data.evds.api_key')}")

    # Production config uses REDIS_URL with default
    prod_config = Config.load('configs/data_sources.yaml', env='production')
    print(f"Redis URL (from env): {prod_config.get('collection.cache.redis_url')}")


def example_4_multiple_configs():
    """Example 4: Loading and merging multiple configurations"""
    print("\n" + "=" * 70)
    print("Example 4: Loading Multiple Configurations")
    print("=" * 70)

    # Load and merge multiple config files
    config = Config.load_multiple([
        'configs/data_sources.yaml',
        'configs/scheduler_config.yaml'
    ], env='development')

    # Access values from both configs
    print(f"Data Source - Rate Limit: {config.get('macro_data.fred.rate_limit')}")
    print(f"Scheduler - Run on Start: {config.get('scheduler.run_on_start')}")
    print(f"Scheduler - Timezone: {config.get('scheduler.timezone')}")
    print(f"Market Hours - Open: {config.get('market_hours.regular.open')}")


def example_5_global_config():
    """Example 5: Using global singleton configuration"""
    print("\n" + "=" * 70)
    print("Example 5: Global Singleton Configuration")
    print("=" * 70)

    # Initialize global configuration
    init_config('configs/data_sources.yaml', env='development')

    # Access global config from anywhere
    config = get_config()
    print(f"Environment: {config.env}")
    print(f"Is Development: {config.is_development}")
    print(f"Is Production: {config.is_production}")

    # You can also set a config instance as global
    custom_config = Config.load('configs/scheduler_config.yaml', env='production')
    set_config(custom_config)

    # Now get_config() returns the custom config
    config = get_config()
    print(f"New Global Config Environment: {config.env}")
    print(f"Is Production: {config.is_production}")


def example_6_validation():
    """Example 6: Configuration validation"""
    print("\n" + "=" * 70)
    print("Example 6: Configuration Validation")
    print("=" * 70)

    # Load configuration
    config = Config.load('configs/data_sources.yaml', env='development')

    # Validate required fields
    try:
        config.validate_required_fields([
            'macro_data.fred.api_key',
            'macro_data.evds.api_key',
            'collection.cache.enabled'
        ])
        print("Required fields validation: PASSED")
    except ConfigValidationError as e:
        print(f"Validation failed: {e}")

    # Validate types
    try:
        config.validate_types({
            'macro_data.fred.rate_limit': int,
            'collection.cache.enabled': bool,
            'logging.level': str
        })
        print("Type validation: PASSED")
    except ConfigValidationError as e:
        print(f"Type validation failed: {e}")

    # Use pre-configured validators
    try:
        DataSourceConfigValidator.validate(config)
        print("DataSource validation: PASSED")
    except ConfigValidationError as e:
        print(f"DataSource validation failed: {e}")


def example_7_modify_and_save():
    """Example 7: Modifying and saving configuration"""
    print("\n" + "=" * 70)
    print("Example 7: Modifying and Saving Configuration")
    print("=" * 70)

    # Load configuration
    config = Config.load('configs/data_sources.yaml', env='development')

    # Modify values
    config.set('macro_data.fred.rate_limit', 200)
    config.set('custom.my_setting', 'custom_value')
    config.set('custom.nested.value', 42)

    print(f"Modified Rate Limit: {config.get('macro_data.fred.rate_limit')}")
    print(f"Custom Setting: {config.get('custom.my_setting')}")
    print(f"Nested Value: {config.get('custom.nested.value')}")

    # Save to new file
    output_path = 'configs/data_sources.custom.yaml'
    config.save(output_path)
    print(f"\nConfiguration saved to: {output_path}")

    # Clean up
    if Path(output_path).exists():
        Path(output_path).unlink()
        print("Cleanup: Custom config file removed")


def example_8_error_handling():
    """Example 8: Error handling"""
    print("\n" + "=" * 70)
    print("Example 8: Error Handling")
    print("=" * 70)

    # Try to load non-existent file
    try:
        config = Config.load('configs/nonexistent.yaml')
    except ConfigError as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")

    # Try to get required value that doesn't exist
    try:
        config = Config.load('configs/data_sources.yaml')
        value = config.get_required('nonexistent.path.to.value')
    except ConfigError as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")

    # Try to access global config before initialization
    try:
        clear_cache()  # Clear any previous initialization
        # Note: We need to reset the global config for this example
        import src.utils.config as config_module
        config_module._global_config = None
        config = get_config()
    except ConfigError as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")


def example_9_from_dict():
    """Example 9: Creating configuration from dictionary"""
    print("\n" + "=" * 70)
    print("Example 9: Creating Configuration from Dictionary")
    print("=" * 70)

    # Create config from dictionary
    config_dict = {
        'api': {
            'endpoint': 'https://api.example.com',
            'timeout': 30,
            'retry': {
                'max_attempts': 3,
                'backoff': 2
            }
        },
        'features': {
            'caching': True,
            'logging': True
        }
    }

    config = Config.from_dict(config_dict, env='development')

    print(f"API Endpoint: {config.get('api.endpoint')}")
    print(f"API Timeout: {config.get('api.timeout')}")
    print(f"Retry Max Attempts: {config.get('api.retry.max_attempts')}")
    print(f"Features Caching: {config.get('features.caching')}")


def example_10_cache_management():
    """Example 10: Cache management"""
    print("\n" + "=" * 70)
    print("Example 10: Cache Management")
    print("=" * 70)

    # Load config (will be cached)
    config1 = Config.load('configs/data_sources.yaml', env='development')
    print("First load: Config loaded and cached")

    # Load again (will use cache)
    config2 = Config.load('configs/data_sources.yaml', env='development')
    print(f"Second load: Using cached config (same instance: {config1 is config2})")

    # Clear cache
    clear_cache()
    print("Cache cleared")

    # Load again (will reload from file)
    config3 = Config.load('configs/data_sources.yaml', env='development')
    print(f"Third load: Reloaded from file (same instance as first: {config1 is config3})")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("BIST AI Trading System - Configuration Management Examples")
    print("=" * 70)

    # Set environment variable for examples
    os.environ['ENVIRONMENT'] = 'development'

    examples = [
        example_1_basic_usage,
        example_2_environment_specific,
        example_3_environment_variables,
        example_4_multiple_configs,
        example_5_global_config,
        example_6_validation,
        example_7_modify_and_save,
        example_8_error_handling,
        example_9_from_dict,
        example_10_cache_management,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
