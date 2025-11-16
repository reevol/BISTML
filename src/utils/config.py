"""
Configuration Management Module for BIST AI Trading System

This module provides comprehensive configuration management with:
- YAML file loading and parsing
- Environment-specific configurations (dev, staging, prod)
- Environment variable substitution
- Configuration validation
- Type-safe access to configuration values
- Configuration merging and overrides
- Singleton pattern for global config access

Usage:
    from src.utils.config import Config, get_config

    # Load configuration
    config = Config.load('configs/data_sources.yaml', env='production')

    # Access values
    api_key = config.get('macro_data.fred.api_key')
    rate_limit = config.get('macro_data.fred.rate_limit', default=100)

    # Or use singleton
    config = get_config()
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from copy import deepcopy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails"""
    pass


class ConfigNotFoundError(ConfigError):
    """Exception raised when configuration file is not found"""
    pass


class ConfigValidator:
    """Validates configuration schemas and values"""

    @staticmethod
    def validate_required_fields(config: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Validate that required fields are present in configuration

        Args:
            config: Configuration dictionary
            required_fields: List of required field paths (e.g., 'macro_data.fred.api_key')

        Raises:
            ConfigValidationError: If required field is missing
        """
        missing_fields = []
        for field_path in required_fields:
            if not ConfigValidator._get_nested_value(config, field_path):
                missing_fields.append(field_path)

        if missing_fields:
            raise ConfigValidationError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )

    @staticmethod
    def validate_types(config: Dict[str, Any], type_specs: Dict[str, type]) -> None:
        """
        Validate that configuration values match expected types

        Args:
            config: Configuration dictionary
            type_specs: Dictionary mapping field paths to expected types

        Raises:
            ConfigValidationError: If type validation fails
        """
        type_errors = []
        for field_path, expected_type in type_specs.items():
            value = ConfigValidator._get_nested_value(config, field_path)
            if value is not None and not isinstance(value, expected_type):
                type_errors.append(
                    f"{field_path}: expected {expected_type.__name__}, got {type(value).__name__}"
                )

        if type_errors:
            raise ConfigValidationError(
                f"Type validation errors:\n" + "\n".join(type_errors)
            )

    @staticmethod
    def validate_range(config: Dict[str, Any], field_path: str, min_val: Any = None,
                      max_val: Any = None) -> None:
        """
        Validate that a numeric configuration value is within specified range

        Args:
            config: Configuration dictionary
            field_path: Path to the field
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Raises:
            ConfigValidationError: If value is out of range
        """
        value = ConfigValidator._get_nested_value(config, field_path)
        if value is None:
            return

        if min_val is not None and value < min_val:
            raise ConfigValidationError(
                f"{field_path}: value {value} is less than minimum {min_val}"
            )

        if max_val is not None and value > max_val:
            raise ConfigValidationError(
                f"{field_path}: value {value} is greater than maximum {max_val}"
            )

    @staticmethod
    def validate_choices(config: Dict[str, Any], field_path: str, choices: List[Any]) -> None:
        """
        Validate that a configuration value is one of allowed choices

        Args:
            config: Configuration dictionary
            field_path: Path to the field
            choices: List of allowed values

        Raises:
            ConfigValidationError: If value is not in choices
        """
        value = ConfigValidator._get_nested_value(config, field_path)
        if value is not None and value not in choices:
            raise ConfigValidationError(
                f"{field_path}: value '{value}' not in allowed choices {choices}"
            )

    @staticmethod
    def _get_nested_value(config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


class Config:
    """
    Configuration management class with environment-specific support
    """

    # Class-level cache for loaded configurations
    _instances: Dict[str, 'Config'] = {}
    _global_instance: Optional['Config'] = None

    def __init__(self, config_data: Dict[str, Any], env: str = 'development'):
        """
        Initialize configuration

        Args:
            config_data: Configuration dictionary
            env: Environment name (development, staging, production)
        """
        self._config = config_data
        self._env = env
        self._loaded_at = datetime.now()
        self._validator = ConfigValidator()

    @classmethod
    def load(cls, config_path: Union[str, Path], env: Optional[str] = None,
             validate: bool = True) -> 'Config':
        """
        Load configuration from YAML file

        Args:
            config_path: Path to configuration file
            env: Environment name (defaults to ENVIRONMENT env var or 'development')
            validate: Whether to validate configuration

        Returns:
            Config instance

        Raises:
            ConfigNotFoundError: If config file not found
            ConfigValidationError: If validation fails
        """
        # Determine environment
        if env is None:
            env = os.getenv('ENVIRONMENT', 'development')

        # Convert to Path object
        config_path = Path(config_path)

        # Check cache
        cache_key = f"{config_path}:{env}"
        if cache_key in cls._instances:
            logger.debug(f"Using cached config for {cache_key}")
            return cls._instances[cache_key]

        # Load base configuration
        base_config = cls._load_yaml_file(config_path)

        # Load environment-specific override if exists
        env_config_path = cls._get_env_config_path(config_path, env)
        if env_config_path.exists():
            logger.info(f"Loading environment-specific config: {env_config_path}")
            env_config = cls._load_yaml_file(env_config_path)
            base_config = cls._deep_merge(base_config, env_config)

        # Substitute environment variables
        base_config = cls._substitute_env_vars(base_config)

        # Create instance
        instance = cls(base_config, env)

        # Validate if requested
        if validate:
            instance._validate_config()

        # Cache instance
        cls._instances[cache_key] = instance

        logger.info(f"Loaded configuration from {config_path} (env: {env})")
        return instance

    @classmethod
    def load_multiple(cls, config_paths: List[Union[str, Path]],
                     env: Optional[str] = None) -> 'Config':
        """
        Load and merge multiple configuration files

        Args:
            config_paths: List of configuration file paths
            env: Environment name

        Returns:
            Merged Config instance
        """
        if not config_paths:
            raise ConfigError("No configuration paths provided")

        # Load first config
        merged_config = cls.load(config_paths[0], env=env, validate=False)

        # Merge remaining configs
        for config_path in config_paths[1:]:
            config = cls.load(config_path, env=env, validate=False)
            merged_config._config = cls._deep_merge(
                merged_config._config,
                config._config
            )

        # Validate merged config
        merged_config._validate_config()

        return merged_config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], env: str = 'development',
                  substitute_env_vars: bool = True) -> 'Config':
        """
        Create configuration from dictionary

        Args:
            config_dict: Configuration dictionary
            env: Environment name
            substitute_env_vars: Whether to substitute environment variables

        Returns:
            Config instance
        """
        # Substitute environment variables if requested
        if substitute_env_vars:
            config_dict = cls._substitute_env_vars(config_dict)

        return cls(config_dict, env)

    @staticmethod
    def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
        """Load YAML file and return parsed data"""
        if not file_path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading configuration file {file_path}: {e}")

    @staticmethod
    def _get_env_config_path(base_path: Path, env: str) -> Path:
        """
        Get environment-specific configuration file path

        Examples:
            config.yaml -> config.production.yaml
            data_sources.yaml -> data_sources.staging.yaml
        """
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        return parent / f"{stem}.{env}{suffix}"

    @staticmethod
    def _substitute_env_vars(data: Any) -> Any:
        """
        Recursively substitute environment variables in configuration

        Supports formats:
            ${VAR_NAME}
            ${VAR_NAME:default_value}
        """
        if isinstance(data, dict):
            return {k: Config._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return Config._replace_env_var(data)
        else:
            return data

    @staticmethod
    def _replace_env_var(value: str) -> str:
        """Replace environment variable placeholders in string"""
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replacer, value)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            path: Dot-separated path to value (e.g., 'macro_data.fred.api_key')
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            api_key = config.get('macro_data.fred.api_key')
            rate_limit = config.get('macro_data.fred.rate_limit', default=100)
        """
        keys = path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_required(self, path: str) -> Any:
        """
        Get required configuration value, raise error if not found

        Args:
            path: Dot-separated path to value

        Returns:
            Configuration value

        Raises:
            ConfigError: If value not found
        """
        value = self.get(path)
        if value is None:
            raise ConfigError(f"Required configuration key not found: {path}")
        return value

    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            path: Dot-separated path to value
            value: Value to set
        """
        keys = path.split('.')
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def has(self, path: str) -> bool:
        """
        Check if configuration path exists

        Args:
            path: Dot-separated path to check

        Returns:
            True if path exists, False otherwise
        """
        return self.get(path) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return deepcopy(self._config)

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    def validate_required_fields(self, required_fields: List[str]) -> None:
        """Validate that required fields are present"""
        self._validator.validate_required_fields(self._config, required_fields)

    def validate_types(self, type_specs: Dict[str, type]) -> None:
        """Validate configuration value types"""
        self._validator.validate_types(self._config, type_specs)

    def _validate_config(self) -> None:
        """
        Validate configuration based on environment
        Override this method or use validate_* methods for custom validation
        """
        # Base validation - can be extended based on config type
        pass

    @property
    def env(self) -> str:
        """Get current environment"""
        return self._env

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self._env.lower() in ('production', 'prod')

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self._env.lower() in ('development', 'dev')

    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self._env.lower() == 'staging'

    def __repr__(self) -> str:
        return f"Config(env={self._env}, loaded_at={self._loaded_at})"

    def __str__(self) -> str:
        return yaml.dump(self._config, default_flow_style=False)


# Singleton pattern for global config access
_global_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance (singleton)

    Args:
        reload: Force reload configuration

    Returns:
        Global Config instance

    Raises:
        ConfigError: If config not initialized
    """
    global _global_config

    if _global_config is None or reload:
        raise ConfigError(
            "Global configuration not initialized. "
            "Call init_config() or set_config() first."
        )

    return _global_config


def set_config(config: Config) -> None:
    """
    Set global configuration instance

    Args:
        config: Config instance to use as global
    """
    global _global_config
    _global_config = config
    logger.info(f"Global configuration set (env: {config.env})")


def init_config(config_paths: Union[str, Path, List[Union[str, Path]]],
                env: Optional[str] = None) -> Config:
    """
    Initialize global configuration from file(s)

    Args:
        config_paths: Single path or list of paths to configuration files
        env: Environment name

    Returns:
        Initialized global Config instance
    """
    global _global_config

    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]

    if len(config_paths) == 1:
        _global_config = Config.load(config_paths[0], env=env)
    else:
        _global_config = Config.load_multiple(config_paths, env=env)

    return _global_config


def clear_cache() -> None:
    """Clear configuration cache"""
    Config._instances.clear()
    logger.info("Configuration cache cleared")


# Pre-configured validators for common patterns
class DataSourceConfigValidator:
    """Validator for data source configurations"""

    @staticmethod
    def validate(config: Config) -> None:
        """Validate data source configuration"""
        # Check required API sections
        required_apis = ['macro_data']
        for api in required_apis:
            if not config.has(api):
                raise ConfigValidationError(f"Missing required section: {api}")

        # Validate rate limits
        if config.has('macro_data.fred.rate_limit'):
            config._validator.validate_range(
                config._config,
                'macro_data.fred.rate_limit',
                min_val=1,
                max_val=10000
            )

        # Validate cache settings
        if config.has('collection.cache.enabled'):
            config._validator.validate_types(
                config._config,
                {'collection.cache.enabled': bool}
            )


class SchedulerConfigValidator:
    """Validator for scheduler configurations"""

    @staticmethod
    def validate(config: Config) -> None:
        """Validate scheduler configuration"""
        # Validate timezone
        if config.has('scheduler.timezone'):
            config._validator.validate_choices(
                config._config,
                'scheduler.timezone',
                ['Europe/Istanbul', 'UTC', 'America/New_York']
            )

        # Validate log level
        if config.has('logging.level'):
            config._validator.validate_choices(
                config._config,
                'logging.level',
                ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            )

        # Validate retry settings
        if config.has('execution.max_retries'):
            config._validator.validate_range(
                config._config,
                'execution.max_retries',
                min_val=0,
                max_val=10
            )
