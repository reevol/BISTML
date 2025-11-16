"""
Unit tests for configuration management system

Run tests with:
    pytest tests/test_config.py -v
"""

import os
import pytest
import tempfile
from pathlib import Path
import yaml

from src.utils.config import (
    Config,
    ConfigError,
    ConfigValidationError,
    ConfigNotFoundError,
    ConfigValidator,
    init_config,
    get_config,
    set_config,
    clear_cache,
    DataSourceConfigValidator,
    SchedulerConfigValidator
)


@pytest.fixture(autouse=True)
def clear_config_cache_fixture():
    """Clear config cache before and after each test"""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary"""
    return {
        'app': {
            'name': 'TestApp',
            'version': '1.0.0',
            'debug': True
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'testdb'
        },
        'api': {
            'timeout': 30,
            'rate_limit': 100,
            'retry': {
                'max_attempts': 3,
                'backoff_factor': 2
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create temporary configuration file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def temp_env_config_file():
    """Create temporary environment-specific config file"""
    env_config = {
        'app': {
            'debug': False  # Override debug setting
        },
        'database': {
            'host': 'prod-db.example.com'  # Override host
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(env_config, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


class TestConfigBasics:
    """Test basic configuration functionality"""

    def test_load_from_dict(self, sample_config_dict):
        """Test creating config from dictionary"""
        config = Config.from_dict(sample_config_dict)
        assert config.get('app.name') == 'TestApp'
        assert config.get('database.port') == 5432

    def test_load_from_file(self, temp_config_file):
        """Test loading config from file"""
        config = Config.load(temp_config_file, validate=False)
        assert config.get('app.name') == 'TestApp'
        assert config.get('database.host') == 'localhost'

    def test_get_nested_values(self, sample_config_dict):
        """Test getting nested configuration values"""
        config = Config.from_dict(sample_config_dict)
        assert config.get('api.retry.max_attempts') == 3
        assert config.get('api.retry.backoff_factor') == 2

    def test_get_with_default(self, sample_config_dict):
        """Test getting values with default"""
        config = Config.from_dict(sample_config_dict)
        assert config.get('nonexistent.key', default='default_value') == 'default_value'
        assert config.get('api.timeout', default=60) == 30  # Actual value

    def test_get_required_success(self, sample_config_dict):
        """Test getting required value that exists"""
        config = Config.from_dict(sample_config_dict)
        assert config.get_required('app.name') == 'TestApp'

    def test_get_required_failure(self, sample_config_dict):
        """Test getting required value that doesn't exist"""
        config = Config.from_dict(sample_config_dict)
        with pytest.raises(ConfigError):
            config.get_required('nonexistent.key')

    def test_has(self, sample_config_dict):
        """Test checking if path exists"""
        config = Config.from_dict(sample_config_dict)
        assert config.has('app.name') is True
        assert config.has('nonexistent.key') is False

    def test_set(self, sample_config_dict):
        """Test setting configuration values"""
        config = Config.from_dict(sample_config_dict)

        # Set existing value
        config.set('app.name', 'NewAppName')
        assert config.get('app.name') == 'NewAppName'

        # Set new nested value
        config.set('new.nested.value', 'test')
        assert config.get('new.nested.value') == 'test'

    def test_to_dict(self, sample_config_dict):
        """Test converting config to dictionary"""
        config = Config.from_dict(sample_config_dict)
        result = config.to_dict()

        assert result['app']['name'] == 'TestApp'
        assert result['database']['port'] == 5432

        # Ensure it's a copy
        result['app']['name'] = 'Modified'
        assert config.get('app.name') == 'TestApp'


class TestEnvironmentSpecific:
    """Test environment-specific configuration"""

    def test_environment_property(self, sample_config_dict):
        """Test environment property"""
        dev_config = Config.from_dict(sample_config_dict, env='development')
        prod_config = Config.from_dict(sample_config_dict, env='production')

        assert dev_config.env == 'development'
        assert prod_config.env == 'production'

    def test_is_development(self, sample_config_dict):
        """Test is_development property"""
        dev_config = Config.from_dict(sample_config_dict, env='development')
        prod_config = Config.from_dict(sample_config_dict, env='production')

        assert dev_config.is_development is True
        assert prod_config.is_development is False

    def test_is_production(self, sample_config_dict):
        """Test is_production property"""
        dev_config = Config.from_dict(sample_config_dict, env='development')
        prod_config = Config.from_dict(sample_config_dict, env='production')

        assert dev_config.is_production is False
        assert prod_config.is_production is True

    def test_is_staging(self, sample_config_dict):
        """Test is_staging property"""
        staging_config = Config.from_dict(sample_config_dict, env='staging')
        dev_config = Config.from_dict(sample_config_dict, env='development')

        assert staging_config.is_staging is True
        assert dev_config.is_staging is False


class TestEnvironmentVariables:
    """Test environment variable substitution"""

    def test_env_var_substitution_basic(self):
        """Test basic environment variable substitution"""
        os.environ['TEST_VAR'] = 'test_value'

        config_dict = {
            'api': {
                'key': '${TEST_VAR}'
            }
        }

        config = Config.from_dict(config_dict)
        assert config.get('api.key') == 'test_value'

        # Cleanup
        del os.environ['TEST_VAR']

    def test_env_var_substitution_with_default(self):
        """Test environment variable substitution with default"""
        # Variable doesn't exist
        config_dict = {
            'api': {
                'key': '${NONEXISTENT_VAR:default_key}'
            }
        }

        config = Config.from_dict(config_dict)
        assert config.get('api.key') == 'default_key'

    def test_env_var_substitution_in_complex_string(self):
        """Test environment variable substitution in complex strings"""
        os.environ['DB_USER'] = 'admin'
        os.environ['DB_PASS'] = 'secret'
        os.environ['DB_HOST'] = 'localhost'

        config_dict = {
            'database': {
                'url': 'postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}/mydb'
            }
        }

        config = Config.from_dict(config_dict)
        assert config.get('database.url') == 'postgresql://admin:secret@localhost/mydb'

        # Cleanup
        del os.environ['DB_USER']
        del os.environ['DB_PASS']
        del os.environ['DB_HOST']


class TestConfigMerging:
    """Test configuration merging"""

    def test_deep_merge(self):
        """Test deep merging of configurations"""
        base = {
            'app': {
                'name': 'BaseApp',
                'version': '1.0.0',
                'debug': True
            },
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }

        override = {
            'app': {
                'debug': False,  # Override
                'new_setting': 'value'  # New
            },
            'database': {
                'host': 'prod-db'  # Override
                # port remains from base
            }
        }

        result = Config._deep_merge(base, override)

        assert result['app']['name'] == 'BaseApp'  # From base
        assert result['app']['debug'] is False  # Overridden
        assert result['app']['new_setting'] == 'value'  # New
        assert result['database']['host'] == 'prod-db'  # Overridden
        assert result['database']['port'] == 5432  # From base


class TestValidation:
    """Test configuration validation"""

    def test_validate_required_fields_success(self, sample_config_dict):
        """Test validation of required fields - success"""
        config = Config.from_dict(sample_config_dict)

        # Should not raise
        config.validate_required_fields([
            'app.name',
            'database.host',
            'api.timeout'
        ])

    def test_validate_required_fields_failure(self, sample_config_dict):
        """Test validation of required fields - failure"""
        config = Config.from_dict(sample_config_dict)

        with pytest.raises(ConfigValidationError):
            config.validate_required_fields([
                'app.name',
                'nonexistent.field'
            ])

    def test_validate_types_success(self, sample_config_dict):
        """Test type validation - success"""
        config = Config.from_dict(sample_config_dict)

        # Should not raise
        config.validate_types({
            'app.name': str,
            'database.port': int,
            'app.debug': bool
        })

    def test_validate_types_failure(self, sample_config_dict):
        """Test type validation - failure"""
        config = Config.from_dict(sample_config_dict)

        with pytest.raises(ConfigValidationError):
            config.validate_types({
                'database.port': str  # Wrong type
            })

    def test_validate_range(self, sample_config_dict):
        """Test range validation"""
        config = Config.from_dict(sample_config_dict)
        validator = ConfigValidator()

        # Should not raise
        validator.validate_range(
            config._config,
            'database.port',
            min_val=1024,
            max_val=65535
        )

        # Should raise
        with pytest.raises(ConfigValidationError):
            validator.validate_range(
                config._config,
                'database.port',
                min_val=10000,
                max_val=65535
            )

    def test_validate_choices(self, sample_config_dict):
        """Test choices validation"""
        config = Config.from_dict(sample_config_dict)
        validator = ConfigValidator()

        # Add a choice field
        config.set('logging.level', 'INFO')

        # Should not raise
        validator.validate_choices(
            config._config,
            'logging.level',
            ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        )

        # Should raise
        with pytest.raises(ConfigValidationError):
            validator.validate_choices(
                config._config,
                'logging.level',
                ['DEBUG', 'WARNING', 'ERROR']  # INFO not in choices
            )


class TestGlobalConfig:
    """Test global configuration singleton"""

    def test_init_config(self, temp_config_file):
        """Test initializing global config"""
        config = init_config(temp_config_file)

        assert config.get('app.name') == 'TestApp'
        assert get_config() is config

    def test_set_config(self, sample_config_dict):
        """Test setting global config"""
        config = Config.from_dict(sample_config_dict)
        set_config(config)

        assert get_config() is config

    def test_get_config_not_initialized(self):
        """Test getting global config before initialization"""
        # Reset global config
        import src.utils.config as config_module
        config_module._global_config = None

        with pytest.raises(ConfigError):
            get_config()


class TestCaching:
    """Test configuration caching"""

    def test_cache_hit(self, temp_config_file):
        """Test that second load uses cache"""
        config1 = Config.load(temp_config_file, validate=False)
        config2 = Config.load(temp_config_file, validate=False)

        # Should be same instance (from cache)
        assert config1 is config2

    def test_cache_different_env(self, temp_config_file):
        """Test that different environments have different cache entries"""
        dev_config = Config.load(temp_config_file, env='development', validate=False)
        prod_config = Config.load(temp_config_file, env='production', validate=False)

        # Should be different instances
        assert dev_config is not prod_config

    def test_clear_cache(self, temp_config_file):
        """Test cache clearing"""
        config1 = Config.load(temp_config_file, validate=False)

        clear_cache()

        config2 = Config.load(temp_config_file, validate=False)

        # Should be different instances (cache was cleared)
        assert config1 is not config2


class TestErrorHandling:
    """Test error handling"""

    def test_file_not_found(self):
        """Test loading non-existent file"""
        with pytest.raises(ConfigNotFoundError):
            Config.load('nonexistent_file.yaml')

    def test_invalid_yaml(self):
        """Test loading invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content:\n  - bad structure')
            temp_path = f.name

        try:
            with pytest.raises(ConfigError):
                Config.load(temp_path)
        finally:
            Path(temp_path).unlink()


class TestSaveLoad:
    """Test saving and loading configurations"""

    def test_save_and_load(self, sample_config_dict):
        """Test saving config to file and loading it back"""
        config = Config.from_dict(sample_config_dict)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.save(temp_path)

            # Load it back
            clear_cache()
            loaded_config = Config.load(temp_path, validate=False)

            # Verify contents
            assert loaded_config.get('app.name') == 'TestApp'
            assert loaded_config.get('database.port') == 5432
            assert loaded_config.get('api.retry.max_attempts') == 3

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestRealWorldConfigs:
    """Test with real project configuration files"""

    def test_data_sources_config(self):
        """Test loading actual data_sources.yaml"""
        config_path = Path('/home/user/BISTML/configs/data_sources.yaml')

        if not config_path.exists():
            pytest.skip("data_sources.yaml not found")

        config = Config.load(config_path, env='development', validate=False)

        # Verify key sections exist
        assert config.has('macro_data')
        assert config.has('collection')
        assert config.has('logging')

    def test_scheduler_config(self):
        """Test loading actual scheduler_config.yaml"""
        config_path = Path('/home/user/BISTML/configs/scheduler_config.yaml')

        if not config_path.exists():
            pytest.skip("scheduler_config.yaml not found")

        config = Config.load(config_path, env='development', validate=False)

        # Verify key sections exist
        assert config.has('scheduler')
        assert config.has('market_hours')
        assert config.has('logging')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
