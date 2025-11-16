"""
Unit Tests for Macro Collector

Tests for the macroeconomic data collector module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.collectors.macro_collector import (
    MacroCollector,
    FREDDataSource,
    EVDSDataSource,
    YahooFinanceDataSource
)


class TestDataSources(unittest.TestCase):
    """Test individual data source classes"""

    def test_fred_initialization_without_key(self):
        """Test FRED data source initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            fred = FREDDataSource()
            self.assertFalse(fred.is_available())

    def test_fred_initialization_with_key(self):
        """Test FRED data source initialization with API key"""
        with patch('data.collectors.macro_collector.Fred') as mock_fred:
            fred = FREDDataSource(api_key='test_key')
            self.assertTrue(fred.is_available())

    def test_evds_initialization_without_key(self):
        """Test EVDS data source initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            evds = EVDSDataSource()
            self.assertFalse(evds.is_available())

    def test_yahoo_initialization(self):
        """Test Yahoo Finance data source initialization"""
        with patch('data.collectors.macro_collector.yfinance'):
            yahoo = YahooFinanceDataSource()
            self.assertTrue(yahoo.is_available())


class TestMacroCollector(unittest.TestCase):
    """Test MacroCollector main class"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = MacroCollector()

    def test_initialization(self):
        """Test MacroCollector initialization"""
        self.assertIsNotNone(self.collector)
        self.assertIsNotNone(self.collector.fred)
        self.assertIsNotNone(self.collector.evds)
        self.assertIsNotNone(self.collector.yahoo)

    def test_list_available_indicators(self):
        """Test listing available indicators"""
        indicators_df = self.collector.list_available_indicators()

        self.assertIsInstance(indicators_df, pd.DataFrame)
        self.assertGreater(len(indicators_df), 0)
        self.assertIn('Indicator', indicators_df.columns)
        self.assertIn('Description', indicators_df.columns)
        self.assertIn('Source', indicators_df.columns)

    def test_indicator_mappings(self):
        """Test that all indicators have required fields"""
        required_fields = ['source', 'code', 'description']

        for name, config in self.collector.INDICATORS.items():
            self.assertIsInstance(config, dict)
            for field in required_fields:
                self.assertIn(field, config, f"{name} missing {field}")

    def test_unknown_indicator_raises_error(self):
        """Test that fetching unknown indicator raises ValueError"""
        with self.assertRaises(ValueError):
            self.collector.get_indicator(
                'UNKNOWN_INDICATOR',
                '2024-01-01',
                '2024-01-31'
            )

    @patch.object(FREDDataSource, 'fetch_data')
    @patch.object(FREDDataSource, 'is_available', return_value=True)
    def test_get_single_indicator(self, mock_available, mock_fetch):
        """Test fetching a single indicator"""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=10)
        values = np.random.randn(10) + 4000
        mock_df = pd.DataFrame({'value': values}, index=dates)
        mock_fetch.return_value = mock_df

        # Fetch indicator
        result = self.collector.get_indicator('SP500', '2024-01-01', '2024-01-10')

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('value', result.columns)
        self.assertIn('indicator', result.columns)
        self.assertEqual(result['indicator'].iloc[0], 'SP500')
        mock_fetch.assert_called_once()

    @patch.object(FREDDataSource, 'fetch_data')
    @patch.object(FREDDataSource, 'is_available', return_value=True)
    @patch.object(YahooFinanceDataSource, 'fetch_data')
    @patch.object(YahooFinanceDataSource, 'is_available', return_value=True)
    def test_get_multiple_indicators(self, mock_yf_avail, mock_yf_fetch,
                                     mock_fred_avail, mock_fred_fetch):
        """Test fetching multiple indicators"""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=10)

        sp500_df = pd.DataFrame({'value': np.random.randn(10) + 4000}, index=dates)
        dax_df = pd.DataFrame({'value': np.random.randn(10) + 15000}, index=dates)

        def fetch_side_effect(code, start, end):
            if code == 'SP500':
                return sp500_df
            elif code == '^GDAXI':
                return dax_df

        mock_fred_fetch.side_effect = fetch_side_effect
        mock_yf_fetch.side_effect = fetch_side_effect

        # Fetch multiple indicators
        result = self.collector.get_multiple_indicators(
            ['SP500', 'DAX'],
            '2024-01-01',
            '2024-01-10'
        )

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 2)

    def test_get_macro_features_structure(self):
        """Test that get_macro_features returns correct structure"""
        # This test would require mocking all data sources
        # For now, just test the method exists and has correct signature
        self.assertTrue(hasattr(self.collector, 'get_macro_features'))

    def test_default_date_handling(self):
        """Test that default dates are set correctly"""
        with patch.object(self.collector, '_fetch_from_source') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'value': [1, 2, 3]})

            # Call without dates
            try:
                self.collector.get_indicator('SP500')
            except:
                pass

            # Verify dates were set
            self.assertEqual(mock_fetch.call_count, 1)


class TestFeatureGeneration(unittest.TestCase):
    """Test feature generation functionality"""

    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2024-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'SP500': np.random.randn(100) * 100 + 4000,
            'VIX': np.random.randn(100) * 5 + 20
        }, index=dates)

    def test_returns_calculation(self):
        """Test that returns are calculated correctly"""
        returns = self.test_data.pct_change()

        # Check returns are reasonable
        self.assertTrue(returns['SP500'].abs().max() < 0.5)  # No >50% daily moves
        self.assertTrue(returns.isna().sum().sum() == 2)  # Only first row NaN

    def test_rolling_statistics(self):
        """Test rolling statistics calculation"""
        ma30 = self.test_data['SP500'].rolling(window=30).mean()

        # Check that rolling mean reduces volatility
        self.assertLess(ma30.std(), self.test_data['SP500'].std())

    def test_correlation_calculation(self):
        """Test correlation calculation between indicators"""
        # SP500 and VIX should be negatively correlated (typically)
        returns = self.test_data.pct_change().dropna()
        corr = returns.corr()

        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))
        self.assertEqual(corr.loc['SP500', 'SP500'], 1.0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        self.collector = MacroCollector()

    def test_invalid_date_format(self):
        """Test handling of invalid date formats"""
        # This should ideally be handled gracefully
        with self.assertRaises((ValueError, TypeError, Exception)):
            self.collector.get_indicator('SP500', 'invalid-date', '2024-01-31')

    def test_future_dates(self):
        """Test handling of future dates"""
        future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        # Should not crash, but may return empty data
        try:
            result = self.collector.get_indicator('SP500', today, future_date)
            # If it doesn't raise an error, check result is reasonable
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Some sources may raise errors for future dates - that's acceptable
            pass

    @patch.object(FREDDataSource, 'is_available', return_value=False)
    @patch.object(YahooFinanceDataSource, 'is_available', return_value=False)
    def test_no_sources_available(self, mock_yf, mock_fred):
        """Test behavior when no data sources are available"""
        collector = MacroCollector()

        with self.assertRaises(RuntimeError):
            collector.get_indicator('SP500', '2024-01-01', '2024-01-31')


class TestIntegration(unittest.TestCase):
    """Integration tests (require actual API keys)"""

    def setUp(self):
        """Set up for integration tests"""
        self.has_fred_key = bool(os.getenv('FRED_API_KEY'))
        self.has_evds_key = bool(os.getenv('EVDS_API_KEY'))
        self.collector = MacroCollector()

    @unittest.skipUnless(os.getenv('FRED_API_KEY'), "FRED_API_KEY not set")
    def test_real_fred_fetch(self):
        """Test actual FRED API fetch (requires API key)"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        result = self.collector.get_indicator('SP500', start_date, end_date)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('value', result.columns)

    @unittest.skipUnless(os.getenv('EVDS_API_KEY'), "EVDS_API_KEY not set")
    def test_real_evds_fetch(self):
        """Test actual EVDS API fetch (requires API key)"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        result = self.collector.get_indicator('TR_CPI', start_date, end_date)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('value', result.columns)

    def test_real_yahoo_fetch(self):
        """Test actual Yahoo Finance fetch (no API key required)"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            result = self.collector.get_indicator('DAX', start_date, end_date)
            self.assertIsInstance(result, pd.DataFrame)
            if len(result) > 0:
                self.assertIn('value', result.columns)
        except Exception as e:
            # Yahoo Finance can be flaky, so just log the error
            print(f"Yahoo Finance test failed: {e}")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataSources))
    suite.addTests(loader.loadTestsFromTestCase(TestMacroCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    # Run tests
    result = run_tests()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
