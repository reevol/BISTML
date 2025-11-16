"""
Unit tests for Backtesting Simulator

This module provides tests for the backtesting simulator functionality.

Author: BIST AI Trading System
Date: 2025-11-16
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from simulator import (
    BacktestSimulator,
    BacktestConfig,
    WalkForwardConfig,
    MonteCarloConfig,
    BacktestMode,
    ExecutionModel,
    Trade,
    BacktestResult
)


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig class"""

    def test_config_creation(self):
        """Test creating a basic config"""
        config = BacktestConfig(
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=100000.0
        )

        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.commission_rate, 0.001)
        self.assertIsInstance(config.start_date, datetime)

    def test_config_validation(self):
        """Test config validation"""
        with self.assertRaises(ValueError):
            # start_date after end_date
            BacktestConfig(
                start_date='2023-12-31',
                end_date='2023-01-01',
                initial_capital=100000.0
            )

        with self.assertRaises(ValueError):
            # negative capital
            BacktestConfig(
                start_date='2023-01-01',
                end_date='2023-12-31',
                initial_capital=-1000.0
            )


class TestTrade(unittest.TestCase):
    """Test Trade class"""

    def test_trade_creation(self):
        """Test creating a trade"""
        trade = Trade(
            symbol='THYAO',
            entry_time=datetime(2023, 1, 1),
            exit_time=datetime(2023, 1, 5),
            entry_price=100.0,
            exit_price=105.0,
            shares=100.0,
            direction=1,
            commission=10.0,
            slippage=5.0
        )

        # P&L should be calculated automatically
        expected_pnl = (105.0 - 100.0) * 100.0 - 10.0 - 5.0
        self.assertAlmostEqual(trade.pnl, expected_pnl)

        # P&L percentage
        self.assertGreater(trade.pnl_pct, 0)

        # Holding period
        self.assertEqual(trade.holding_period, 4)


class TestBacktestSimulator(unittest.TestCase):
    """Test BacktestSimulator class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='1D')

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))

        self.price_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'THYAO',
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': 1000000
        })
        self.price_data.set_index('timestamp', inplace=True)

        # Simple signal generator
        def simple_signals(timestamp, market_data):
            signals = []
            for _, row in market_data.iterrows():
                if np.random.random() > 0.9:
                    signals.append({
                        'stock_code': row['symbol'],
                        'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
                        'confidence_score': 0.7
                    })
            return signals

        self.signal_generator = simple_signals

        self.config = BacktestConfig(
            start_date='2023-01-01',
            end_date='2023-03-31',
            initial_capital=100000.0,
            timeframe='1d'
        )

    def test_simulator_initialization(self):
        """Test simulator initialization"""
        simulator = BacktestSimulator(
            config=self.config,
            signal_generator=self.signal_generator,
            price_data=self.price_data
        )

        self.assertEqual(simulator.equity, self.config.initial_capital)
        self.assertEqual(simulator.cash, self.config.initial_capital)
        self.assertEqual(len(simulator.positions), 0)
        self.assertEqual(len(simulator.trades), 0)

    def test_historical_backtest(self):
        """Test running a historical backtest"""
        simulator = BacktestSimulator(
            config=self.config,
            signal_generator=self.signal_generator,
            price_data=self.price_data
        )

        result = simulator.run_historical_backtest()

        # Check result structure
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.trades, list)
        self.assertIsInstance(result.equity_curve, pd.DataFrame)
        self.assertIsInstance(result.metrics, dict)

        # Check metrics exist
        self.assertIn('n_trades', result.metrics)
        self.assertIn('total_return', result.metrics)
        self.assertIn('sharpe_ratio', result.metrics)
        self.assertIn('max_drawdown', result.metrics)

    def test_empty_signals(self):
        """Test backtest with no signals"""
        def no_signals(timestamp, market_data):
            return []

        simulator = BacktestSimulator(
            config=self.config,
            signal_generator=no_signals,
            price_data=self.price_data
        )

        result = simulator.run_historical_backtest()

        # Should have no trades
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.metrics['n_trades'], 0)


class TestWalkForward(unittest.TestCase):
    """Test walk-forward analysis"""

    def setUp(self):
        """Set up test fixtures"""
        # Create longer price data for walk-forward
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='1D')

        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, len(dates))))

        self.price_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'THYAO',
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': 1000000
        })
        self.price_data.set_index('timestamp', inplace=True)

        def simple_signals(timestamp, market_data):
            signals = []
            for _, row in market_data.iterrows():
                if np.random.random() > 0.85:
                    signals.append({
                        'stock_code': row['symbol'],
                        'signal': 'BUY' if np.random.random() > 0.5 else 'SELL'
                    })
            return signals

        self.signal_generator = simple_signals

        self.config = BacktestConfig(
            start_date='2022-01-01',
            end_date='2023-12-31',
            initial_capital=100000.0,
            timeframe='1d'
        )

    def test_wf_period_generation(self):
        """Test walk-forward period generation"""
        simulator = BacktestSimulator(
            config=self.config,
            signal_generator=self.signal_generator,
            price_data=self.price_data
        )

        wf_config = WalkForwardConfig(
            train_period_days=180,
            test_period_days=60,
            step_size_days=60
        )

        periods = simulator._generate_wf_periods(wf_config)

        # Should have multiple periods
        self.assertGreater(len(periods), 0)

        # Each period should have correct structure
        for period in periods:
            self.assertIn('train_start', period)
            self.assertIn('train_end', period)
            self.assertIn('test_start', period)
            self.assertIn('test_end', period)

            # Test should start after train
            self.assertGreater(period['test_start'], period['train_start'])


class TestMonteCarlo(unittest.TestCase):
    """Test Monte Carlo simulation"""

    def test_trade_randomization(self):
        """Test trade randomization methods"""
        # Create sample trades
        trades = []
        for i in range(50):
            trade = Trade(
                symbol='THYAO',
                entry_time=datetime(2023, 1, 1) + timedelta(days=i),
                exit_time=datetime(2023, 1, 1) + timedelta(days=i+1),
                entry_price=100.0,
                exit_price=100.0 + np.random.normal(0, 2),
                shares=100.0,
                direction=1
            )
            trades.append(trade)

        config = BacktestConfig(
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=100000.0
        )

        simulator = BacktestSimulator(config=config)

        # Test shuffle method
        mc_config = MonteCarloConfig(
            n_simulations=10,
            randomization_method='shuffle',
            random_seed=42
        )

        shuffled = simulator._randomize_trades(trades, mc_config)
        self.assertEqual(len(shuffled), len(trades))

        # Test bootstrap method
        mc_config.randomization_method = 'bootstrap'
        bootstrapped = simulator._randomize_trades(trades, mc_config)
        self.assertEqual(len(bootstrapped), len(trades))

        # Test block bootstrap
        mc_config.randomization_method = 'block_bootstrap'
        mc_config.block_length = 10
        block_boot = simulator._randomize_trades(trades, mc_config)
        self.assertEqual(len(block_boot), len(trades))


class TestMetrics(unittest.TestCase):
    """Test performance metrics calculation"""

    def test_metric_calculation(self):
        """Test calculating metrics from trades"""
        # Create sample trades
        trades = [
            Trade('THYAO', datetime(2023,1,1), datetime(2023,1,2),
                  100, 105, 100, 1),  # Win
            Trade('THYAO', datetime(2023,1,3), datetime(2023,1,4),
                  100, 95, 100, 1),   # Loss
            Trade('THYAO', datetime(2023,1,5), datetime(2023,1,6),
                  100, 110, 100, 1),  # Win
        ]

        # Create dummy equity curve
        equity_curve = pd.DataFrame({
            'equity': [100000, 105000, 100000, 110000],
            'cash': [100000, 105000, 100000, 110000],
            'positions_value': [0, 0, 0, 0],
            'n_positions': [0, 0, 0, 0]
        }, index=pd.date_range('2023-01-01', periods=4, freq='1D'))

        config = BacktestConfig(
            start_date='2023-01-01',
            end_date='2023-01-04',
            initial_capital=100000.0
        )

        simulator = BacktestSimulator(config=config)
        metrics = simulator._calculate_performance_metrics(trades, equity_curve)

        # Check basic metrics
        self.assertEqual(metrics['n_trades'], 3)
        self.assertEqual(metrics['n_winners'], 2)
        self.assertEqual(metrics['n_losers'], 1)
        self.assertAlmostEqual(metrics['win_rate'], 2/3)

        # Check P&L metrics
        self.assertIn('total_pnl', metrics)
        self.assertIn('avg_pnl', metrics)

        # Check risk metrics
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestTrade))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForward))
    suite.addTests(loader.loadTestsFromTestCase(TestMonteCarlo))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("Running Backtesting Simulator Tests")
    print("=" * 80)
    result = run_tests()
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
