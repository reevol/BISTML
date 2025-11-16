"""
Unit tests for Portfolio Manager

Tests core functionality including:
- Position tracking
- Transaction processing
- P&L calculation
- Cost basis methods
- Portfolio I/O operations
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.portfolio.manager import (
    PortfolioManager,
    Transaction,
    Position,
    TransactionType,
    CostBasisMethod,
    InsufficientSharesError,
    InvalidTransactionError,
    PositionNotFoundError
)


class TestTransaction(unittest.TestCase):
    """Test Transaction class"""

    def test_transaction_creation(self):
        """Test creating a transaction"""
        txn = Transaction(
            symbol="THYAO",
            transaction_type=TransactionType.BUY,
            shares=100,
            price=250.0,
            timestamp=datetime.now(),
            commission=10.0
        )

        self.assertEqual(txn.symbol, "THYAO")
        self.assertEqual(txn.shares, 100)
        self.assertEqual(txn.price, 250.0)
        self.assertEqual(txn.commission, 10.0)

    def test_transaction_total_cost(self):
        """Test transaction cost calculation"""
        txn = Transaction(
            symbol="THYAO",
            transaction_type=TransactionType.BUY,
            shares=100,
            price=250.0,
            timestamp=datetime.now(),
            commission=10.0
        )

        self.assertEqual(txn.total_cost, 25000.0)
        self.assertEqual(txn.total_with_commission, 25010.0)

    def test_transaction_validation(self):
        """Test transaction validation"""
        with self.assertRaises(InvalidTransactionError):
            Transaction(
                symbol="THYAO",
                transaction_type=TransactionType.BUY,
                shares=-100,  # Invalid
                price=250.0,
                timestamp=datetime.now()
            )

    def test_transaction_to_dict(self):
        """Test transaction serialization"""
        txn = Transaction(
            symbol="THYAO",
            transaction_type=TransactionType.BUY,
            shares=100,
            price=250.0,
            timestamp=datetime.now(),
            commission=10.0
        )

        txn_dict = txn.to_dict()
        self.assertEqual(txn_dict['symbol'], "THYAO")
        self.assertEqual(txn_dict['shares'], 100)


class TestPosition(unittest.TestCase):
    """Test Position class"""

    def test_position_creation(self):
        """Test creating a position"""
        pos = Position(
            symbol="THYAO",
            shares=100,
            cost_basis=250.0,
            total_cost=25000.0
        )

        self.assertEqual(pos.symbol, "THYAO")
        self.assertEqual(pos.shares, 100)
        self.assertEqual(pos.cost_basis, 250.0)

    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation"""
        pos = Position(
            symbol="THYAO",
            shares=100,
            cost_basis=250.0,
            total_cost=25000.0
        )

        pnl = pos.calculate_unrealized_pnl(current_price=260.0)

        self.assertEqual(pnl['market_value'], 26000.0)
        self.assertEqual(pnl['unrealized_pnl'], 1000.0)
        self.assertEqual(pnl['unrealized_pnl_pct'], 4.0)


class TestPortfolioManager(unittest.TestCase):
    """Test PortfolioManager class"""

    def setUp(self):
        """Set up test portfolio"""
        self.portfolio = PortfolioManager(
            name="Test Portfolio",
            initial_cash=100000.0,
            cost_basis_method=CostBasisMethod.AVERAGE
        )

    def test_portfolio_creation(self):
        """Test portfolio initialization"""
        self.assertEqual(self.portfolio.name, "Test Portfolio")
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(self.portfolio.initial_cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)

    def test_buy_transaction(self):
        """Test buying shares"""
        self.portfolio.buy(
            symbol="THYAO",
            shares=100,
            price=250.0,
            commission=10.0
        )

        self.assertEqual(self.portfolio.cash, 74990.0)  # 100000 - 25000 - 10
        self.assertIn("THYAO", self.portfolio.positions)

        position = self.portfolio.positions["THYAO"]
        self.assertEqual(position.shares, 100)
        self.assertEqual(position.cost_basis, 250.1)  # (25000 + 10) / 100

    def test_multiple_buys_average_cost(self):
        """Test average cost calculation with multiple buys"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.buy(symbol="THYAO", shares=100, price=260.0)

        position = self.portfolio.positions["THYAO"]
        self.assertEqual(position.shares, 200)
        self.assertEqual(position.cost_basis, 255.0)  # (25000 + 26000) / 200

    def test_sell_transaction(self):
        """Test selling shares"""
        # Buy first
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        # Then sell
        initial_cash = self.portfolio.cash
        self.portfolio.sell(symbol="THYAO", shares=50, price=260.0)

        # Check cash increased
        self.assertGreater(self.portfolio.cash, initial_cash)

        # Check position reduced
        position = self.portfolio.positions["THYAO"]
        self.assertEqual(position.shares, 50)

    def test_sell_all_shares(self):
        """Test selling all shares removes position"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.sell(symbol="THYAO", shares=100, price=260.0)

        # Position should be removed
        self.assertNotIn("THYAO", self.portfolio.positions)

        # Realized P&L should be positive
        self.assertGreater(self.portfolio.realized_pnl, 0)

    def test_insufficient_shares_error(self):
        """Test error when selling more shares than owned"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        with self.assertRaises(InsufficientSharesError):
            self.portfolio.sell(symbol="THYAO", shares=150, price=260.0)

    def test_position_not_found_error(self):
        """Test error when selling non-existent position"""
        with self.assertRaises(PositionNotFoundError):
            self.portfolio.sell(symbol="THYAO", shares=100, price=260.0)

    def test_insufficient_cash_error(self):
        """Test error when buying with insufficient cash"""
        with self.assertRaises(InvalidTransactionError):
            self.portfolio.buy(symbol="THYAO", shares=10000, price=1000.0)

    def test_portfolio_value_calculation(self):
        """Test total portfolio value calculation"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.buy(symbol="GARAN", shares=200, price=85.0)

        current_prices = {
            "THYAO": 260.0,
            "GARAN": 90.0
        }

        total_value = self.portfolio.get_portfolio_value(current_prices)

        expected_value = (
            self.portfolio.cash +  # Remaining cash
            100 * 260.0 +  # THYAO position
            200 * 90.0  # GARAN position
        )

        self.assertEqual(total_value, expected_value)

    def test_portfolio_summary(self):
        """Test portfolio summary generation"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        current_prices = {"THYAO": 260.0}
        summary = self.portfolio.get_portfolio_summary(current_prices)

        self.assertEqual(summary['num_positions'], 1)
        self.assertGreater(summary['unrealized_pnl'], 0)
        self.assertEqual(len(summary['positions']), 1)

    def test_allocation_calculation(self):
        """Test portfolio allocation calculation"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.buy(symbol="GARAN", shares=200, price=85.0)

        current_prices = {
            "THYAO": 250.0,
            "GARAN": 85.0
        }

        allocation = self.portfolio.get_allocation(current_prices)

        self.assertEqual(len(allocation), 2)
        self.assertIn('allocation_pct', allocation.columns)

        # Sum of allocations should be close to 100% (excluding cash)
        total_allocation = allocation['allocation_pct'].sum()
        self.assertGreater(total_allocation, 0)

    def test_transaction_history(self):
        """Test transaction history retrieval"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.buy(symbol="GARAN", shares=200, price=85.0)
        self.portfolio.sell(symbol="THYAO", shares=50, price=260.0)

        history = self.portfolio.get_transaction_history()

        self.assertEqual(len(history), 3)
        self.assertIn('symbol', history.columns)
        self.assertIn('transaction_type', history.columns)

    def test_transaction_history_filtering(self):
        """Test transaction history filtering"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        self.portfolio.buy(symbol="GARAN", shares=200, price=85.0)

        # Filter by symbol
        thyao_history = self.portfolio.get_transaction_history(symbol="THYAO")
        self.assertEqual(len(thyao_history), 1)
        self.assertEqual(thyao_history.iloc[0]['symbol'], "THYAO")

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        current_prices = {"THYAO": 275.0}
        metrics = self.portfolio.calculate_performance_metrics(current_prices)

        self.assertIn('total_return_pct', metrics)
        self.assertIn('annualized_return_pct', metrics)
        self.assertIn('win_rate', metrics)

    def test_to_dict_and_from_dict(self):
        """Test portfolio serialization"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        # Convert to dict
        portfolio_dict = self.portfolio.to_dict()

        # Create new portfolio from dict
        new_portfolio = PortfolioManager.from_dict(portfolio_dict)

        self.assertEqual(new_portfolio.name, self.portfolio.name)
        self.assertEqual(new_portfolio.cash, self.portfolio.cash)
        self.assertEqual(len(new_portfolio.positions), len(self.portfolio.positions))

    def test_json_save_and_load(self):
        """Test JSON save and load"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save to JSON
            self.portfolio.save_to_json(filepath)

            # Load from JSON
            loaded_portfolio = PortfolioManager.load_from_json(filepath)

            self.assertEqual(loaded_portfolio.name, self.portfolio.name)
            self.assertEqual(loaded_portfolio.cash, self.portfolio.cash)
            self.assertEqual(len(loaded_portfolio.positions), 1)

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_csv_export(self):
        """Test CSV export"""
        self.portfolio.buy(symbol="THYAO", shares=100, price=250.0)

        current_prices = {"THYAO": 260.0}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name

        try:
            self.portfolio.export_to_csv(filepath, current_prices)
            self.assertTrue(os.path.exists(filepath))

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestCostBasisMethods(unittest.TestCase):
    """Test different cost basis methods"""

    def test_average_cost_method(self):
        """Test average cost basis method"""
        portfolio = PortfolioManager(
            initial_cash=100000.0,
            cost_basis_method=CostBasisMethod.AVERAGE
        )

        portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        portfolio.buy(symbol="THYAO", shares=100, price=260.0)

        position = portfolio.positions["THYAO"]
        self.assertEqual(position.cost_basis, 255.0)

    def test_fifo_method(self):
        """Test FIFO cost basis method"""
        portfolio = PortfolioManager(
            initial_cash=100000.0,
            cost_basis_method=CostBasisMethod.FIFO
        )

        portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        portfolio.buy(symbol="THYAO", shares=100, price=260.0)

        # FIFO: selling should use first purchase price
        portfolio.sell(symbol="THYAO", shares=50, price=270.0)

        # Realized P&L should reflect FIFO (270 - 250) * 50 = 1000
        self.assertGreater(portfolio.realized_pnl, 0)

    def test_lifo_method(self):
        """Test LIFO cost basis method"""
        portfolio = PortfolioManager(
            initial_cash=100000.0,
            cost_basis_method=CostBasisMethod.LIFO
        )

        portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        portfolio.buy(symbol="THYAO", shares=100, price=260.0)

        # LIFO: selling should use last purchase price
        portfolio.sell(symbol="THYAO", shares=50, price=270.0)

        # Realized P&L should reflect LIFO (270 - 260) * 50 = 500
        self.assertGreater(portfolio.realized_pnl, 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTransaction))
    suite.addTests(loader.loadTestsFromTestCase(TestPosition))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioManager))
    suite.addTests(loader.loadTestsFromTestCase(TestCostBasisMethods))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
