"""
Portfolio Manager - Position Tracking and P&L Management for BIST

This module provides comprehensive portfolio management functionality including:
- Position tracking with real-time holdings
- Cost basis calculation (FIFO, LIFO, Average)
- Profit & Loss (P&L) calculation (realized and unrealized)
- Portfolio performance metrics
- Portfolio I/O (JSON, CSV, pickle)
- Transaction history management
- Risk metrics and portfolio analytics

Features:
- Multiple cost basis methods (FIFO, LIFO, Average Cost)
- Real-time P&L tracking
- Commission and fee handling
- Position sizing and allocation
- Portfolio diversification metrics
- Risk-adjusted returns
- Export/import functionality

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class TransactionType(Enum):
    """Transaction types"""
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    MERGER = "MERGER"
    SPINOFF = "SPINOFF"


class CostBasisMethod(Enum):
    """Cost basis calculation methods"""
    FIFO = "FIFO"  # First In, First Out
    LIFO = "LIFO"  # Last In, First Out
    AVERAGE = "AVERAGE"  # Average Cost
    SPECIFIC = "SPECIFIC"  # Specific Identification


# ============================================================================
# Exceptions
# ============================================================================

class PortfolioError(Exception):
    """Base exception for portfolio errors"""
    pass


class InsufficientSharesError(PortfolioError):
    """Raised when trying to sell more shares than owned"""
    pass


class InvalidTransactionError(PortfolioError):
    """Raised when transaction is invalid"""
    pass


class PositionNotFoundError(PortfolioError):
    """Raised when position doesn't exist"""
    pass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Transaction:
    """
    Represents a single transaction

    Attributes:
        symbol: Stock symbol
        transaction_type: Type of transaction (BUY, SELL, etc.)
        shares: Number of shares
        price: Price per share
        timestamp: Transaction timestamp
        commission: Commission/fees paid
        notes: Optional notes
        transaction_id: Unique transaction identifier
    """
    symbol: str
    transaction_type: TransactionType
    shares: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    notes: str = ""
    transaction_id: str = field(default_factory=lambda: f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")

    def __post_init__(self):
        """Validate transaction data"""
        if self.shares <= 0:
            raise InvalidTransactionError("Shares must be positive")
        if self.price < 0:
            raise InvalidTransactionError("Price cannot be negative")
        if self.commission < 0:
            raise InvalidTransactionError("Commission cannot be negative")

        # Convert string to enum if needed
        if isinstance(self.transaction_type, str):
            self.transaction_type = TransactionType(self.transaction_type)

        # Convert string to datetime if needed
        if isinstance(self.timestamp, str):
            self.timestamp = pd.to_datetime(self.timestamp)

    @property
    def total_cost(self) -> float:
        """Calculate total transaction cost (excluding commission)"""
        return self.shares * self.price

    @property
    def total_with_commission(self) -> float:
        """Calculate total transaction cost including commission"""
        if self.transaction_type == TransactionType.BUY:
            return self.total_cost + self.commission
        else:  # SELL
            return self.total_cost - self.commission

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['transaction_type'] = self.transaction_type.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """Create from dictionary"""
        data = data.copy()
        data['transaction_type'] = TransactionType(data['transaction_type'])
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return cls(**data)


@dataclass
class Position:
    """
    Represents a position in a single security

    Attributes:
        symbol: Stock symbol
        shares: Current number of shares held
        cost_basis: Average cost per share
        total_cost: Total cost of position
        transactions: List of all transactions for this position
        last_updated: Last update timestamp
    """
    symbol: str
    shares: float = 0.0
    cost_basis: float = 0.0
    total_cost: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Market value (requires current price)"""
        return 0.0  # To be calculated with current price

    def calculate_unrealized_pnl(self, current_price: float) -> Dict[str, float]:
        """
        Calculate unrealized P&L

        Args:
            current_price: Current market price

        Returns:
            Dictionary with P&L metrics
        """
        market_value = self.shares * current_price
        unrealized_pnl = market_value - self.total_cost
        unrealized_pnl_pct = (unrealized_pnl / self.total_cost * 100) if self.total_cost > 0 else 0.0

        return {
            'market_value': market_value,
            'cost_basis': self.cost_basis,
            'total_cost': self.total_cost,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'shares': self.shares
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'cost_basis': self.cost_basis,
            'total_cost': self.total_cost,
            'last_updated': self.last_updated.isoformat(),
            'transactions': [t.to_dict() for t in self.transactions]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary"""
        data = data.copy()
        data['last_updated'] = pd.to_datetime(data['last_updated'])
        data['transactions'] = [Transaction.from_dict(t) for t in data['transactions']]
        return cls(**data)


# ============================================================================
# Portfolio Manager Class
# ============================================================================

class PortfolioManager:
    """
    Comprehensive portfolio management system

    Manages positions, tracks transactions, calculates P&L, and provides
    portfolio analytics for BIST stocks.
    """

    def __init__(
        self,
        name: str = "My Portfolio",
        initial_cash: float = 0.0,
        cost_basis_method: CostBasisMethod = CostBasisMethod.AVERAGE,
        currency: str = "TRY"
    ):
        """
        Initialize Portfolio Manager

        Args:
            name: Portfolio name
            initial_cash: Initial cash balance
            cost_basis_method: Method for calculating cost basis
            currency: Portfolio currency (default: TRY for Turkish Lira)
        """
        self.name = name
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.cost_basis_method = cost_basis_method
        self.currency = currency

        # Positions dictionary: symbol -> Position
        self.positions: Dict[str, Position] = {}

        # Transaction history
        self.transaction_history: List[Transaction] = []

        # Realized P&L tracking
        self.realized_pnl = 0.0
        self.total_commissions = 0.0

        # Metadata
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        logger.info(f"Portfolio '{name}' initialized with {initial_cash} {currency}")

    # ========================================================================
    # Transaction Methods
    # ========================================================================

    def buy(
        self,
        symbol: str,
        shares: float,
        price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
        notes: str = ""
    ) -> Transaction:
        """
        Execute a buy transaction

        Args:
            symbol: Stock symbol
            shares: Number of shares to buy
            price: Purchase price per share
            timestamp: Transaction timestamp (default: now)
            commission: Commission/fees paid
            notes: Optional notes

        Returns:
            Transaction object

        Raises:
            InvalidTransactionError: If insufficient cash
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create transaction
        transaction = Transaction(
            symbol=symbol,
            transaction_type=TransactionType.BUY,
            shares=shares,
            price=price,
            timestamp=timestamp,
            commission=commission,
            notes=notes
        )

        total_cost = transaction.total_with_commission

        # Check if we have enough cash
        if total_cost > self.cash:
            raise InvalidTransactionError(
                f"Insufficient cash: need {total_cost:.2f}, have {self.cash:.2f}"
            )

        # Update cash
        self.cash -= total_cost
        self.total_commissions += commission

        # Update position
        self._update_position_buy(transaction)

        # Add to history
        self.transaction_history.append(transaction)
        self.last_updated = datetime.now()

        logger.info(
            f"BUY: {shares} shares of {symbol} @ {price:.2f} "
            f"(total: {total_cost:.2f} {self.currency})"
        )

        return transaction

    def sell(
        self,
        symbol: str,
        shares: float,
        price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0,
        notes: str = ""
    ) -> Transaction:
        """
        Execute a sell transaction

        Args:
            symbol: Stock symbol
            shares: Number of shares to sell
            price: Sale price per share
            timestamp: Transaction timestamp (default: now)
            commission: Commission/fees paid
            notes: Optional notes

        Returns:
            Transaction object

        Raises:
            InsufficientSharesError: If trying to sell more shares than owned
            PositionNotFoundError: If position doesn't exist
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check if position exists
        if symbol not in self.positions:
            raise PositionNotFoundError(f"No position in {symbol}")

        position = self.positions[symbol]

        # Check if we have enough shares
        if shares > position.shares:
            raise InsufficientSharesError(
                f"Insufficient shares: trying to sell {shares}, have {position.shares}"
            )

        # Create transaction
        transaction = Transaction(
            symbol=symbol,
            transaction_type=TransactionType.SELL,
            shares=shares,
            price=price,
            timestamp=timestamp,
            commission=commission,
            notes=notes
        )

        total_proceeds = transaction.total_with_commission

        # Calculate realized P&L
        cost_basis_for_shares = self._calculate_cost_basis_for_sale(symbol, shares)
        realized_pnl = total_proceeds - cost_basis_for_shares
        self.realized_pnl += realized_pnl

        # Update cash
        self.cash += total_proceeds
        self.total_commissions += commission

        # Update position
        self._update_position_sell(transaction, cost_basis_for_shares)

        # Add to history
        self.transaction_history.append(transaction)
        self.last_updated = datetime.now()

        logger.info(
            f"SELL: {shares} shares of {symbol} @ {price:.2f} "
            f"(proceeds: {total_proceeds:.2f}, realized P&L: {realized_pnl:.2f} {self.currency})"
        )

        return transaction

    def _update_position_buy(self, transaction: Transaction):
        """Update position for a buy transaction"""
        symbol = transaction.symbol

        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        # Update position based on cost basis method
        if self.cost_basis_method == CostBasisMethod.AVERAGE:
            # Average cost method
            total_cost_before = position.total_cost
            total_shares_before = position.shares

            new_total_cost = total_cost_before + transaction.total_with_commission
            new_total_shares = total_shares_before + transaction.shares

            position.shares = new_total_shares
            position.total_cost = new_total_cost
            position.cost_basis = new_total_cost / new_total_shares if new_total_shares > 0 else 0

        elif self.cost_basis_method in [CostBasisMethod.FIFO, CostBasisMethod.LIFO]:
            # For FIFO/LIFO, we track each lot separately via transactions
            position.shares += transaction.shares
            position.total_cost += transaction.total_with_commission
            position.cost_basis = position.total_cost / position.shares if position.shares > 0 else 0

        # Add transaction to position
        position.transactions.append(transaction)
        position.last_updated = datetime.now()

    def _update_position_sell(self, transaction: Transaction, cost_basis: float):
        """Update position for a sell transaction"""
        symbol = transaction.symbol
        position = self.positions[symbol]

        # Update shares and cost
        position.shares -= transaction.shares
        position.total_cost -= cost_basis

        # Recalculate cost basis
        if position.shares > 0:
            position.cost_basis = position.total_cost / position.shares
        else:
            position.cost_basis = 0
            position.total_cost = 0  # Ensure it's exactly zero

        # Add transaction to position
        position.transactions.append(transaction)
        position.last_updated = datetime.now()

        # Remove position if no shares left
        if position.shares <= 0.0001:  # Account for floating point errors
            logger.info(f"Position in {symbol} closed")
            del self.positions[symbol]

    def _calculate_cost_basis_for_sale(self, symbol: str, shares: float) -> float:
        """
        Calculate cost basis for shares being sold

        Args:
            symbol: Stock symbol
            shares: Number of shares being sold

        Returns:
            Cost basis for the shares being sold
        """
        position = self.positions[symbol]

        if self.cost_basis_method == CostBasisMethod.AVERAGE:
            # Average cost: simple calculation
            return position.cost_basis * shares

        elif self.cost_basis_method == CostBasisMethod.FIFO:
            # First In, First Out
            return self._calculate_fifo_cost_basis(position, shares)

        elif self.cost_basis_method == CostBasisMethod.LIFO:
            # Last In, First Out
            return self._calculate_lifo_cost_basis(position, shares)

        else:
            # Default to average
            return position.cost_basis * shares

    def _calculate_fifo_cost_basis(self, position: Position, shares_to_sell: float) -> float:
        """Calculate FIFO cost basis"""
        remaining_shares = shares_to_sell
        total_cost = 0.0

        # Get buy transactions in chronological order
        buy_transactions = [
            t for t in position.transactions
            if t.transaction_type == TransactionType.BUY
        ]
        buy_transactions.sort(key=lambda x: x.timestamp)

        for txn in buy_transactions:
            if remaining_shares <= 0:
                break

            # Calculate how many shares from this lot we're selling
            shares_from_lot = min(remaining_shares, txn.shares)
            cost_from_lot = (txn.total_with_commission / txn.shares) * shares_from_lot

            total_cost += cost_from_lot
            remaining_shares -= shares_from_lot

        return total_cost

    def _calculate_lifo_cost_basis(self, position: Position, shares_to_sell: float) -> float:
        """Calculate LIFO cost basis"""
        remaining_shares = shares_to_sell
        total_cost = 0.0

        # Get buy transactions in reverse chronological order
        buy_transactions = [
            t for t in position.transactions
            if t.transaction_type == TransactionType.BUY
        ]
        buy_transactions.sort(key=lambda x: x.timestamp, reverse=True)

        for txn in buy_transactions:
            if remaining_shares <= 0:
                break

            shares_from_lot = min(remaining_shares, txn.shares)
            cost_from_lot = (txn.total_with_commission / txn.shares) * shares_from_lot

            total_cost += cost_from_lot
            remaining_shares -= shares_from_lot

        return total_cost

    # ========================================================================
    # Portfolio Queries and Analytics
    # ========================================================================

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value

        Args:
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Total portfolio value including cash
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position.shares * current_prices[symbol]
            else:
                logger.warning(f"No price available for {symbol}, using cost basis")
                total_value += position.total_cost

        return total_value

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get comprehensive portfolio summary

        Args:
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Dictionary with portfolio metrics
        """
        total_market_value = 0.0
        total_cost_basis = 0.0
        positions_list = []

        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.cost_basis)
            pnl_metrics = position.calculate_unrealized_pnl(current_price)

            market_value = pnl_metrics['market_value']
            total_market_value += market_value
            total_cost_basis += position.total_cost

            positions_list.append({
                'symbol': symbol,
                'shares': position.shares,
                'cost_basis': position.cost_basis,
                'current_price': current_price,
                'market_value': market_value,
                'total_cost': position.total_cost,
                'unrealized_pnl': pnl_metrics['unrealized_pnl'],
                'unrealized_pnl_pct': pnl_metrics['unrealized_pnl_pct']
            })

        total_value = total_market_value + self.cash
        unrealized_pnl = total_market_value - total_cost_basis
        total_pnl = self.realized_pnl + unrealized_pnl

        # Calculate return on investment
        total_invested = self.initial_cash
        total_return_pct = ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0

        return {
            'portfolio_name': self.name,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_market_value,
            'total_cost_basis': total_cost_basis,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': total_pnl,
            'total_commissions': self.total_commissions,
            'initial_cash': self.initial_cash,
            'total_return_pct': total_return_pct,
            'num_positions': len(self.positions),
            'positions': positions_list,
            'currency': self.currency,
            'last_updated': self.last_updated.isoformat()
        }

    def get_allocation(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get portfolio allocation breakdown

        Args:
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            DataFrame with allocation by position
        """
        if not self.positions:
            return pd.DataFrame()

        total_value = self.get_portfolio_value(current_prices)

        allocations = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.cost_basis)
            market_value = position.shares * current_price
            allocation_pct = (market_value / total_value * 100) if total_value > 0 else 0

            allocations.append({
                'symbol': symbol,
                'shares': position.shares,
                'current_price': current_price,
                'market_value': market_value,
                'allocation_pct': allocation_pct
            })

        df = pd.DataFrame(allocations)
        df = df.sort_values('allocation_pct', ascending=False)

        return df

    def get_transaction_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get transaction history

        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            DataFrame with transaction history
        """
        if not self.transaction_history:
            return pd.DataFrame()

        transactions = self.transaction_history

        # Filter by symbol
        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]

        # Filter by date
        if start_date:
            transactions = [t for t in transactions if t.timestamp >= start_date]
        if end_date:
            transactions = [t for t in transactions if t.timestamp <= end_date]

        # Convert to DataFrame
        df = pd.DataFrame([t.to_dict() for t in transactions])

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        return df

    def calculate_performance_metrics(
        self,
        current_prices: Dict[str, float],
        benchmark_return: Optional[float] = None
    ) -> Dict:
        """
        Calculate portfolio performance metrics

        Args:
            current_prices: Current market prices
            benchmark_return: Benchmark return for comparison (optional)

        Returns:
            Dictionary with performance metrics
        """
        summary = self.get_portfolio_summary(current_prices)

        total_value = summary['total_value']
        total_invested = self.initial_cash
        total_return = total_value - total_invested
        total_return_pct = summary['total_return_pct']

        # Calculate time-based metrics
        days_invested = (datetime.now() - self.created_at).days
        years_invested = days_invested / 365.25

        # Annualized return
        if years_invested > 0:
            annualized_return = ((total_value / total_invested) ** (1 / years_invested) - 1) * 100
        else:
            annualized_return = 0.0

        # Win rate (percentage of profitable positions)
        profitable_positions = sum(1 for p in summary['positions'] if p['unrealized_pnl'] > 0)
        total_positions = len(summary['positions'])
        win_rate = (profitable_positions / total_positions * 100) if total_positions > 0 else 0

        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return,
            'win_rate': win_rate,
            'profitable_positions': profitable_positions,
            'total_positions': total_positions,
            'days_invested': days_invested,
            'years_invested': years_invested,
            'total_commissions': self.total_commissions,
            'commission_pct_of_invested': (self.total_commissions / total_invested * 100) if total_invested > 0 else 0
        }

        # Add benchmark comparison if provided
        if benchmark_return is not None:
            metrics['benchmark_return'] = benchmark_return
            metrics['excess_return'] = total_return_pct - benchmark_return
            metrics['alpha'] = metrics['excess_return']

        return metrics

    # ========================================================================
    # Portfolio I/O Operations
    # ========================================================================

    def to_dict(self) -> Dict:
        """
        Export portfolio to dictionary

        Returns:
            Dictionary representation of portfolio
        """
        return {
            'name': self.name,
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'cost_basis_method': self.cost_basis_method.value,
            'currency': self.currency,
            'realized_pnl': self.realized_pnl,
            'total_commissions': self.total_commissions,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'transaction_history': [t.to_dict() for t in self.transaction_history]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioManager':
        """
        Create portfolio from dictionary

        Args:
            data: Dictionary representation

        Returns:
            PortfolioManager instance
        """
        portfolio = cls(
            name=data['name'],
            initial_cash=data['initial_cash'],
            cost_basis_method=CostBasisMethod(data['cost_basis_method']),
            currency=data['currency']
        )

        portfolio.cash = data['cash']
        portfolio.realized_pnl = data['realized_pnl']
        portfolio.total_commissions = data['total_commissions']
        portfolio.created_at = pd.to_datetime(data['created_at'])
        portfolio.last_updated = pd.to_datetime(data['last_updated'])

        # Restore positions
        portfolio.positions = {
            symbol: Position.from_dict(pos_data)
            for symbol, pos_data in data['positions'].items()
        }

        # Restore transaction history
        portfolio.transaction_history = [
            Transaction.from_dict(t) for t in data['transaction_history']
        ]

        return portfolio

    def save_to_json(self, filepath: str):
        """
        Save portfolio to JSON file

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Portfolio saved to {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str) -> 'PortfolioManager':
        """
        Load portfolio from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            PortfolioManager instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        portfolio = cls.from_dict(data)
        logger.info(f"Portfolio loaded from {filepath}")

        return portfolio

    def save_to_pickle(self, filepath: str):
        """
        Save portfolio to pickle file

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Portfolio saved to {filepath}")

    @classmethod
    def load_from_pickle(cls, filepath: str) -> 'PortfolioManager':
        """
        Load portfolio from pickle file

        Args:
            filepath: Path to pickle file

        Returns:
            PortfolioManager instance
        """
        with open(filepath, 'rb') as f:
            portfolio = pickle.load(f)

        logger.info(f"Portfolio loaded from {filepath}")

        return portfolio

    def export_to_csv(self, filepath: str, current_prices: Dict[str, float]):
        """
        Export portfolio positions to CSV

        Args:
            filepath: Path to save file
            current_prices: Current market prices
        """
        summary = self.get_portfolio_summary(current_prices)
        df = pd.DataFrame(summary['positions'])

        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Portfolio positions exported to {filepath}")
        else:
            logger.warning("No positions to export")

    def export_transactions_to_csv(self, filepath: str):
        """
        Export transaction history to CSV

        Args:
            filepath: Path to save file
        """
        df = self.get_transaction_history()

        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Transaction history exported to {filepath}")
        else:
            logger.warning("No transactions to export")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"PortfolioManager(name='{self.name}', "
            f"positions={len(self.positions)}, "
            f"cash={self.cash:.2f} {self.currency})"
        )

    def __str__(self) -> str:
        """Detailed string representation"""
        lines = [
            f"Portfolio: {self.name}",
            f"Cash: {self.cash:.2f} {self.currency}",
            f"Positions: {len(self.positions)}",
            f"Realized P&L: {self.realized_pnl:.2f} {self.currency}",
            f"Total Commissions: {self.total_commissions:.2f} {self.currency}"
        ]

        if self.positions:
            lines.append("\nPositions:")
            for symbol, position in self.positions.items():
                lines.append(
                    f"  {symbol}: {position.shares:.2f} shares @ {position.cost_basis:.2f}"
                )

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_portfolio(
    name: str = "My Portfolio",
    initial_cash: float = 100000.0,
    cost_basis_method: str = "AVERAGE",
    currency: str = "TRY"
) -> PortfolioManager:
    """
    Convenience function to create a portfolio

    Args:
        name: Portfolio name
        initial_cash: Initial cash balance
        cost_basis_method: Cost basis calculation method (FIFO, LIFO, AVERAGE)
        currency: Portfolio currency

    Returns:
        PortfolioManager instance
    """
    method = CostBasisMethod(cost_basis_method)
    return PortfolioManager(
        name=name,
        initial_cash=initial_cash,
        cost_basis_method=method,
        currency=currency
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage of PortfolioManager"""

    print("=" * 80)
    print("BIST Portfolio Manager - Example Usage")
    print("=" * 80)

    # Create a portfolio
    print("\n1. Creating Portfolio")
    print("-" * 80)
    portfolio = PortfolioManager(
        name="BIST Trading Portfolio",
        initial_cash=100000.0,
        cost_basis_method=CostBasisMethod.AVERAGE,
        currency="TRY"
    )
    print(portfolio)

    # Execute some buy transactions
    print("\n2. Executing Buy Transactions")
    print("-" * 80)

    portfolio.buy(symbol="THYAO", shares=100, price=250.0, commission=10.0, notes="Initial purchase")
    portfolio.buy(symbol="GARAN", shares=200, price=85.0, commission=15.0, notes="Initial purchase")
    portfolio.buy(symbol="THYAO", shares=50, price=260.0, commission=5.0, notes="Additional purchase")

    print(f"\nCash remaining: {portfolio.cash:.2f} TRY")
    print(f"Positions: {len(portfolio.positions)}")

    # View positions
    print("\n3. Current Positions")
    print("-" * 80)
    for symbol, position in portfolio.get_all_positions().items():
        print(f"{symbol}: {position.shares} shares @ {position.cost_basis:.2f} TRY (total cost: {position.total_cost:.2f})")

    # Execute a sell transaction
    print("\n4. Executing Sell Transaction")
    print("-" * 80)

    portfolio.sell(symbol="THYAO", shares=50, price=270.0, commission=7.0, notes="Partial sale")

    print(f"Realized P&L: {portfolio.realized_pnl:.2f} TRY")
    print(f"Cash: {portfolio.cash:.2f} TRY")

    # Get portfolio summary with current prices
    print("\n5. Portfolio Summary")
    print("-" * 80)

    current_prices = {
        "THYAO": 265.0,
        "GARAN": 90.0
    }

    summary = portfolio.get_portfolio_summary(current_prices)

    print(f"Total Value: {summary['total_value']:.2f} TRY")
    print(f"Cash: {summary['cash']:.2f} TRY")
    print(f"Positions Value: {summary['positions_value']:.2f} TRY")
    print(f"Unrealized P&L: {summary['unrealized_pnl']:.2f} TRY")
    print(f"Realized P&L: {summary['realized_pnl']:.2f} TRY")
    print(f"Total P&L: {summary['total_pnl']:.2f} TRY")
    print(f"Total Return: {summary['total_return_pct']:.2f}%")

    print("\nPositions Detail:")
    for pos in summary['positions']:
        print(f"  {pos['symbol']}: {pos['shares']:.2f} shares")
        print(f"    Cost: {pos['total_cost']:.2f} TRY, Market Value: {pos['market_value']:.2f} TRY")
        print(f"    P&L: {pos['unrealized_pnl']:.2f} TRY ({pos['unrealized_pnl_pct']:.2f}%)")

    # Get allocation
    print("\n6. Portfolio Allocation")
    print("-" * 80)

    allocation = portfolio.get_allocation(current_prices)
    print(allocation.to_string(index=False))

    # Transaction history
    print("\n7. Transaction History")
    print("-" * 80)

    history = portfolio.get_transaction_history()
    print(history[['timestamp', 'symbol', 'transaction_type', 'shares', 'price', 'commission']].to_string(index=False))

    # Performance metrics
    print("\n8. Performance Metrics")
    print("-" * 80)

    metrics = portfolio.calculate_performance_metrics(current_prices, benchmark_return=5.0)

    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Days Invested: {metrics['days_invested']}")
    print(f"Total Commissions: {metrics['total_commissions']:.2f} TRY ({metrics['commission_pct_of_invested']:.2f}%)")
    if 'excess_return' in metrics:
        print(f"Excess Return vs Benchmark: {metrics['excess_return']:.2f}%")

    # Save and load portfolio
    print("\n9. Saving and Loading Portfolio")
    print("-" * 80)

    # Save to JSON
    portfolio.save_to_json("portfolio.json")

    # Load from JSON
    loaded_portfolio = PortfolioManager.load_from_json("portfolio.json")
    print(f"Loaded portfolio: {loaded_portfolio.name}")
    print(f"Positions: {len(loaded_portfolio.positions)}")
    print(f"Cash: {loaded_portfolio.cash:.2f} TRY")

    # Export to CSV
    portfolio.export_to_csv("positions.csv", current_prices)
    portfolio.export_transactions_to_csv("transactions.csv")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
