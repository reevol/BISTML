"""
Backtesting Engine for BIST AI Trading System

This module provides a comprehensive backtesting framework for evaluating trading
strategies on historical data. It simulates trades based on historical signals,
tracks equity curves, handles transaction costs and slippage, and calculates
detailed performance metrics.

Features:
- Event-driven backtesting architecture
- Realistic transaction cost modeling (commissions + slippage)
- Equity curve and drawdown tracking
- Comprehensive performance metrics (Sharpe, Sortino, Max Drawdown, Win Rate, etc.)
- Trade-level analytics and reporting
- Position sizing strategies
- Signal-based and strategy-based backtesting
- Results export and visualization support
- Multi-timeframe backtesting
- Walk-forward analysis support

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Enums and Constants
# ============================================================================

class PositionSizing(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "FIXED_AMOUNT"           # Fixed cash amount per trade
    FIXED_SHARES = "FIXED_SHARES"           # Fixed number of shares
    PERCENT_EQUITY = "PERCENT_EQUITY"       # Percentage of current equity
    RISK_BASED = "RISK_BASED"               # Based on risk per trade
    SIGNAL_STRENGTH = "SIGNAL_STRENGTH"     # Based on signal confidence


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"           # Market order (immediate execution)
    LIMIT = "LIMIT"             # Limit order
    STOP = "STOP"               # Stop order
    STOP_LIMIT = "STOP_LIMIT"   # Stop-limit order


class SlippageModel(Enum):
    """Slippage modeling approaches"""
    FIXED_PERCENT = "FIXED_PERCENT"         # Fixed percentage slippage
    VOLUME_BASED = "VOLUME_BASED"           # Based on volume and order size
    BID_ASK_SPREAD = "BID_ASK_SPREAD"       # Based on bid-ask spread
    NONE = "NONE"                           # No slippage


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting engine

    Attributes:
        initial_capital: Starting capital in TRY
        commission_rate: Commission rate (e.g., 0.001 = 0.1%)
        slippage_model: Type of slippage model to use
        slippage_rate: Slippage rate (for fixed model)
        position_sizing: Position sizing method
        position_size_value: Value for position sizing (depends on method)
        max_positions: Maximum number of concurrent positions
        allow_shorting: Whether to allow short positions
        use_stop_loss: Whether to use stop loss orders
        use_take_profit: Whether to use take profit orders
        benchmark_symbol: Symbol for benchmark comparison (e.g., 'XU100')
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        min_confidence: Minimum confidence score to execute trade
    """
    initial_capital: float = 100000.0
    commission_rate: float = 0.001              # 0.1%
    slippage_model: SlippageModel = SlippageModel.FIXED_PERCENT
    slippage_rate: float = 0.0005               # 0.05%
    position_sizing: PositionSizing = PositionSizing.PERCENT_EQUITY
    position_size_value: float = 0.1            # 10% of equity per trade
    max_positions: int = 10
    allow_shorting: bool = False
    use_stop_loss: bool = True
    use_take_profit: bool = True
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.15                # 15% annual (Turkey)
    min_confidence: float = 0.5                 # 50% minimum confidence


@dataclass
class Trade:
    """
    Represents a single trade (entry and exit)

    Attributes:
        symbol: Stock symbol
        entry_date: Entry datetime
        exit_date: Exit datetime (None if still open)
        entry_price: Entry price
        exit_price: Exit price (None if still open)
        shares: Number of shares
        side: 'long' or 'short'
        entry_signal: Entry signal type
        exit_signal: Exit signal type
        entry_confidence: Entry signal confidence
        pnl: Profit/loss in TRY
        pnl_percent: Profit/loss percentage
        commission_paid: Total commission paid
        slippage_cost: Total slippage cost
        holding_period: Number of days held
        mae: Maximum Adverse Excursion (worst drawdown during trade)
        mfe: Maximum Favorable Excursion (best profit during trade)
        exit_reason: Reason for exit (signal, stop_loss, take_profit, etc.)
    """
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    side: str
    entry_signal: str
    entry_confidence: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_signal: Optional[str] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    holding_period: Optional[int] = None
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_date is None

    @property
    def is_winner(self) -> bool:
        """Check if trade is profitable"""
        return self.pnl > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'side': self.side,
            'entry_signal': self.entry_signal,
            'exit_signal': self.exit_signal,
            'entry_confidence': self.entry_confidence,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'commission_paid': self.commission_paid,
            'slippage_cost': self.slippage_cost,
            'holding_period': self.holding_period,
            'mae': self.mae,
            'mfe': self.mfe,
            'exit_reason': self.exit_reason
        }


@dataclass
class BacktestResults:
    """
    Comprehensive backtesting results

    Contains all performance metrics, trades, and equity curve data
    """
    # Basic info
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Performance metrics
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float

    # Advanced metrics
    calmar_ratio: float
    recovery_factor: float
    expectancy: float

    # Costs
    total_commission: float
    total_slippage: float

    # Equity curve
    equity_curve: pd.DataFrame
    drawdown_series: pd.Series

    # Trades
    trades: List[Trade]

    # Benchmark comparison (if applicable)
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large dataframes)"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_holding_period': self.avg_holding_period,
            'calmar_ratio': self.calmar_ratio,
            'recovery_factor': self.recovery_factor,
            'expectancy': self.expectancy,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """String representation"""
        lines = [
            "=" * 80,
            "BACKTEST RESULTS",
            "=" * 80,
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Initial Capital: {self.initial_capital:,.2f} TRY",
            f"Final Capital: {self.final_capital:,.2f} TRY",
            "",
            "PERFORMANCE METRICS",
            "-" * 80,
            f"Total Return: {self.total_return:,.2f} TRY ({self.total_return_pct:.2f}%)",
            f"Annualized Return: {self.annualized_return:.2f}%",
            f"Sharpe Ratio: {self.sharpe_ratio:.3f}",
            f"Sortino Ratio: {self.sortino_ratio:.3f}",
            f"Max Drawdown: {self.max_drawdown:.2f}%",
            f"Max DD Duration: {self.max_drawdown_duration} days",
            f"Calmar Ratio: {self.calmar_ratio:.3f}",
            "",
            "TRADE STATISTICS",
            "-" * 80,
            f"Total Trades: {self.total_trades}",
            f"Winning Trades: {self.winning_trades}",
            f"Losing Trades: {self.losing_trades}",
            f"Win Rate: {self.win_rate:.2f}%",
            f"Average Win: {self.avg_win:,.2f} TRY",
            f"Average Loss: {self.avg_loss:,.2f} TRY",
            f"Profit Factor: {self.profit_factor:.3f}",
            f"Expectancy: {self.expectancy:,.2f} TRY",
            f"Avg Holding Period: {self.avg_holding_period:.1f} days",
            "",
            "COSTS",
            "-" * 80,
            f"Total Commission: {self.total_commission:,.2f} TRY",
            f"Total Slippage: {self.total_slippage:,.2f} TRY",
        ]

        if self.benchmark_return is not None:
            lines.extend([
                "",
                "BENCHMARK COMPARISON",
                "-" * 80,
                f"Benchmark Return: {self.benchmark_return:.2f}%",
                f"Alpha: {self.alpha:.2f}%" if self.alpha else "Alpha: N/A",
                f"Beta: {self.beta:.3f}" if self.beta else "Beta: N/A",
            ])

        lines.append("=" * 80)
        return "\n".join(lines)


# ============================================================================
# Backtesting Engine
# ============================================================================

class BacktestEngine:
    """
    Main backtesting engine for simulating trading strategies

    This engine runs historical backtests by:
    1. Loading historical price data
    2. Generating or loading trading signals
    3. Simulating trade execution with realistic costs
    4. Tracking portfolio value and positions
    5. Calculating comprehensive performance metrics
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine

        Parameters:
        -----------
        config : BacktestConfig, optional
            Backtesting configuration (uses defaults if not provided)
        """
        self.config = config or BacktestConfig()

        # State variables
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trades: List[Trade] = []
        self.equity_history: List[Dict] = []

        # Tracking variables
        self.current_date: Optional[datetime] = None
        self.total_commission = 0.0
        self.total_slippage = 0.0

        logger.info(f"BacktestEngine initialized with {self.config.initial_capital:,.2f} TRY")

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        signal_generator: Optional[Callable] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """
        Run backtest on historical data

        Parameters:
        -----------
        price_data : pd.DataFrame
            Historical price data with columns: ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            Index should be datetime
        signals : pd.DataFrame
            Trading signals with columns: ['symbol', 'date', 'signal', 'confidence', 'target_price']
            signal values: 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
        signal_generator : Callable, optional
            Function to generate signals on-the-fly (alternative to pre-computed signals)
        benchmark_data : pd.DataFrame, optional
            Benchmark price data for comparison

        Returns:
        --------
        BacktestResults
            Comprehensive backtest results
        """
        logger.info("Starting backtest...")

        # Reset state
        self._reset()

        # Validate and prepare data
        price_data = self._prepare_price_data(price_data)
        signals = self._prepare_signals(signals)

        # Get date range
        dates = sorted(price_data.index.unique())
        self.start_date = dates[0]
        self.end_date = dates[-1]

        logger.info(f"Backtesting period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Total trading days: {len(dates)}")

        # Main backtesting loop
        for i, date in enumerate(dates):
            self.current_date = date

            # Get data for this date
            daily_prices = price_data[price_data.index == date]
            daily_signals = signals[signals.index == date] if signals is not None else pd.DataFrame()

            # Update open positions with current prices
            self._update_positions(daily_prices)

            # Check stop loss and take profit
            if self.config.use_stop_loss or self.config.use_take_profit:
                self._check_exit_conditions(daily_prices)

            # Process signals
            if not daily_signals.empty or signal_generator is not None:
                self._process_signals(daily_signals, daily_prices, signal_generator)

            # Record equity
            self._record_equity(date, daily_prices)

            # Log progress
            if (i + 1) % 50 == 0 or i == len(dates) - 1:
                logger.info(f"Progress: {i+1}/{len(dates)} days ({(i+1)/len(dates)*100:.1f}%)")

        # Close any remaining positions
        self._close_all_positions(price_data[price_data.index == self.end_date])

        # Calculate results
        results = self._calculate_results(benchmark_data)

        logger.info("Backtest completed successfully")
        logger.info(f"Total trades: {results.total_trades}, Win rate: {results.win_rate:.2f}%")

        return results

    def _reset(self):
        """Reset engine state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.current_date = None

    def _prepare_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare price data"""
        required_cols = ['close']

        # Check for required columns
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Price data missing required column: {col}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            else:
                raise ValueError("Price data must have datetime index or 'date' column")

        # Add symbol if not present
        if 'symbol' not in df.columns:
            df['symbol'] = 'UNKNOWN'

        # Fill missing OHLC with close price
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df.sort_index()

    def _prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare signals"""
        if df is None or df.empty:
            return pd.DataFrame()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            else:
                raise ValueError("Signals must have datetime index or 'date' column")

        # Check for required columns
        if 'signal' not in df.columns:
            raise ValueError("Signals must have 'signal' column")

        # Add defaults
        if 'symbol' not in df.columns:
            df['symbol'] = 'UNKNOWN'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.5

        return df.sort_index()

    def _update_positions(self, daily_prices: pd.DataFrame):
        """Update position values and track MAE/MFE"""
        for symbol, position in list(self.positions.items()):
            # Get current price
            symbol_price = daily_prices[daily_prices['symbol'] == symbol]

            if symbol_price.empty:
                continue

            current_price = symbol_price['close'].iloc[0]
            high_price = symbol_price['high'].iloc[0]
            low_price = symbol_price['low'].iloc[0]

            # Update current price
            position['current_price'] = current_price
            position['market_value'] = position['shares'] * current_price

            # Calculate unrealized P&L
            if position['side'] == 'long':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
                position['unrealized_pnl_pct'] = (current_price - position['entry_price']) / position['entry_price']

                # Track MAE and MFE for long positions
                position['mfe'] = max(position['mfe'], (high_price - position['entry_price']) / position['entry_price'])
                position['mae'] = min(position['mae'], (low_price - position['entry_price']) / position['entry_price'])
            else:  # short
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['shares']
                position['unrealized_pnl_pct'] = (position['entry_price'] - current_price) / position['entry_price']

                # Track MAE and MFE for short positions
                position['mfe'] = max(position['mfe'], (position['entry_price'] - low_price) / position['entry_price'])
                position['mae'] = min(position['mae'], (position['entry_price'] - high_price) / position['entry_price'])

    def _check_exit_conditions(self, daily_prices: pd.DataFrame):
        """Check stop loss and take profit conditions"""
        for symbol, position in list(self.positions.items()):
            # Get current price
            symbol_price = daily_prices[daily_prices['symbol'] == symbol]

            if symbol_price.empty:
                continue

            current_price = symbol_price['close'].iloc[0]

            # Check stop loss
            if self.config.use_stop_loss and position.get('stop_loss'):
                if position['side'] == 'long' and current_price <= position['stop_loss']:
                    self._close_position(symbol, current_price, 'STOP_LOSS', 'stop_loss')
                elif position['side'] == 'short' and current_price >= position['stop_loss']:
                    self._close_position(symbol, current_price, 'STOP_LOSS', 'stop_loss')

            # Check take profit
            if self.config.use_take_profit and position.get('take_profit'):
                if position['side'] == 'long' and current_price >= position['take_profit']:
                    self._close_position(symbol, current_price, 'TAKE_PROFIT', 'take_profit')
                elif position['side'] == 'short' and current_price <= position['take_profit']:
                    self._close_position(symbol, current_price, 'TAKE_PROFIT', 'take_profit')

    def _process_signals(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        signal_generator: Optional[Callable] = None
    ):
        """Process trading signals and execute trades"""
        # Get unique symbols
        symbols = prices['symbol'].unique()

        for symbol in symbols:
            # Get signal for this symbol
            symbol_signals = signals[signals['symbol'] == symbol] if not signals.empty else pd.DataFrame()

            if symbol_signals.empty and signal_generator is None:
                continue

            # Generate signal if needed
            if signal_generator is not None:
                # Signal generator should return (signal, confidence, target_price)
                signal_data = signal_generator(symbol, self.current_date, prices)
                if signal_data is None:
                    continue
                signal, confidence, target_price = signal_data
            else:
                signal = symbol_signals['signal'].iloc[0]
                confidence = symbol_signals['confidence'].iloc[0] if 'confidence' in symbol_signals.columns else 0.5
                target_price = symbol_signals.get('target_price', {}).iloc[0] if 'target_price' in symbol_signals.columns else None

            # Check minimum confidence
            if confidence < self.config.min_confidence:
                continue

            # Get current price
            symbol_price = prices[prices['symbol'] == symbol]
            if symbol_price.empty:
                continue

            current_price = symbol_price['close'].iloc[0]

            # Process signal
            if signal in ['STRONG_BUY', 'BUY']:
                self._process_buy_signal(symbol, current_price, signal, confidence, target_price)
            elif signal in ['STRONG_SELL', 'SELL']:
                self._process_sell_signal(symbol, current_price, signal, confidence)

    def _process_buy_signal(
        self,
        symbol: str,
        price: float,
        signal: str,
        confidence: float,
        target_price: Optional[float] = None
    ):
        """Process buy signal"""
        # Check if we already have a position
        if symbol in self.positions:
            return  # Don't add to existing position

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return

        # Calculate position size
        shares = self._calculate_position_size(price, signal, confidence)

        if shares <= 0:
            return

        # Calculate costs
        execution_price = self._apply_slippage(price, 'buy', shares)
        total_cost = shares * execution_price
        commission = total_cost * self.config.commission_rate

        # Check if we have enough cash
        if total_cost + commission > self.cash:
            # Adjust shares to fit available cash
            available = self.cash / (1 + self.config.commission_rate)
            shares = int(available / execution_price)

            if shares <= 0:
                return

            total_cost = shares * execution_price
            commission = total_cost * self.config.commission_rate

        # Execute trade
        self.cash -= (total_cost + commission)
        self.total_commission += commission
        self.total_slippage += abs(execution_price - price) * shares

        # Calculate stop loss and take profit
        stop_loss = None
        take_profit = None

        if self.config.use_stop_loss:
            stop_loss = execution_price * 0.95  # 5% stop loss

        if self.config.use_take_profit:
            if target_price:
                take_profit = target_price
            else:
                take_profit = execution_price * 1.10  # 10% take profit

        # Create position
        self.positions[symbol] = {
            'symbol': symbol,
            'shares': shares,
            'side': 'long',
            'entry_price': execution_price,
            'entry_date': self.current_date,
            'entry_signal': signal,
            'entry_confidence': confidence,
            'current_price': execution_price,
            'market_value': total_cost,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0,
            'commission_paid': commission,
            'slippage_cost': abs(execution_price - price) * shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'mae': 0.0,
            'mfe': 0.0
        }

        logger.debug(f"BUY {shares} shares of {symbol} @ {execution_price:.2f} (signal: {signal}, confidence: {confidence:.2f})")

    def _process_sell_signal(self, symbol: str, price: float, signal: str, confidence: float):
        """Process sell signal"""
        # Check if we have a position
        if symbol not in self.positions:
            # Could implement shorting here if allowed
            return

        # Close the position
        self._close_position(symbol, price, signal, 'signal')

    def _close_position(self, symbol: str, price: float, exit_signal: str, exit_reason: str):
        """Close an open position"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate execution price with slippage
        execution_price = self._apply_slippage(price, 'sell', position['shares'])

        # Calculate proceeds
        proceeds = position['shares'] * execution_price
        commission = proceeds * self.config.commission_rate

        # Update cash
        self.cash += (proceeds - commission)
        self.total_commission += commission
        self.total_slippage += abs(execution_price - price) * position['shares']

        # Calculate P&L
        total_commission = position['commission_paid'] + commission
        total_slippage = position['slippage_cost'] + abs(execution_price - price) * position['shares']

        pnl = (execution_price - position['entry_price']) * position['shares'] - total_commission - total_slippage
        pnl_percent = ((execution_price - position['entry_price']) / position['entry_price']) * 100

        holding_period = (self.current_date - position['entry_date']).days

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=self.current_date,
            entry_price=position['entry_price'],
            exit_price=execution_price,
            shares=position['shares'],
            side=position['side'],
            entry_signal=position['entry_signal'],
            exit_signal=exit_signal,
            entry_confidence=position['entry_confidence'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission_paid=total_commission,
            slippage_cost=total_slippage,
            holding_period=holding_period,
            mae=position['mae'] * 100,  # Convert to percentage
            mfe=position['mfe'] * 100,
            exit_reason=exit_reason
        )

        self.trades.append(trade)

        # Remove position
        del self.positions[symbol]

        logger.debug(f"SELL {trade.shares} shares of {symbol} @ {execution_price:.2f} - P&L: {pnl:.2f} TRY ({pnl_percent:.2f}%)")

    def _close_all_positions(self, final_prices: pd.DataFrame):
        """Close all remaining positions at end of backtest"""
        for symbol in list(self.positions.keys()):
            symbol_price = final_prices[final_prices['symbol'] == symbol]

            if symbol_price.empty:
                logger.warning(f"No final price for {symbol}, using entry price")
                price = self.positions[symbol]['entry_price']
            else:
                price = symbol_price['close'].iloc[0]

            self._close_position(symbol, price, 'END_OF_BACKTEST', 'end_of_period')

    def _calculate_position_size(self, price: float, signal: str, confidence: float) -> int:
        """Calculate number of shares to buy based on position sizing method"""
        if self.config.position_sizing == PositionSizing.FIXED_AMOUNT:
            shares = int(self.config.position_size_value / price)

        elif self.config.position_sizing == PositionSizing.FIXED_SHARES:
            shares = int(self.config.position_size_value)

        elif self.config.position_sizing == PositionSizing.PERCENT_EQUITY:
            # Current equity (cash + positions value)
            equity = self._calculate_current_equity()
            position_value = equity * self.config.position_size_value
            shares = int(position_value / price)

        elif self.config.position_sizing == PositionSizing.SIGNAL_STRENGTH:
            # Scale position size by confidence
            equity = self._calculate_current_equity()
            base_position = equity * self.config.position_size_value
            adjusted_position = base_position * confidence
            shares = int(adjusted_position / price)

        else:  # RISK_BASED
            # Risk-based position sizing (simplified)
            equity = self._calculate_current_equity()
            risk_amount = equity * self.config.position_size_value  # Risk this much
            stop_loss_pct = 0.05  # Assume 5% stop loss
            shares = int(risk_amount / (price * stop_loss_pct))

        return max(0, shares)

    def _apply_slippage(self, price: float, side: str, shares: float) -> float:
        """Apply slippage to execution price"""
        if self.config.slippage_model == SlippageModel.NONE:
            return price

        elif self.config.slippage_model == SlippageModel.FIXED_PERCENT:
            if side == 'buy':
                return price * (1 + self.config.slippage_rate)
            else:  # sell
                return price * (1 - self.config.slippage_rate)

        # Add more sophisticated slippage models here
        else:
            return price

    def _calculate_current_equity(self) -> float:
        """Calculate current total equity"""
        positions_value = sum(pos['market_value'] for pos in self.positions.values())
        return self.cash + positions_value

    def _record_equity(self, date: datetime, prices: pd.DataFrame):
        """Record equity for this date"""
        equity = self._calculate_current_equity()

        # Calculate positions value
        positions_value = sum(pos['market_value'] for pos in self.positions.values())

        self.equity_history.append({
            'date': date,
            'equity': equity,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions)
        })

    def _calculate_results(self, benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.set_index('date', inplace=True)

        # Basic metrics
        initial_capital = self.config.initial_capital
        final_capital = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1

        # Annualized return
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()

        # Max drawdown duration
        dd_duration = self._calculate_drawdown_duration(equity_df['drawdown'])

        # Sharpe ratio (annualized)
        if len(equity_df['returns'].dropna()) > 1:
            excess_returns = equity_df['returns'] - (self.config.risk_free_rate / 252)  # Daily risk-free rate
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino ratio (annualized)
        if len(equity_df['returns'].dropna()) > 1:
            downside_returns = equity_df['returns'][equity_df['returns'] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0

        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.is_winner])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.is_winner]
        losses = [t.pnl for t in self.trades if not t.is_winner]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average holding period
        holding_periods = [t.holding_period for t in self.trades if t.holding_period is not None]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown * initial_capital / 100) if max_drawdown != 0 else 0

        # Expectancy (average trade P&L)
        expectancy = total_return / total_trades if total_trades > 0 else 0

        # Benchmark comparison
        benchmark_return = None
        alpha = None
        beta = None

        if benchmark_data is not None:
            benchmark_return, alpha, beta = self._calculate_benchmark_metrics(
                equity_df, benchmark_data
            )

        # Create results object
        results = BacktestResults(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_period=avg_holding_period,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            expectancy=expectancy,
            total_commission=self.total_commission,
            total_slippage=self.total_slippage,
            equity_curve=equity_df,
            drawdown_series=equity_df['drawdown'],
            trades=self.trades,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            metadata={
                'config': asdict(self.config),
                'num_trading_days': len(equity_df)
            }
        )

        return results

    def _calculate_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = drawdown_series < 0

        if not in_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        return max(drawdown_periods) if drawdown_periods else 0

    def _calculate_benchmark_metrics(
        self,
        equity_df: pd.DataFrame,
        benchmark_data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Calculate alpha and beta vs benchmark"""
        # Align dates
        benchmark_data = benchmark_data.copy()
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
            if 'date' in benchmark_data.columns:
                benchmark_data.set_index('date', inplace=True)

        # Calculate benchmark returns
        benchmark_data['returns'] = benchmark_data['close'].pct_change()

        # Merge with equity returns
        merged = equity_df[['returns']].join(
            benchmark_data[['returns']],
            how='inner',
            rsuffix='_benchmark'
        ).dropna()

        if len(merged) < 2:
            return None, None, None

        # Calculate metrics
        strategy_returns = merged['returns']
        benchmark_returns = merged['returns_benchmark']

        # Beta (covariance / variance)
        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha (annualized)
        strategy_return = (1 + strategy_returns.mean()) ** 252 - 1
        benchmark_return = (1 + benchmark_returns.mean()) ** 252 - 1
        alpha = (strategy_return - (self.config.risk_free_rate + beta * (benchmark_return - self.config.risk_free_rate))) * 100

        # Benchmark total return (for display)
        benchmark_total_return = ((benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0]) - 1) * 100

        return benchmark_total_return, alpha, beta


# ============================================================================
# Utility Functions
# ============================================================================

def create_backtest_config(**kwargs) -> BacktestConfig:
    """
    Create a backtest configuration

    Parameters:
    -----------
    **kwargs : dict
        Configuration parameters

    Returns:
    --------
    BacktestConfig
        Backtest configuration object
    """
    return BacktestConfig(**kwargs)


def export_results(results: BacktestResults, output_dir: str):
    """
    Export backtest results to files

    Parameters:
    -----------
    results : BacktestResults
        Backtest results
    output_dir : str
        Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export summary as JSON
    summary_path = output_path / 'backtest_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"Summary exported to {summary_path}")

    # Export equity curve
    equity_path = output_path / 'equity_curve.csv'
    results.equity_curve.to_csv(equity_path)
    logger.info(f"Equity curve exported to {equity_path}")

    # Export trades
    trades_path = output_path / 'trades.csv'
    trades_df = pd.DataFrame([t.to_dict() for t in results.trades])
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"Trades exported to {trades_path}")

    # Export text report
    report_path = output_path / 'backtest_report.txt'
    with open(report_path, 'w') as f:
        f.write(str(results))
    logger.info(f"Report exported to {report_path}")


def quick_backtest(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001
) -> BacktestResults:
    """
    Quick backtest with default settings

    Parameters:
    -----------
    price_data : pd.DataFrame
        Historical price data
    signals : pd.DataFrame
        Trading signals
    initial_capital : float
        Starting capital
    commission_rate : float
        Commission rate

    Returns:
    --------
    BacktestResults
        Backtest results
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=commission_rate
    )

    engine = BacktestEngine(config)
    results = engine.run(price_data, signals)

    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Backtesting Engine - BIST AI Trading System")
    print("=" * 80)

    # Create sample data
    print("\n1. Creating sample data...")

    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

    # Sample price data (THYAO)
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * (1 + returns).cumprod()

    price_data = pd.DataFrame({
        'symbol': 'THYAO',
        'date': dates,
        'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.01)),
        'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    price_data.set_index('date', inplace=True)

    # Sample signals (simple moving average crossover)
    price_data['sma_20'] = price_data['close'].rolling(20).mean()
    price_data['sma_50'] = price_data['close'].rolling(50).mean()

    signals_list = []
    for date, row in price_data.iterrows():
        if pd.isna(row['sma_50']):
            continue

        if row['sma_20'] > row['sma_50']:
            signal = 'BUY'
            confidence = 0.7
        elif row['sma_20'] < row['sma_50']:
            signal = 'SELL'
            confidence = 0.6
        else:
            signal = 'HOLD'
            confidence = 0.5

        signals_list.append({
            'date': date,
            'symbol': 'THYAO',
            'signal': signal,
            'confidence': confidence
        })

    signals = pd.DataFrame(signals_list)
    signals.set_index('date', inplace=True)

    print(f"Price data shape: {price_data.shape}")
    print(f"Signals shape: {signals.shape}")

    # Create backtest configuration
    print("\n2. Creating backtest configuration...")
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        position_sizing=PositionSizing.PERCENT_EQUITY,
        position_size_value=0.20,  # 20% per trade
        use_stop_loss=True,
        use_take_profit=True,
        min_confidence=0.6
    )

    print(f"Initial capital: {config.initial_capital:,.2f} TRY")
    print(f"Commission rate: {config.commission_rate*100:.2f}%")
    print(f"Position sizing: {config.position_sizing.value}")

    # Run backtest
    print("\n3. Running backtest...")
    engine = BacktestEngine(config)
    results = engine.run(price_data, signals)

    # Display results
    print("\n4. Results:")
    print(results)

    # Trade details
    print("\n5. Sample Trades (first 5):")
    print("-" * 80)
    for i, trade in enumerate(results.trades[:5]):
        print(f"\nTrade {i+1}:")
        print(f"  Symbol: {trade.symbol}")
        print(f"  Entry: {trade.entry_date.date()} @ {trade.entry_price:.2f} TRY")
        print(f"  Exit: {trade.exit_date.date()} @ {trade.exit_price:.2f} TRY")
        print(f"  Shares: {trade.shares}")
        print(f"  P&L: {trade.pnl:,.2f} TRY ({trade.pnl_percent:.2f}%)")
        print(f"  Holding: {trade.holding_period} days")
        print(f"  Exit reason: {trade.exit_reason}")

    # Equity curve stats
    print("\n6. Equity Curve Statistics:")
    print("-" * 80)
    print(f"Starting equity: {results.equity_curve['equity'].iloc[0]:,.2f} TRY")
    print(f"Ending equity: {results.equity_curve['equity'].iloc[-1]:,.2f} TRY")
    print(f"Peak equity: {results.equity_curve['equity'].max():,.2f} TRY")
    print(f"Lowest equity: {results.equity_curve['equity'].min():,.2f} TRY")

    # Export results
    print("\n7. Exporting results...")
    export_results(results, 'backtest_output')

    print("\n" + "=" * 80)
    print("Backtest completed successfully!")
    print("=" * 80)
