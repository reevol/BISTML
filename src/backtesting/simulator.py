"""
Backtesting Simulator for BIST AI Trading System

This module provides comprehensive backtesting capabilities including:
- Full historical simulation across user-defined time ranges
- Walk-forward analysis with rolling training/testing windows
- Monte Carlo simulation for robustness testing
- Performance metrics and analytics
- Transaction cost modeling (commissions, slippage)
- Risk management simulation
- Multiple timeframe support

Features:
- Realistic order execution simulation
- Position sizing and portfolio management
- Drawdown analysis
- Risk-adjusted returns
- Statistical significance testing
- Equity curve generation
- Trade-by-trade analysis

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class BacktestMode(Enum):
    """Backtesting mode"""
    HISTORICAL = "historical"  # Simple historical backtest
    WALK_FORWARD = "walk_forward"  # Walk-forward analysis
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    COMBINED = "combined"  # All methods combined


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExecutionModel(Enum):
    """Order execution models"""
    IMMEDIATE = "immediate"  # Execute at next bar open
    REALISTIC = "realistic"  # Execute with slippage and partial fills
    PESSIMISTIC = "pessimistic"  # Conservative execution assumptions


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters

    Attributes:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital
        commission_rate: Commission as fraction (e.g., 0.001 = 0.1%)
        slippage_model: Type of slippage model
        slippage_bps: Slippage in basis points
        position_size_pct: Default position size as % of portfolio
        max_positions: Maximum concurrent positions
        risk_per_trade: Maximum risk per trade as % of capital
        execution_model: Order execution model
        timeframe: Trading timeframe ('30min', '1h', '1d')
    """
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 5.0  # 5 basis points
    position_size_pct: float = 0.1  # 10% per position
    max_positions: int = 10
    risk_per_trade: float = 0.02  # 2% risk per trade
    execution_model: ExecutionModel = ExecutionModel.REALISTIC
    timeframe: str = '1h'
    min_trade_size: float = 1000.0  # Minimum trade size
    max_leverage: float = 1.0  # Maximum leverage

    def __post_init__(self):
        """Validate configuration"""
        if isinstance(self.execution_model, str):
            self.execution_model = ExecutionModel(self.execution_model)
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date)

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward analysis

    Attributes:
        train_period_days: Training period in days
        test_period_days: Testing period in days
        step_size_days: Step size for rolling window
        min_train_samples: Minimum samples required for training
        reoptimize: Whether to reoptimize parameters each period
    """
    train_period_days: int = 365  # 1 year training
    test_period_days: int = 90  # 3 months testing
    step_size_days: int = 30  # Move forward 1 month at a time
    min_train_samples: int = 100
    reoptimize: bool = True
    anchored: bool = False  # If True, always start from beginning


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation

    Attributes:
        n_simulations: Number of Monte Carlo runs
        randomization_method: Method for randomizing trades
        resample_trades: Whether to resample trades with replacement
        shuffle_returns: Whether to shuffle return sequence
        block_length: Block length for block bootstrap
        confidence_levels: Confidence levels for metrics (e.g., [0.05, 0.95])
    """
    n_simulations: int = 1000
    randomization_method: str = 'shuffle'  # 'shuffle', 'bootstrap', 'block_bootstrap'
    resample_trades: bool = True
    shuffle_returns: bool = True
    block_length: int = 20
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.50, 0.95])
    random_seed: Optional[int] = None


@dataclass
class Trade:
    """
    Represents a completed trade in the backtest

    Attributes:
        symbol: Stock symbol
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Entry price
        exit_price: Exit price
        shares: Number of shares
        direction: 1 for long, -1 for short
        pnl: Profit/loss
        pnl_pct: Profit/loss percentage
        commission: Total commission paid
        slippage: Total slippage cost
        holding_period: Holding period in bars
        entry_signal: Entry signal details
        exit_signal: Exit signal details
    """
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    direction: int = 1  # 1 = long, -1 = short
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    holding_period: int = 0
    entry_signal: Optional[Dict] = None
    exit_signal: Optional[Dict] = None
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.pnl == 0.0:
            gross_pnl = (self.exit_price - self.entry_price) * self.shares * self.direction
            self.pnl = gross_pnl - self.commission - self.slippage

        if self.pnl_pct == 0.0:
            cost_basis = self.entry_price * self.shares
            if cost_basis > 0:
                self.pnl_pct = (self.pnl / cost_basis) * 100

        if self.holding_period == 0 and self.exit_time and self.entry_time:
            self.holding_period = (self.exit_time - self.entry_time).days


@dataclass
class BacktestResult:
    """
    Results from a backtest run

    Attributes:
        trades: List of all trades
        equity_curve: Equity curve over time
        metrics: Performance metrics dictionary
        config: Backtest configuration used
        returns: Daily/periodic returns
        positions_history: History of positions over time
    """
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]
    config: BacktestConfig
    returns: pd.Series
    positions_history: Optional[pd.DataFrame] = None
    drawdown_series: Optional[pd.Series] = None

    def summary(self) -> pd.DataFrame:
        """Generate summary statistics DataFrame"""
        summary_data = {
            'Metric': [],
            'Value': []
        }

        for key, value in self.metrics.items():
            summary_data['Metric'].append(key)
            if isinstance(value, (int, float)):
                summary_data['Value'].append(f"{value:.4f}" if isinstance(value, float) else value)
            else:
                summary_data['Value'].append(str(value))

        return pd.DataFrame(summary_data)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'trades': [asdict(t) for t in self.trades],
            'equity_curve': self.equity_curve.to_dict(),
            'metrics': self.metrics,
            'config': asdict(self.config),
            'returns': self.returns.to_dict()
        }


# ============================================================================
# Main Backtesting Simulator
# ============================================================================

class BacktestSimulator:
    """
    Comprehensive backtesting simulator with multiple analysis methods

    This class provides:
    - Historical backtesting with realistic transaction costs
    - Walk-forward analysis for out-of-sample validation
    - Monte Carlo simulation for robustness testing
    - Comprehensive performance metrics
    - Risk analytics and reporting
    """

    def __init__(
        self,
        config: BacktestConfig,
        signal_generator: Optional[Callable] = None,
        price_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize backtesting simulator

        Args:
            config: Backtest configuration
            signal_generator: Function that generates trading signals
            price_data: Historical price data (OHLCV format)
        """
        self.config = config
        self.signal_generator = signal_generator
        self.price_data = price_data

        # State variables
        self.equity = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.open_orders: List[Dict] = []

        logger.info(f"Initialized BacktestSimulator with {config.initial_capital} capital")

    def load_price_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Load historical price data

        Args:
            data: DataFrame or path to CSV file with OHLCV data
            symbols: List of symbols to load (if None, load all)
        """
        if isinstance(data, (str, Path)):
            self.price_data = pd.read_csv(data, parse_dates=['timestamp'])
        else:
            self.price_data = data.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in self.price_data.columns:
            self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
            self.price_data.set_index('timestamp', inplace=True)

        # Filter by symbols if provided
        if symbols and 'symbol' in self.price_data.columns:
            self.price_data = self.price_data[self.price_data['symbol'].isin(symbols)]

        logger.info(f"Loaded price data: {len(self.price_data)} bars")

    def run_historical_backtest(
        self,
        signals: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run a historical backtest over the configured time period

        Args:
            signals: Pre-generated signals DataFrame (if None, use signal_generator)
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        logger.info(f"Starting historical backtest from {self.config.start_date} to {self.config.end_date}")

        # Reset state
        self._reset_state()

        # Get date range
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self._get_frequency()
        )

        # Main simulation loop
        for i, current_time in enumerate(date_range):
            # Update progress
            if progress_callback and i % 100 == 0:
                progress = (i / len(date_range)) * 100
                progress_callback(progress)

            # Get current market data
            market_data = self._get_market_data(current_time)
            if market_data is None or market_data.empty:
                continue

            # Update positions with current prices
            self._update_positions(current_time, market_data)

            # Generate or retrieve signals
            if signals is not None:
                current_signals = self._get_signals_for_time(signals, current_time)
            elif self.signal_generator is not None:
                current_signals = self.signal_generator(current_time, market_data)
            else:
                current_signals = []

            # Process signals and execute trades
            self._process_signals(current_time, current_signals, market_data)

            # Execute pending orders
            self._execute_orders(current_time, market_data)

            # Update equity curve
            self._update_equity_curve(current_time)

        # Close all remaining positions
        self._close_all_positions(date_range[-1])

        # Calculate results
        result = self._calculate_results()

        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return result

    def run_walk_forward_analysis(
        self,
        wf_config: WalkForwardConfig,
        strategy_optimizer: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis

        Walk-forward analysis divides the data into multiple train/test periods,
        trains the model on training data, and tests on out-of-sample data.

        Args:
            wf_config: Walk-forward configuration
            strategy_optimizer: Function to optimize strategy on training data
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing:
            - periods: List of WF periods with results
            - combined_result: Combined backtest result
            - metrics: Aggregated metrics
        """
        logger.info("Starting walk-forward analysis")

        # Generate walk-forward periods
        periods = self._generate_wf_periods(wf_config)
        logger.info(f"Generated {len(periods)} walk-forward periods")

        results = []
        all_trades = []
        combined_equity = []

        for i, period in enumerate(periods):
            logger.info(f"Processing WF period {i+1}/{len(periods)}: "
                       f"Train: {period['train_start']} to {period['train_end']}, "
                       f"Test: {period['test_start']} to {period['test_end']}")

            # Update progress
            if progress_callback:
                progress = (i / len(periods)) * 100
                progress_callback(progress)

            # Optimize strategy on training data if optimizer provided
            optimized_params = None
            if strategy_optimizer is not None:
                train_data = self._get_data_for_period(
                    period['train_start'],
                    period['train_end']
                )
                optimized_params = strategy_optimizer(train_data)
                logger.info(f"Optimized parameters: {optimized_params}")

            # Run backtest on test period
            test_config = BacktestConfig(
                start_date=period['test_start'],
                end_date=period['test_end'],
                initial_capital=self.config.initial_capital,
                commission_rate=self.config.commission_rate,
                slippage_bps=self.config.slippage_bps,
                position_size_pct=self.config.position_size_pct,
                max_positions=self.config.max_positions,
                execution_model=self.config.execution_model,
                timeframe=self.config.timeframe
            )

            # Create new simulator for this period
            period_sim = BacktestSimulator(
                config=test_config,
                signal_generator=self.signal_generator,
                price_data=self.price_data
            )

            # Run backtest
            period_result = period_sim.run_historical_backtest()

            # Store results
            results.append({
                'period': i + 1,
                'train_start': period['train_start'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'optimized_params': optimized_params,
                'result': period_result,
                'n_trades': len(period_result.trades),
                'total_return': period_result.metrics.get('total_return', 0),
                'sharpe_ratio': period_result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': period_result.metrics.get('max_drawdown', 0)
            })

            all_trades.extend(period_result.trades)

        # Calculate combined metrics
        combined_metrics = self._calculate_wf_metrics(results)

        logger.info(f"Walk-forward analysis complete. Total periods: {len(periods)}")
        logger.info(f"Combined Sharpe Ratio: {combined_metrics.get('avg_sharpe_ratio', 0):.3f}")

        return {
            'periods': results,
            'combined_metrics': combined_metrics,
            'all_trades': all_trades,
            'config': wf_config
        }

    def run_monte_carlo_simulation(
        self,
        mc_config: MonteCarloConfig,
        base_result: Optional[BacktestResult] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to test strategy robustness

        Monte Carlo simulation randomizes the sequence of trades or returns
        to generate a distribution of possible outcomes and assess risk.

        Args:
            mc_config: Monte Carlo configuration
            base_result: Base backtest result to use for simulation
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing:
            - simulations: List of simulation results
            - statistics: Statistical analysis of simulations
            - confidence_intervals: Confidence intervals for key metrics
        """
        logger.info(f"Starting Monte Carlo simulation with {mc_config.n_simulations} runs")

        # Run base backtest if not provided
        if base_result is None:
            base_result = self.run_historical_backtest()

        # Set random seed if provided
        if mc_config.random_seed is not None:
            np.random.seed(mc_config.random_seed)

        # Extract base trades
        base_trades = base_result.trades
        if len(base_trades) == 0:
            raise ValueError("No trades in base result for Monte Carlo simulation")

        logger.info(f"Base result has {len(base_trades)} trades")

        # Run simulations
        simulation_results = []

        for i in range(mc_config.n_simulations):
            if progress_callback and i % 100 == 0:
                progress = (i / mc_config.n_simulations) * 100
                progress_callback(progress)

            # Randomize trades based on method
            simulated_trades = self._randomize_trades(base_trades, mc_config)

            # Calculate equity curve for simulated trades
            equity_curve = self._calculate_equity_curve_from_trades(
                simulated_trades,
                self.config.initial_capital
            )

            # Calculate metrics
            metrics = self._calculate_metrics_from_trades(
                simulated_trades,
                equity_curve
            )

            simulation_results.append({
                'simulation': i + 1,
                'final_equity': equity_curve[-1] if len(equity_curve) > 0 else self.config.initial_capital,
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'n_trades': len(simulated_trades)
            })

        # Calculate statistics
        statistics = self._calculate_mc_statistics(simulation_results, mc_config)

        logger.info(f"Monte Carlo simulation complete")
        logger.info(f"Mean return: {statistics['mean_return']:.2%}")
        logger.info(f"Std return: {statistics['std_return']:.2%}")
        logger.info(f"Probability of profit: {statistics['prob_profit']:.2%}")

        return {
            'simulations': simulation_results,
            'statistics': statistics,
            'confidence_intervals': statistics.get('confidence_intervals', {}),
            'base_result': base_result,
            'config': mc_config
        }

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _reset_state(self):
        """Reset simulator state"""
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.open_orders = []

    def _get_frequency(self) -> str:
        """Get pandas frequency string from timeframe"""
        freq_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        return freq_map.get(self.config.timeframe, '1H')

    def _get_market_data(self, timestamp: datetime) -> Optional[pd.DataFrame]:
        """Get market data for a specific timestamp"""
        if self.price_data is None:
            return None

        # Get data for the current timestamp
        if isinstance(self.price_data.index, pd.DatetimeIndex):
            mask = self.price_data.index == timestamp
            return self.price_data[mask]
        else:
            mask = self.price_data['timestamp'] == timestamp
            return self.price_data[mask]

    def _get_signals_for_time(
        self,
        signals: pd.DataFrame,
        timestamp: datetime
    ) -> List[Dict]:
        """Extract signals for a specific timestamp"""
        if 'timestamp' in signals.columns:
            mask = signals['timestamp'] == timestamp
            signal_rows = signals[mask]
        else:
            mask = signals.index == timestamp
            signal_rows = signals[mask]

        return signal_rows.to_dict('records')

    def _update_positions(self, timestamp: datetime, market_data: pd.DataFrame):
        """Update position values with current market prices"""
        for symbol, position in self.positions.items():
            # Get current price
            symbol_data = market_data[market_data['symbol'] == symbol]
            if not symbol_data.empty:
                current_price = symbol_data.iloc[0]['close']
                position['current_price'] = current_price
                position['market_value'] = position['shares'] * current_price

                # Update MAE and MFE
                if 'entry_price' in position:
                    pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100

                    if pnl_pct < position.get('mae', 0):
                        position['mae'] = pnl_pct
                    if pnl_pct > position.get('mfe', 0):
                        position['mfe'] = pnl_pct

        # Update equity
        positions_value = sum(p['market_value'] for p in self.positions.values())
        self.equity = self.cash + positions_value

    def _process_signals(
        self,
        timestamp: datetime,
        signals: List[Dict],
        market_data: pd.DataFrame
    ):
        """Process trading signals and generate orders"""
        for signal in signals:
            symbol = signal.get('stock_code') or signal.get('symbol')
            signal_type = signal.get('signal') or signal.get('signal_type')

            # Check if we should enter a position
            if signal_type in ['STRONG_BUY', 'BUY'] and symbol not in self.positions:
                self._create_buy_order(timestamp, symbol, signal, market_data)

            # Check if we should exit a position
            elif signal_type in ['STRONG_SELL', 'SELL'] and symbol in self.positions:
                self._create_sell_order(timestamp, symbol, signal, market_data)

    def _create_buy_order(
        self,
        timestamp: datetime,
        symbol: str,
        signal: Dict,
        market_data: pd.DataFrame
    ):
        """Create a buy order"""
        # Check if we have capacity for new position
        if len(self.positions) >= self.config.max_positions:
            return

        # Get current price
        symbol_data = market_data[market_data['symbol'] == symbol]
        if symbol_data.empty:
            return

        current_price = symbol_data.iloc[0]['close']

        # Calculate position size
        position_value = self.equity * self.config.position_size_pct
        shares = int(position_value / current_price)

        if shares == 0 or position_value < self.config.min_trade_size:
            return

        # Check if we have enough cash
        total_cost = shares * current_price * (1 + self.config.commission_rate)
        if total_cost > self.cash:
            # Reduce shares to fit available cash
            shares = int(self.cash / (current_price * (1 + self.config.commission_rate)))
            if shares == 0:
                return

        # Create order
        order = {
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': current_price,
            'timestamp': timestamp,
            'signal': signal,
            'status': 'pending'
        }

        self.open_orders.append(order)

    def _create_sell_order(
        self,
        timestamp: datetime,
        symbol: str,
        signal: Dict,
        market_data: pd.DataFrame
    ):
        """Create a sell order"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Get current price
        symbol_data = market_data[market_data['symbol'] == symbol]
        if symbol_data.empty:
            return

        current_price = symbol_data.iloc[0]['close']

        # Create order to close position
        order = {
            'type': 'SELL',
            'symbol': symbol,
            'shares': position['shares'],
            'price': current_price,
            'timestamp': timestamp,
            'signal': signal,
            'status': 'pending'
        }

        self.open_orders.append(order)

    def _execute_orders(self, timestamp: datetime, market_data: pd.DataFrame):
        """Execute pending orders"""
        executed_orders = []

        for order in self.open_orders:
            if self._execute_single_order(order, timestamp, market_data):
                executed_orders.append(order)

        # Remove executed orders
        self.open_orders = [o for o in self.open_orders if o not in executed_orders]

    def _execute_single_order(
        self,
        order: Dict,
        timestamp: datetime,
        market_data: pd.DataFrame
    ) -> bool:
        """Execute a single order"""
        symbol = order['symbol']
        shares = order['shares']

        # Get execution price with slippage
        execution_price = self._calculate_execution_price(
            order['price'],
            order['type'],
            self.config.execution_model
        )

        # Calculate costs
        commission = execution_price * shares * self.config.commission_rate
        slippage_cost = abs(execution_price - order['price']) * shares

        if order['type'] == 'BUY':
            # Execute buy
            total_cost = execution_price * shares + commission

            if total_cost <= self.cash:
                # Create position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': execution_price,
                    'entry_time': timestamp,
                    'current_price': execution_price,
                    'market_value': execution_price * shares,
                    'entry_signal': order.get('signal'),
                    'mae': 0.0,
                    'mfe': 0.0
                }

                self.cash -= total_cost
                logger.debug(f"BUY {shares} shares of {symbol} at {execution_price:.2f}")
                return True

        elif order['type'] == 'SELL':
            # Execute sell
            if symbol in self.positions:
                position = self.positions[symbol]
                proceeds = execution_price * shares - commission

                # Create trade record
                trade = Trade(
                    symbol=symbol,
                    entry_time=position['entry_time'],
                    exit_time=timestamp,
                    entry_price=position['entry_price'],
                    exit_price=execution_price,
                    shares=shares,
                    direction=1,
                    commission=commission,
                    slippage=slippage_cost,
                    entry_signal=position.get('entry_signal'),
                    exit_signal=order.get('signal'),
                    mae=position.get('mae', 0.0),
                    mfe=position.get('mfe', 0.0)
                )

                self.trades.append(trade)
                self.cash += proceeds

                # Remove position
                del self.positions[symbol]

                logger.debug(f"SELL {shares} shares of {symbol} at {execution_price:.2f}, P&L: {trade.pnl:.2f}")
                return True

        return False

    def _calculate_execution_price(
        self,
        order_price: float,
        order_type: str,
        execution_model: ExecutionModel
    ) -> float:
        """Calculate execution price with slippage"""
        if execution_model == ExecutionModel.IMMEDIATE:
            return order_price

        # Calculate slippage
        slippage = order_price * (self.config.slippage_bps / 10000)

        if execution_model == ExecutionModel.PESSIMISTIC:
            slippage *= 2  # Double slippage for pessimistic model

        # Apply slippage based on order type
        if order_type == 'BUY':
            return order_price + slippage
        else:  # SELL
            return order_price - slippage

    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'positions_value': sum(p['market_value'] for p in self.positions.values()),
            'n_positions': len(self.positions)
        })

    def _close_all_positions(self, timestamp: datetime):
        """Close all open positions at end of backtest"""
        logger.info(f"Closing {len(self.positions)} open positions")

        for symbol, position in list(self.positions.items()):
            # Create sell order
            order = {
                'type': 'SELL',
                'symbol': symbol,
                'shares': position['shares'],
                'price': position['current_price'],
                'timestamp': timestamp,
                'signal': {'type': 'FORCED_EXIT'},
                'status': 'pending'
            }

            # Execute immediately
            self._execute_single_order(order, timestamp, pd.DataFrame())

    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results"""
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        # Calculate returns
        if not equity_df.empty and len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
        else:
            returns = pd.Series()

        # Calculate metrics
        metrics = self._calculate_performance_metrics(self.trades, equity_df)

        # Calculate drawdown series
        drawdown_series = None
        if not equity_df.empty:
            drawdown_series = self._calculate_drawdown_series(equity_df['equity'])

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics,
            config=self.config,
            returns=returns,
            drawdown_series=drawdown_series
        )

    def _calculate_performance_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Basic metrics
        metrics['n_trades'] = len(trades)

        if len(trades) == 0:
            return self._get_empty_metrics()

        # P&L metrics
        pnls = [t.pnl for t in trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['avg_pnl'] = np.mean(pnls)
        metrics['median_pnl'] = np.median(pnls)
        metrics['std_pnl'] = np.std(pnls)

        # Win/Loss metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        metrics['n_winners'] = len(winning_trades)
        metrics['n_losers'] = len(losing_trades)
        metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0

        if len(winning_trades) > 0:
            metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
            metrics['largest_win'] = max([t.pnl for t in winning_trades])
        else:
            metrics['avg_win'] = 0
            metrics['largest_win'] = 0

        if len(losing_trades) > 0:
            metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
            metrics['largest_loss'] = min([t.pnl for t in losing_trades])
        else:
            metrics['avg_loss'] = 0
            metrics['largest_loss'] = 0

        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Returns
        if not equity_curve.empty:
            initial_capital = self.config.initial_capital
            final_equity = equity_curve['equity'].iloc[-1]
            metrics['total_return'] = ((final_equity - initial_capital) / initial_capital)
            metrics['total_return_pct'] = metrics['total_return'] * 100

            # Annualized return
            n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if n_days > 0:
                metrics['annualized_return'] = ((final_equity / initial_capital) ** (365 / n_days)) - 1
                metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
            else:
                metrics['annualized_return'] = 0
                metrics['annualized_return_pct'] = 0

            # Sharpe ratio
            returns = equity_curve['equity'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Assuming 252 trading days per year
                metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0

            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics['sortino_ratio'] = (returns.mean() / downside_returns.std()) * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0

            # Maximum drawdown
            drawdown_series = self._calculate_drawdown_series(equity_curve['equity'])
            metrics['max_drawdown'] = drawdown_series.min()
            metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100

            # Calmar ratio
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
        else:
            metrics['total_return'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0

        # Average holding period
        holding_periods = [t.holding_period for t in trades if t.holding_period > 0]
        if holding_periods:
            metrics['avg_holding_period_days'] = np.mean(holding_periods)
        else:
            metrics['avg_holding_period_days'] = 0

        # Expectancy
        metrics['expectancy'] = (
            (metrics['win_rate'] * metrics['avg_win']) +
            ((1 - metrics['win_rate']) * metrics['avg_loss'])
        )

        return metrics

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no trades"""
        return {
            'n_trades': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0
        }

    def _calculate_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown

    def _generate_wf_periods(self, wf_config: WalkForwardConfig) -> List[Dict]:
        """Generate walk-forward periods"""
        periods = []

        current_date = self.config.start_date

        while current_date < self.config.end_date:
            # Calculate train period
            train_start = current_date
            train_end = current_date + timedelta(days=wf_config.train_period_days)

            # Calculate test period
            test_start = train_end
            test_end = test_start + timedelta(days=wf_config.test_period_days)

            # Check if test period exceeds end date
            if test_end > self.config.end_date:
                test_end = self.config.end_date

            if test_start >= self.config.end_date:
                break

            periods.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            # Move to next period
            current_date += timedelta(days=wf_config.step_size_days)

            # If anchored, always start from beginning
            if wf_config.anchored:
                current_date = self.config.start_date
                train_end = periods[-1]['test_end']

        return periods

    def _get_data_for_period(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Get data for a specific period"""
        if self.price_data is None:
            return pd.DataFrame()

        mask = (self.price_data.index >= start) & (self.price_data.index <= end)
        return self.price_data[mask]

    def _calculate_wf_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated walk-forward metrics"""
        if not results:
            return {}

        metrics = {
            'n_periods': len(results),
            'avg_return': np.mean([r['total_return'] for r in results]),
            'std_return': np.std([r['total_return'] for r in results]),
            'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
            'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'total_trades': sum([r['n_trades'] for r in results]),
            'consistency_score': sum([1 for r in results if r['total_return'] > 0]) / len(results)
        }

        return metrics

    def _randomize_trades(
        self,
        trades: List[Trade],
        mc_config: MonteCarloConfig
    ) -> List[Trade]:
        """Randomize trades for Monte Carlo simulation"""
        if mc_config.randomization_method == 'shuffle':
            # Simple shuffle
            shuffled = trades.copy()
            np.random.shuffle(shuffled)
            return shuffled

        elif mc_config.randomization_method == 'bootstrap':
            # Bootstrap resampling with replacement
            indices = np.random.choice(len(trades), size=len(trades), replace=True)
            return [trades[i] for i in indices]

        elif mc_config.randomization_method == 'block_bootstrap':
            # Block bootstrap to preserve serial correlation
            blocks = []
            i = 0
            while i < len(trades):
                block_size = min(mc_config.block_length, len(trades) - i)
                blocks.append(trades[i:i + block_size])
                i += block_size

            # Shuffle blocks
            np.random.shuffle(blocks)
            return [trade for block in blocks for trade in block]

        else:
            return trades.copy()

    def _calculate_equity_curve_from_trades(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> List[float]:
        """Calculate equity curve from a list of trades"""
        equity = [initial_capital]
        current_equity = initial_capital

        for trade in trades:
            current_equity += trade.pnl
            equity.append(current_equity)

        return equity

    def _calculate_metrics_from_trades(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> Dict[str, Any]:
        """Calculate metrics from trades and equity curve"""
        if len(trades) == 0 or len(equity_curve) == 0:
            return self._get_empty_metrics()

        # Create temporary equity series
        equity_series = pd.Series(equity_curve)

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Basic metrics
        metrics = {}
        metrics['n_trades'] = len(trades)

        pnls = [t.pnl for t in trades]
        metrics['total_pnl'] = sum(pnls)

        winning_trades = [t for t in trades if t.pnl > 0]
        metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0

        # Returns
        initial = equity_curve[0]
        final = equity_curve[-1]
        metrics['total_return'] = ((final - initial) / initial) if initial > 0 else 0

        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0

        # Max drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()

        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades])
        losing_trades = [t for t in trades if t.pnl < 0]
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return metrics

    def _calculate_mc_statistics(
        self,
        simulations: List[Dict],
        mc_config: MonteCarloConfig
    ) -> Dict[str, Any]:
        """Calculate statistics from Monte Carlo simulations"""
        if not simulations:
            return {}

        # Extract metrics
        returns = [s['total_return'] for s in simulations]
        sharpe_ratios = [s['sharpe_ratio'] for s in simulations]
        max_drawdowns = [s['max_drawdown'] for s in simulations]
        final_equities = [s['final_equity'] for s in simulations]

        # Calculate statistics
        statistics = {
            'n_simulations': len(simulations),
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'prob_profit': sum([1 for r in returns if r > 0]) / len(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'best_sharpe': max(sharpe_ratios),
            'worst_sharpe': min(sharpe_ratios)
        }

        # Calculate confidence intervals
        confidence_intervals = {}
        for level in mc_config.confidence_levels:
            percentile = level * 100
            confidence_intervals[f'return_{int(percentile)}pct'] = np.percentile(returns, percentile)
            confidence_intervals[f'sharpe_{int(percentile)}pct'] = np.percentile(sharpe_ratios, percentile)
            confidence_intervals[f'drawdown_{int(percentile)}pct'] = np.percentile(max_drawdowns, percentile)
            confidence_intervals[f'equity_{int(percentile)}pct'] = np.percentile(final_equities, percentile)

        statistics['confidence_intervals'] = confidence_intervals

        return statistics

    # ========================================================================
    # Visualization and Reporting Methods
    # ========================================================================

    def plot_equity_curve(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
        show_drawdown: bool = True
    ):
        """Plot equity curve and drawdown"""
        fig, axes = plt.subplots(2 if show_drawdown else 1, 1, figsize=(12, 8))

        if not show_drawdown:
            axes = [axes]

        # Plot equity curve
        axes[0].plot(result.equity_curve.index, result.equity_curve['equity'], label='Equity')
        axes[0].set_title('Equity Curve')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot drawdown
        if show_drawdown and result.drawdown_series is not None:
            axes[1].fill_between(
                result.drawdown_series.index,
                result.drawdown_series * 100,
                0,
                color='red',
                alpha=0.3
            )
            axes[1].set_title('Drawdown')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_monte_carlo_distribution(
        self,
        mc_result: Dict[str, Any],
        metric: str = 'total_return',
        save_path: Optional[str] = None
    ):
        """Plot Monte Carlo simulation distribution"""
        simulations = mc_result['simulations']
        values = [s[metric] for s in simulations]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')

        # Add vertical lines for percentiles
        statistics = mc_result['statistics']
        ci = statistics.get('confidence_intervals', {})

        for level, color in [(5, 'red'), (50, 'green'), (95, 'blue')]:
            key = f'{metric}_{level}pct'
            if key in ci:
                ax.axvline(ci[key], color=color, linestyle='--',
                          label=f'{level}th percentile: {ci[key]:.4f}')

        ax.set_title(f'Monte Carlo Distribution - {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Monte Carlo distribution to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_results(
        self,
        result: BacktestResult,
        output_dir: Union[str, Path],
        formats: List[str] = ['csv', 'json']
    ):
        """
        Export backtest results to various formats

        Args:
            result: Backtest result
            output_dir: Output directory
            formats: List of export formats ('csv', 'json', 'pickle')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export trades
        if 'csv' in formats and result.trades:
            trades_df = pd.DataFrame([asdict(t) for t in result.trades])
            trades_path = output_dir / f'trades_{timestamp}.csv'
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Exported trades to {trades_path}")

        # Export equity curve
        if 'csv' in formats and not result.equity_curve.empty:
            equity_path = output_dir / f'equity_curve_{timestamp}.csv'
            result.equity_curve.to_csv(equity_path)
            logger.info(f"Exported equity curve to {equity_path}")

        # Export metrics
        if 'json' in formats:
            metrics_path = output_dir / f'metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(result.metrics, f, indent=2, default=str)
            logger.info(f"Exported metrics to {metrics_path}")

        # Export full result
        if 'pickle' in formats:
            result_path = output_dir / f'backtest_result_{timestamp}.pkl'
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Exported full result to {result_path}")

        logger.info(f"Export complete to {output_dir}")


# ============================================================================
# Convenience Functions
# ============================================================================

def run_quick_backtest(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    signal_generator: Callable,
    price_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    **kwargs
) -> BacktestResult:
    """
    Convenience function to run a quick backtest

    Args:
        start_date: Start date
        end_date: End date
        signal_generator: Signal generation function
        price_data: Historical price data
        initial_capital: Starting capital
        **kwargs: Additional config parameters

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        initial_capital=initial_capital,
        **kwargs
    )

    simulator = BacktestSimulator(
        config=config,
        signal_generator=signal_generator,
        price_data=price_data
    )

    return simulator.run_historical_backtest()


def compare_strategies(
    strategies: Dict[str, Callable],
    price_data: pd.DataFrame,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple trading strategies

    Args:
        strategies: Dictionary of {strategy_name: signal_generator}
        price_data: Historical price data
        start_date: Start date
        end_date: End date
        **kwargs: Additional config parameters

    Returns:
        DataFrame with comparison metrics
    """
    results = {}

    for name, signal_gen in strategies.items():
        logger.info(f"Backtesting strategy: {name}")
        result = run_quick_backtest(
            start_date, end_date, signal_gen, price_data, **kwargs
        )
        results[name] = result.metrics

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T

    return comparison_df


if __name__ == "__main__":
    # Example usage
    logger.info("Backtesting Simulator Module")
    logger.info("Import this module to use the BacktestSimulator class")
