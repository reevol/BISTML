"""
Backtesting Performance Metrics for BIST AI Trading System

This module provides comprehensive performance metrics for backtesting trading strategies.
It calculates industry-standard risk-adjusted returns and performance indicators including:

- Win Rate: Percentage of profitable trades
- Average Profit/Loss: Mean return per trade
- Maximum Drawdown: Largest peak-to-trough decline
- Sharpe Ratio: Risk-adjusted return metric
- Sortino Ratio: Downside risk-adjusted return
- Calmar Ratio: Return relative to maximum drawdown
- Profit Factor: Ratio of gross profits to gross losses
- Recovery Factor: Net profit relative to maximum drawdown

All metrics support both trade-level and equity curve analysis, with configurable
risk-free rates and flexible data input formats.

Features:
- Multiple metric calculation methods
- Support for various data formats (DataFrame, Series, arrays)
- Annualization of returns
- Comprehensive error handling
- Detailed metric explanations in docstrings

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class MetricsError(Exception):
    """Base exception for metrics calculation errors"""
    pass


class InsufficientDataError(MetricsError):
    """Raised when insufficient data is provided for metric calculation"""
    pass


class InvalidDataError(MetricsError):
    """Raised when data format is invalid"""
    pass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive performance metrics.

    Attributes:
        win_rate: Percentage of profitable trades (0-100)
        avg_profit: Average profit per trade
        avg_loss: Average loss per trade
        avg_profit_loss: Average profit/loss across all trades
        max_drawdown: Maximum drawdown (as decimal, e.g., 0.15 = 15%)
        max_drawdown_pct: Maximum drawdown as percentage
        sharpe_ratio: Sharpe ratio (annualized)
        sortino_ratio: Sortino ratio (annualized)
        calmar_ratio: Calmar ratio (annualized)
        profit_factor: Ratio of gross profits to gross losses
        recovery_factor: Net profit divided by maximum drawdown
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_return: Total return (as decimal)
        total_return_pct: Total return as percentage
        annualized_return: Annualized return (as decimal)
        annualized_volatility: Annualized volatility
        max_consecutive_wins: Maximum consecutive winning trades
        max_consecutive_losses: Maximum consecutive losing trades
        avg_winning_trade: Average return of winning trades
        avg_losing_trade: Average return of losing trades
        largest_win: Largest winning trade
        largest_loss: Largest losing trade (as negative value)
        expectancy: Expected value per trade
        risk_reward_ratio: Average win to average loss ratio
        metadata: Additional metadata
    """
    win_rate: float
    avg_profit: float
    avg_loss: float
    avg_profit_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    recovery_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    total_return_pct: float
    annualized_return: float
    annualized_volatility: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    expectancy: float
    risk_reward_ratio: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame for easy display"""
        data = self.to_dict()
        if data['metadata']:
            data.pop('metadata')  # Remove metadata for cleaner display

        return pd.DataFrame([data]).T.rename(columns={0: 'Value'})

    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = [
            "=" * 70,
            "PERFORMANCE METRICS SUMMARY",
            "=" * 70,
            "",
            "TRADE STATISTICS",
            "-" * 70,
            f"Total Trades:              {self.total_trades}",
            f"Winning Trades:            {self.winning_trades} ({self.win_rate:.2f}%)",
            f"Losing Trades:             {self.losing_trades}",
            f"Max Consecutive Wins:      {self.max_consecutive_wins}",
            f"Max Consecutive Losses:    {self.max_consecutive_losses}",
            "",
            "RETURN METRICS",
            "-" * 70,
            f"Total Return:              {self.total_return_pct:.2f}%",
            f"Annualized Return:         {self.annualized_return * 100:.2f}%",
            f"Annualized Volatility:     {self.annualized_volatility * 100:.2f}%",
            f"Average Profit/Loss:       {self.avg_profit_loss:.4f}",
            f"Average Win:               {self.avg_winning_trade:.4f}",
            f"Average Loss:              {self.avg_losing_trade:.4f}",
            f"Largest Win:               {self.largest_win:.4f}",
            f"Largest Loss:              {self.largest_loss:.4f}",
            f"Expectancy:                {self.expectancy:.4f}",
            "",
            "RISK METRICS",
            "-" * 70,
            f"Maximum Drawdown:          {self.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio:              {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:             {self.sortino_ratio:.3f}",
            f"Calmar Ratio:              {self.calmar_ratio:.3f}",
            f"Profit Factor:             {self.profit_factor:.3f}",
            f"Recovery Factor:           {self.recovery_factor:.3f}",
            f"Risk/Reward Ratio:         {self.risk_reward_ratio:.3f}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ============================================================================
# Core Metrics Functions
# ============================================================================

def calculate_win_rate(returns: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calculate win rate (percentage of profitable trades).

    Args:
        returns: Series, array, or list of trade returns

    Returns:
        Win rate as percentage (0-100)

    Raises:
        InsufficientDataError: If no trades provided

    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> win_rate = calculate_win_rate(returns)
        >>> print(f"Win Rate: {win_rate:.2f}%")
        Win Rate: 60.00%
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) == 0:
        raise InsufficientDataError("No trades provided for win rate calculation")

    winning_trades = np.sum(returns_array > 0)
    total_trades = len(returns_array)

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

    return float(win_rate)


def calculate_average_profit_loss(
    returns: Union[pd.Series, np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    Calculate average profit, average loss, and overall average.

    Args:
        returns: Series, array, or list of trade returns

    Returns:
        Dictionary containing:
            - avg_profit: Average of winning trades
            - avg_loss: Average of losing trades
            - avg_profit_loss: Overall average
            - avg_winning_trade: Same as avg_profit
            - avg_losing_trade: Same as avg_loss

    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> avg_metrics = calculate_average_profit_loss(returns)
        >>> print(f"Avg Profit: {avg_metrics['avg_profit']:.4f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) == 0:
        return {
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'avg_profit_loss': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0
        }

    winning_trades = returns_array[returns_array > 0]
    losing_trades = returns_array[returns_array < 0]

    avg_profit = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
    avg_loss = float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0
    avg_profit_loss = float(np.mean(returns_array))

    return {
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'avg_profit_loss': avg_profit_loss,
        'avg_winning_trade': avg_profit,
        'avg_losing_trade': avg_loss
    }


def calculate_max_drawdown(
    equity_curve: Union[pd.Series, np.ndarray, List[float]],
    return_series: bool = False
) -> Union[float, Tuple[float, pd.Series]]:
    """
    Calculate maximum drawdown from equity curve.

    Maximum drawdown is the largest peak-to-trough decline in the equity curve,
    representing the maximum loss an investor would have experienced.

    Args:
        equity_curve: Equity curve (cumulative portfolio value over time)
        return_series: If True, also return the drawdown series

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 for 15% drawdown)
        If return_series=True, returns (max_drawdown, drawdown_series)

    Example:
        >>> equity = [100, 110, 105, 120, 115, 125]
        >>> max_dd = calculate_max_drawdown(equity)
        >>> print(f"Max Drawdown: {max_dd * 100:.2f}%")
    """
    equity_array = _ensure_array(equity_curve)

    if len(equity_array) < 2:
        if return_series:
            return 0.0, pd.Series([0.0] * len(equity_array))
        return 0.0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)

    # Calculate drawdown at each point
    drawdown = (equity_array - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_drawdown = float(np.min(drawdown))

    if return_series:
        return abs(max_drawdown), pd.Series(drawdown)

    return abs(max_drawdown)


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return metric).

    Sharpe ratio measures the excess return per unit of risk (volatility).
    Higher values indicate better risk-adjusted performance.

    Formula: (Mean Return - Risk Free Rate) / Standard Deviation of Returns

    Args:
        returns: Series, array, or list of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        annualize: Whether to annualize the result

    Returns:
        Sharpe ratio (annualized if annualize=True)

    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
        >>> print(f"Sharpe Ratio: {sharpe:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) < 2:
        return 0.0

    # Calculate mean and std of returns
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)

    if std_return == 0:
        return 0.0

    # Calculate period risk-free rate
    period_rf_rate = risk_free_rate / periods_per_year if annualize else risk_free_rate

    # Calculate Sharpe ratio
    sharpe = (mean_return - period_rf_rate) / std_return

    # Annualize if requested
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    annualize: bool = True,
    target_return: Optional[float] = None
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return metric).

    Similar to Sharpe ratio but only considers downside volatility (negative returns),
    which is more relevant for investors concerned about losses.

    Formula: (Mean Return - Target Return) / Downside Deviation

    Args:
        returns: Series, array, or list of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        annualize: Whether to annualize the result
        target_return: Minimum acceptable return (default: risk_free_rate)

    Returns:
        Sortino ratio (annualized if annualize=True)

    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        >>> print(f"Sortino Ratio: {sortino:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) < 2:
        return 0.0

    # Use risk-free rate as target if not specified
    if target_return is None:
        target_return = risk_free_rate / periods_per_year if annualize else risk_free_rate

    # Calculate mean return
    mean_return = np.mean(returns_array)

    # Calculate downside deviation (only negative returns)
    downside_returns = returns_array - target_return
    downside_returns = downside_returns[downside_returns < 0]

    if len(downside_returns) == 0:
        # No downside - perfect strategy
        return np.inf if mean_return > target_return else 0.0

    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    if downside_deviation == 0:
        return 0.0

    # Calculate Sortino ratio
    sortino = (mean_return - target_return) / downside_deviation

    # Annualize if requested
    if annualize:
        sortino = sortino * np.sqrt(periods_per_year)

    return float(sortino)


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    equity_curve: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (return to maximum drawdown ratio).

    Calmar ratio measures the annualized return relative to maximum drawdown,
    indicating how much return is generated per unit of downside risk.

    Formula: Annualized Return / Maximum Drawdown

    Args:
        returns: Series, array, or list of returns
        equity_curve: Optional equity curve (if None, calculated from returns)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)

    Returns:
        Calmar ratio

    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> calmar = calculate_calmar_ratio(returns)
        >>> print(f"Calmar Ratio: {calmar:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) < 2:
        return 0.0

    # Calculate annualized return
    total_return = np.prod(1 + returns_array) - 1
    num_years = len(returns_array) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    # Calculate or use provided equity curve
    if equity_curve is None:
        equity_curve = (1 + returns_array).cumprod()

    # Calculate max drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        # No drawdown - perfect strategy
        return np.inf if annualized_return > 0 else 0.0

    calmar = annualized_return / max_dd

    return float(calmar)


def calculate_profit_factor(
    returns: Union[pd.Series, np.ndarray, List[float]]
) -> float:
    """
    Calculate profit factor (ratio of gross profits to gross losses).

    Profit factor measures the ratio of total winning trades to total losing trades.
    Values > 1 indicate a profitable strategy.

    Formula: Sum of Winning Trades / Abs(Sum of Losing Trades)

    Args:
        returns: Series, array, or list of trade returns

    Returns:
        Profit factor (>1 is profitable, <1 is losing)

    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> pf = calculate_profit_factor(returns)
        >>> print(f"Profit Factor: {pf:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) == 0:
        return 0.0

    gross_profits = np.sum(returns_array[returns_array > 0])
    gross_losses = abs(np.sum(returns_array[returns_array < 0]))

    if gross_losses == 0:
        # No losses
        return np.inf if gross_profits > 0 else 0.0

    profit_factor = gross_profits / gross_losses

    return float(profit_factor)


def calculate_recovery_factor(
    returns: Union[pd.Series, np.ndarray, List[float]],
    equity_curve: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
) -> float:
    """
    Calculate recovery factor (net profit to maximum drawdown ratio).

    Recovery factor measures how much profit was made relative to the maximum
    drawdown experienced. Higher values indicate better recovery from losses.

    Formula: Net Profit / Maximum Drawdown

    Args:
        returns: Series, array, or list of returns
        equity_curve: Optional equity curve (if None, calculated from returns)

    Returns:
        Recovery factor

    Example:
        >>> returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        >>> rf = calculate_recovery_factor(returns)
        >>> print(f"Recovery Factor: {rf:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) == 0:
        return 0.0

    # Calculate net profit
    net_profit = np.sum(returns_array)

    # Calculate or use provided equity curve
    if equity_curve is None:
        equity_curve = (1 + returns_array).cumprod()

    # Calculate max drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        # No drawdown
        return np.inf if net_profit > 0 else 0.0

    recovery_factor = net_profit / max_dd

    return float(recovery_factor)


# ============================================================================
# Comprehensive Metrics Calculation
# ============================================================================

def calculate_all_metrics(
    returns: Union[pd.Series, np.ndarray, List[float]],
    equity_curve: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    initial_capital: float = 100000.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics in one comprehensive analysis.

    This is the main function for calculating a complete set of trading
    performance metrics from a series of returns.

    Args:
        returns: Series, array, or list of trade returns
        equity_curve: Optional equity curve (if None, calculated from returns)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        initial_capital: Initial portfolio value for equity curve calculation

    Returns:
        PerformanceMetrics object containing all calculated metrics

    Raises:
        InsufficientDataError: If insufficient data provided

    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> metrics = calculate_all_metrics(returns, risk_free_rate=0.02)
        >>> print(metrics)
        >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    """
    returns_array = _ensure_array(returns)

    if len(returns_array) == 0:
        raise InsufficientDataError("No returns provided for metrics calculation")

    # Calculate or use provided equity curve
    if equity_curve is None:
        equity_curve = initial_capital * (1 + returns_array).cumprod()
    else:
        equity_curve = _ensure_array(equity_curve)

    # Basic trade statistics
    total_trades = len(returns_array)
    winning_trades = int(np.sum(returns_array > 0))
    losing_trades = int(np.sum(returns_array < 0))

    # Win rate
    win_rate = calculate_win_rate(returns_array)

    # Average profit/loss metrics
    avg_metrics = calculate_average_profit_loss(returns_array)

    # Drawdown metrics
    max_dd, dd_series = calculate_max_drawdown(equity_curve, return_series=True)

    # Risk-adjusted ratios
    sharpe = calculate_sharpe_ratio(
        returns_array,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        annualize=True
    )

    sortino = calculate_sortino_ratio(
        returns_array,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        annualize=True
    )

    calmar = calculate_calmar_ratio(
        returns_array,
        equity_curve=equity_curve,
        periods_per_year=periods_per_year
    )

    # Profit metrics
    profit_factor = calculate_profit_factor(returns_array)
    recovery_factor = calculate_recovery_factor(returns_array, equity_curve=equity_curve)

    # Return metrics
    total_return = np.prod(1 + returns_array) - 1
    num_years = len(returns_array) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    annualized_volatility = np.std(returns_array, ddof=1) * np.sqrt(periods_per_year)

    # Consecutive wins/losses
    max_consec_wins, max_consec_losses = _calculate_consecutive_trades(returns_array)

    # Largest win/loss
    largest_win = float(np.max(returns_array)) if len(returns_array) > 0 else 0.0
    largest_loss = float(np.min(returns_array)) if len(returns_array) > 0 else 0.0

    # Expectancy
    expectancy = (
        (win_rate / 100) * avg_metrics['avg_profit'] +
        ((100 - win_rate) / 100) * avg_metrics['avg_loss']
    )

    # Risk/Reward ratio
    if avg_metrics['avg_loss'] != 0:
        risk_reward = abs(avg_metrics['avg_profit'] / avg_metrics['avg_loss'])
    else:
        risk_reward = np.inf if avg_metrics['avg_profit'] > 0 else 0.0

    # Create metadata
    metadata = {
        'risk_free_rate': risk_free_rate,
        'periods_per_year': periods_per_year,
        'initial_capital': initial_capital,
        'calculation_date': datetime.now().isoformat(),
        'num_periods': len(returns_array),
        'years_analyzed': num_years
    }

    # Create and return PerformanceMetrics object
    metrics = PerformanceMetrics(
        win_rate=win_rate,
        avg_profit=avg_metrics['avg_profit'],
        avg_loss=avg_metrics['avg_loss'],
        avg_profit_loss=avg_metrics['avg_profit_loss'],
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        profit_factor=profit_factor,
        recovery_factor=recovery_factor,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_return=total_return,
        total_return_pct=total_return * 100,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        avg_winning_trade=avg_metrics['avg_winning_trade'],
        avg_losing_trade=avg_metrics['avg_losing_trade'],
        largest_win=largest_win,
        largest_loss=largest_loss,
        expectancy=expectancy,
        risk_reward_ratio=risk_reward,
        metadata=metadata
    )

    return metrics


# ============================================================================
# Utility Functions
# ============================================================================

def _ensure_array(data: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
    """
    Convert input data to numpy array.

    Args:
        data: Input data (Series, array, or list)

    Returns:
        Numpy array

    Raises:
        InvalidDataError: If data cannot be converted
    """
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise InvalidDataError(f"Unsupported data type: {type(data)}")


def _calculate_consecutive_trades(returns: np.ndarray) -> Tuple[int, int]:
    """
    Calculate maximum consecutive wins and losses.

    Args:
        returns: Array of returns

    Returns:
        Tuple of (max_consecutive_wins, max_consecutive_losses)
    """
    if len(returns) == 0:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for ret in returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif ret < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            # Zero return - reset both
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def rolling_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    window: int = 30,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio over a window.

    Args:
        returns: Series or array of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year for annualization

    Returns:
        Series of rolling Sharpe ratios
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf

    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)

    return sharpe


def rolling_max_drawdown(
    equity_curve: Union[pd.Series, np.ndarray],
    window: int = 30
) -> pd.Series:
    """
    Calculate rolling maximum drawdown over a window.

    Args:
        equity_curve: Series or array of equity values
        window: Rolling window size

    Returns:
        Series of rolling maximum drawdowns
    """
    if isinstance(equity_curve, np.ndarray):
        equity_curve = pd.Series(equity_curve)

    def calc_dd(x):
        if len(x) < 2:
            return 0.0
        running_max = np.maximum.accumulate(x)
        drawdown = (x - running_max) / running_max
        return abs(np.min(drawdown))

    rolling_dd = equity_curve.rolling(window).apply(calc_dd, raw=True)

    return rolling_dd


def compare_strategies(
    strategies_returns: Dict[str, Union[pd.Series, np.ndarray, List[float]]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Args:
        strategies_returns: Dictionary mapping strategy names to their returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        DataFrame comparing all strategies

    Example:
        >>> strategy_a = [0.01, -0.005, 0.02]
        >>> strategy_b = [0.015, 0.005, 0.01]
        >>> comparison = compare_strategies({
        ...     'Strategy A': strategy_a,
        ...     'Strategy B': strategy_b
        ... })
        >>> print(comparison)
    """
    results = {}

    for name, returns in strategies_returns.items():
        try:
            metrics = calculate_all_metrics(
                returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year
            )
            results[name] = metrics.to_dict()
        except Exception as e:
            logger.error(f"Error calculating metrics for {name}: {str(e)}")
            results[name] = None

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Remove metadata column if present
    if 'metadata' in df.columns:
        df = df.drop('metadata', axis=1)

    return df


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage of the metrics module"""

    print("=" * 80)
    print("BACKTESTING METRICS MODULE - EXAMPLE USAGE")
    print("=" * 80)

    # Generate sample returns
    np.random.seed(42)
    n_trades = 100

    # Simulate a profitable strategy with some losses
    winning_returns = np.random.normal(0.02, 0.01, int(n_trades * 0.6))
    losing_returns = np.random.normal(-0.015, 0.01, int(n_trades * 0.4))
    returns = np.concatenate([winning_returns, losing_returns])
    np.random.shuffle(returns)

    print("\n1. Sample Returns Generated")
    print("-" * 80)
    print(f"Total trades: {len(returns)}")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Std return: {np.std(returns):.4f}")

    # Calculate all metrics
    print("\n2. Calculating All Performance Metrics")
    print("-" * 80)

    metrics = calculate_all_metrics(
        returns=returns,
        risk_free_rate=0.02,  # 2% annual risk-free rate
        periods_per_year=252,  # Daily returns
        initial_capital=100000.0
    )

    # Display metrics
    print("\n" + str(metrics))

    # Individual metric calculations
    print("\n3. Individual Metric Calculations")
    print("-" * 80)

    print(f"\nWin Rate: {calculate_win_rate(returns):.2f}%")

    avg_pl = calculate_average_profit_loss(returns)
    print(f"Average Profit: {avg_pl['avg_profit']:.4f}")
    print(f"Average Loss: {avg_pl['avg_loss']:.4f}")

    equity_curve = 100000 * (1 + pd.Series(returns)).cumprod()
    max_dd = calculate_max_drawdown(equity_curve)
    print(f"Maximum Drawdown: {max_dd * 100:.2f}%")

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    print(f"Sharpe Ratio: {sharpe:.3f}")

    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    print(f"Sortino Ratio: {sortino:.3f}")

    calmar = calculate_calmar_ratio(returns, equity_curve=equity_curve)
    print(f"Calmar Ratio: {calmar:.3f}")

    profit_factor = calculate_profit_factor(returns)
    print(f"Profit Factor: {profit_factor:.3f}")

    recovery_factor = calculate_recovery_factor(returns, equity_curve=equity_curve)
    print(f"Recovery Factor: {recovery_factor:.3f}")

    # Export to DataFrame
    print("\n4. Metrics as DataFrame")
    print("-" * 80)
    df = metrics.to_dataframe()
    print(df.head(15))

    # Compare multiple strategies
    print("\n5. Comparing Multiple Strategies")
    print("-" * 80)

    # Generate returns for a second strategy (more conservative)
    returns_conservative = np.random.normal(0.01, 0.005, n_trades)

    comparison = compare_strategies({
        'Aggressive': returns,
        'Conservative': returns_conservative
    })

    print("\nKey Metrics Comparison:")
    print(comparison[['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
                      'win_rate', 'profit_factor']].to_string())

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
