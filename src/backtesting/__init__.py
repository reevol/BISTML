"""
Backtesting Module for BIST AI Trading System

This module provides comprehensive backtesting functionality for evaluating
trading strategies on historical data.

Main Components:
- BacktestEngine: Main engine for running backtests
- BacktestConfig: Configuration for backtest parameters
- BacktestResults: Comprehensive results with metrics and analytics
- Trade: Individual trade representation

Author: BIST AI Trading System
Date: 2025-11-16
"""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResults,
    Trade,
    PositionSizing,
    OrderType,
    SlippageModel,
    create_backtest_config,
    export_results,
    quick_backtest
)

# Import metrics module for standalone metrics calculations
from .metrics import (
    calculate_all_metrics,
    calculate_win_rate,
    calculate_average_profit_loss,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_profit_factor,
    calculate_recovery_factor,
    rolling_sharpe_ratio,
    rolling_max_drawdown,
    compare_strategies,
    PerformanceMetrics,
    MetricsError,
    InsufficientDataError,
    InvalidDataError,
)

__all__ = [
    # Backtesting engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResults',
    'Trade',
    'PositionSizing',
    'OrderType',
    'SlippageModel',
    'create_backtest_config',
    'export_results',
    'quick_backtest',
    # Performance metrics
    'calculate_all_metrics',
    'calculate_win_rate',
    'calculate_average_profit_loss',
    'calculate_max_drawdown',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_profit_factor',
    'calculate_recovery_factor',
    'rolling_sharpe_ratio',
    'rolling_max_drawdown',
    'compare_strategies',
    'PerformanceMetrics',
    'MetricsError',
    'InsufficientDataError',
    'InvalidDataError',
]

__version__ = '1.0.0'
