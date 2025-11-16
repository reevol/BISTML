"""
Portfolio Management Module for BIST AI Trading System

This module provides comprehensive portfolio management and optimization:
- Position tracking and P&L management
- Portfolio optimization and position sizing
- Risk management and constraints
- Kelly Criterion, Risk Parity, Equal Weight strategies
- Mean-Variance optimization (Markowitz)
"""

from .manager import (
    PortfolioManager,
    Position,
    Transaction,
    TransactionType,
    CostBasisMethod,
    PortfolioError,
    InsufficientSharesError,
    InvalidTransactionError,
    PositionNotFoundError,
    create_portfolio
)

from .optimization import (
    PortfolioOptimizer,
    AssetMetrics,
    PortfolioWeights,
    RiskConstraints,
    PositionSizingMethod,
    RiskLevel,
    OptimizationError,
    InvalidParameterError,
    InsufficientDataError,
    create_optimizer,
    optimize_portfolio
)

__all__ = [
    # Portfolio Manager
    'PortfolioManager',
    'Position',
    'Transaction',
    'TransactionType',
    'CostBasisMethod',
    'PortfolioError',
    'InsufficientSharesError',
    'InvalidTransactionError',
    'PositionNotFoundError',
    'create_portfolio',

    # Portfolio Optimizer
    'PortfolioOptimizer',
    'AssetMetrics',
    'PortfolioWeights',
    'RiskConstraints',
    'PositionSizingMethod',
    'RiskLevel',
    'OptimizationError',
    'InvalidParameterError',
    'InsufficientDataError',
    'create_optimizer',
    'optimize_portfolio'
]
