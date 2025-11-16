"""
Portfolio Optimization and Position Sizing for BIST AI Trading System

This module provides advanced portfolio optimization techniques including:
- Kelly Criterion for optimal position sizing
- Risk Parity allocation
- Equal weighting strategies
- Maximum Sharpe Ratio optimization
- Minimum Variance portfolio
- Mean-Variance optimization
- Comprehensive risk management rules

The module integrates with the portfolio manager and signal generator to
provide optimal position sizing based on predicted returns, volatility,
and risk tolerance.

Features:
- Kelly Criterion (full and fractional)
- Risk Parity weighting
- Equal weight allocation
- Volatility-based position sizing
- Maximum drawdown constraints
- Correlation-based diversification
- Dynamic position limits
- Portfolio rebalancing logic

Author: BIST AI Trading System
Date: 2025-11-16
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    KELLY_CRITERION = "KELLY"
    FRACTIONAL_KELLY = "FRACTIONAL_KELLY"
    RISK_PARITY = "RISK_PARITY"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    VOLATILITY_WEIGHTED = "VOLATILITY_WEIGHTED"
    MAX_SHARPE = "MAX_SHARPE"
    MIN_VARIANCE = "MIN_VARIANCE"
    MEAN_VARIANCE = "MEAN_VARIANCE"


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    VERY_AGGRESSIVE = "VERY_AGGRESSIVE"


# Risk level to Kelly fraction mapping
KELLY_FRACTIONS = {
    RiskLevel.CONSERVATIVE: 0.25,
    RiskLevel.MODERATE: 0.50,
    RiskLevel.AGGRESSIVE: 0.75,
    RiskLevel.VERY_AGGRESSIVE: 1.00
}

# Maximum position sizes by risk level
MAX_POSITION_SIZES = {
    RiskLevel.CONSERVATIVE: 0.10,      # 10% max per position
    RiskLevel.MODERATE: 0.15,          # 15% max per position
    RiskLevel.AGGRESSIVE: 0.20,        # 20% max per position
    RiskLevel.VERY_AGGRESSIVE: 0.25    # 25% max per position
}

# Maximum portfolio concentration by risk level
MAX_CONCENTRATION = {
    RiskLevel.CONSERVATIVE: 0.30,      # Top 3 positions max 30%
    RiskLevel.MODERATE: 0.40,          # Top 3 positions max 40%
    RiskLevel.AGGRESSIVE: 0.50,        # Top 3 positions max 50%
    RiskLevel.VERY_AGGRESSIVE: 0.60    # Top 3 positions max 60%
}


# ============================================================================
# Exceptions
# ============================================================================

class OptimizationError(Exception):
    """Base exception for optimization errors"""
    pass


class InvalidParameterError(OptimizationError):
    """Raised when invalid parameters are provided"""
    pass


class InsufficientDataError(OptimizationError):
    """Raised when insufficient data for optimization"""
    pass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AssetMetrics:
    """
    Metrics for a single asset

    Attributes:
        symbol: Stock symbol
        expected_return: Expected return (annualized)
        volatility: Volatility (annualized standard deviation)
        sharpe_ratio: Sharpe ratio
        win_rate: Historical win rate (0-1)
        current_price: Current market price
        max_drawdown: Maximum historical drawdown
        correlation_to_market: Correlation to market index
    """
    symbol: str
    expected_return: float
    volatility: float
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    current_price: Optional[float] = None
    max_drawdown: Optional[float] = None
    correlation_to_market: Optional[float] = None

    def __post_init__(self):
        """Calculate Sharpe ratio if not provided"""
        if self.sharpe_ratio == 0.0 and self.volatility > 0:
            # Assume risk-free rate of 0 for simplicity
            self.sharpe_ratio = self.expected_return / self.volatility


@dataclass
class PortfolioWeights:
    """
    Portfolio allocation weights

    Attributes:
        weights: Dictionary mapping symbols to weights (0-1)
        method: Position sizing method used
        total_weight: Total weight (should be ~1.0)
        expected_return: Expected portfolio return
        expected_volatility: Expected portfolio volatility
        sharpe_ratio: Expected portfolio Sharpe ratio
        metadata: Additional information
    """
    weights: Dict[str, float]
    method: PositionSizingMethod
    total_weight: float = 1.0
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
            self.total_weight = 1.0

    def apply_max_position_size(self, max_size: float):
        """Cap individual position sizes"""
        for symbol in self.weights:
            if self.weights[symbol] > max_size:
                self.weights[symbol] = max_size
        self.normalize()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'weights': self.weights,
            'method': self.method.value,
            'total_weight': self.total_weight,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'metadata': self.metadata
        }


@dataclass
class RiskConstraints:
    """
    Risk management constraints

    Attributes:
        max_position_size: Maximum size per position (0-1)
        max_portfolio_volatility: Maximum portfolio volatility
        max_drawdown: Maximum acceptable drawdown
        max_correlation: Maximum correlation between positions
        min_positions: Minimum number of positions
        max_positions: Maximum number of positions
        max_sector_concentration: Maximum weight in single sector
        min_liquidity: Minimum daily trading volume
    """
    max_position_size: float = 0.20
    max_portfolio_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_correlation: float = 0.80
    min_positions: int = 3
    max_positions: int = 20
    max_sector_concentration: float = 0.40
    min_liquidity: Optional[float] = None


# ============================================================================
# Portfolio Optimizer Class
# ============================================================================

class PortfolioOptimizer:
    """
    Portfolio optimization and position sizing engine

    This class provides multiple position sizing strategies and risk
    management capabilities for optimal portfolio construction.
    """

    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        risk_free_rate: float = 0.0,
        constraints: Optional[RiskConstraints] = None
    ):
        """
        Initialize Portfolio Optimizer

        Args:
            risk_level: Overall risk tolerance level
            risk_free_rate: Risk-free rate for Sharpe calculation (annualized)
            constraints: Risk management constraints
        """
        self.risk_level = risk_level
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or RiskConstraints()

        # Set max position size based on risk level
        if self.constraints.max_position_size > MAX_POSITION_SIZES[risk_level]:
            logger.warning(
                f"Max position size {self.constraints.max_position_size} exceeds "
                f"recommended {MAX_POSITION_SIZES[risk_level]} for {risk_level.value}"
            )

        logger.info(
            f"PortfolioOptimizer initialized with risk_level={risk_level.value}, "
            f"max_position_size={self.constraints.max_position_size}"
        )

    # ========================================================================
    # Kelly Criterion Methods
    # ========================================================================

    def kelly_criterion(
        self,
        expected_return: float,
        win_rate: float,
        win_loss_ratio: Optional[float] = None,
        variance: Optional[float] = None,
        fractional: bool = True
    ) -> float:
        """
        Calculate Kelly Criterion position size

        The Kelly Criterion maximizes logarithmic wealth growth by determining
        the optimal fraction of capital to allocate to each position.

        Two formulas are supported:
        1. Binary outcome: f = (p*b - q) / b
           where p = win rate, q = 1-p, b = win/loss ratio

        2. Continuous outcome: f = μ / σ²
           where μ = expected return, σ² = variance

        Args:
            expected_return: Expected return of the asset
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            variance: Variance of returns (for continuous formula)
            fractional: If True, apply fractional Kelly based on risk level

        Returns:
            Position size as fraction of portfolio (0-1)
        """
        # Input validation
        if win_rate <= 0 or win_rate >= 1:
            raise InvalidParameterError("Win rate must be between 0 and 1")

        if expected_return <= 0:
            logger.warning(f"Expected return {expected_return} is non-positive, returning 0")
            return 0.0

        # Choose Kelly formula
        if win_loss_ratio is not None:
            # Binary outcome Kelly
            # f = (p*b - q) / b where q = 1-p
            lose_rate = 1 - win_rate
            kelly_fraction = (win_rate * win_loss_ratio - lose_rate) / win_loss_ratio

        elif variance is not None and variance > 0:
            # Continuous outcome Kelly
            # f = μ / σ²
            kelly_fraction = expected_return / variance

        else:
            # Simplified approximation: f ≈ expected_return / volatility²
            # Assume volatility ≈ sqrt(expected_return) as rough estimate
            estimated_vol = np.sqrt(abs(expected_return))
            kelly_fraction = expected_return / (estimated_vol ** 2)

        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)

        # Apply fractional Kelly based on risk level
        if fractional:
            fraction_multiplier = KELLY_FRACTIONS[self.risk_level]
            kelly_fraction *= fraction_multiplier

        # Apply position size cap
        kelly_fraction = min(kelly_fraction, self.constraints.max_position_size)

        logger.debug(
            f"Kelly Criterion: expected_return={expected_return:.4f}, "
            f"win_rate={win_rate:.4f}, kelly={kelly_fraction:.4f}"
        )

        return kelly_fraction

    def kelly_portfolio(
        self,
        assets: List[AssetMetrics],
        fractional: bool = True
    ) -> PortfolioWeights:
        """
        Calculate Kelly-optimal portfolio weights

        Args:
            assets: List of asset metrics
            fractional: Apply fractional Kelly

        Returns:
            PortfolioWeights with optimal allocations
        """
        if not assets:
            raise InsufficientDataError("No assets provided")

        weights = {}

        for asset in assets:
            # Calculate Kelly fraction for each asset
            kelly_weight = self.kelly_criterion(
                expected_return=asset.expected_return,
                win_rate=asset.win_rate,
                variance=asset.volatility ** 2,
                fractional=fractional
            )
            weights[asset.symbol] = kelly_weight

        # Create portfolio weights
        portfolio = PortfolioWeights(
            weights=weights,
            method=PositionSizingMethod.FRACTIONAL_KELLY if fractional else PositionSizingMethod.KELLY_CRITERION
        )

        # Normalize if total weight exceeds 1
        if portfolio.total_weight > 1.0:
            portfolio.normalize()

        # Calculate expected portfolio metrics
        portfolio.expected_return = sum(
            asset.expected_return * weights.get(asset.symbol, 0)
            for asset in assets
        )

        return portfolio

    # ========================================================================
    # Risk Parity Methods
    # ========================================================================

    def risk_parity(
        self,
        assets: List[AssetMetrics],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PortfolioWeights:
        """
        Calculate Risk Parity portfolio weights

        Risk Parity allocates capital such that each asset contributes
        equally to the total portfolio risk. This is achieved by weighting
        inversely to volatility.

        Args:
            assets: List of asset metrics
            correlation_matrix: Asset correlation matrix (optional)

        Returns:
            PortfolioWeights with risk parity allocations
        """
        if not assets:
            raise InsufficientDataError("No assets provided")

        # Calculate inverse volatility weights
        weights = {}
        inverse_vols = {}

        for asset in assets:
            if asset.volatility <= 0:
                logger.warning(f"Asset {asset.symbol} has non-positive volatility, skipping")
                continue

            # Weight is inversely proportional to volatility
            inverse_vols[asset.symbol] = 1.0 / asset.volatility

        # Normalize to sum to 1
        total_inverse_vol = sum(inverse_vols.values())

        if total_inverse_vol == 0:
            raise OptimizationError("All assets have invalid volatility")

        for symbol, inv_vol in inverse_vols.items():
            weights[symbol] = inv_vol / total_inverse_vol

        # Create portfolio weights
        portfolio = PortfolioWeights(
            weights=weights,
            method=PositionSizingMethod.RISK_PARITY
        )

        # Apply max position size constraint
        portfolio.apply_max_position_size(self.constraints.max_position_size)

        # Calculate expected portfolio metrics
        portfolio.expected_return = sum(
            asset.expected_return * weights.get(asset.symbol, 0)
            for asset in assets
        )

        # Calculate expected volatility
        if correlation_matrix is not None:
            portfolio.expected_volatility = self._calculate_portfolio_volatility(
                weights, assets, correlation_matrix
            )
        else:
            # Simplified: assume zero correlation
            portfolio.expected_volatility = np.sqrt(sum(
                (weights.get(asset.symbol, 0) * asset.volatility) ** 2
                for asset in assets
            ))

        if portfolio.expected_volatility > 0:
            portfolio.sharpe_ratio = (
                (portfolio.expected_return - self.risk_free_rate) /
                portfolio.expected_volatility
            )

        return portfolio

    # ========================================================================
    # Equal Weight Methods
    # ========================================================================

    def equal_weight(
        self,
        assets: List[AssetMetrics],
        min_threshold: Optional[float] = None
    ) -> PortfolioWeights:
        """
        Calculate Equal Weight portfolio

        Allocates equal weight to all assets, optionally filtering out
        assets below a minimum expected return threshold.

        Args:
            assets: List of asset metrics
            min_threshold: Minimum expected return threshold (optional)

        Returns:
            PortfolioWeights with equal allocations
        """
        if not assets:
            raise InsufficientDataError("No assets provided")

        # Filter assets by minimum threshold if provided
        if min_threshold is not None:
            filtered_assets = [
                asset for asset in assets
                if asset.expected_return >= min_threshold
            ]

            if not filtered_assets:
                logger.warning(
                    f"No assets meet minimum threshold {min_threshold}, "
                    f"using all assets"
                )
                filtered_assets = assets
        else:
            filtered_assets = assets

        # Equal weight
        n = len(filtered_assets)
        equal_weight_value = 1.0 / n

        weights = {asset.symbol: equal_weight_value for asset in filtered_assets}

        # Apply max position size constraint
        equal_weight_value = min(equal_weight_value, self.constraints.max_position_size)

        # Create portfolio weights
        portfolio = PortfolioWeights(
            weights=weights,
            method=PositionSizingMethod.EQUAL_WEIGHT
        )

        # Calculate expected portfolio metrics
        portfolio.expected_return = sum(
            asset.expected_return * weights.get(asset.symbol, 0)
            for asset in filtered_assets
        )

        # Expected volatility (assume equal correlation)
        avg_volatility = np.mean([asset.volatility for asset in filtered_assets])
        portfolio.expected_volatility = avg_volatility / np.sqrt(n)

        if portfolio.expected_volatility > 0:
            portfolio.sharpe_ratio = (
                (portfolio.expected_return - self.risk_free_rate) /
                portfolio.expected_volatility
            )

        return portfolio

    # ========================================================================
    # Volatility-Based Methods
    # ========================================================================

    def volatility_weighted(
        self,
        assets: List[AssetMetrics],
        target_volatility: float = 0.15
    ) -> PortfolioWeights:
        """
        Calculate volatility-weighted portfolio

        Scales position sizes to achieve a target portfolio volatility,
        with individual positions weighted by expected return / volatility.

        Args:
            assets: List of asset metrics
            target_volatility: Target portfolio volatility (annualized)

        Returns:
            PortfolioWeights with volatility-adjusted allocations
        """
        if not assets:
            raise InsufficientDataError("No assets provided")

        weights = {}

        # Weight by Sharpe-like ratio (return / volatility)
        sharpe_scores = {}

        for asset in assets:
            if asset.volatility > 0:
                sharpe_scores[asset.symbol] = asset.expected_return / asset.volatility
            else:
                sharpe_scores[asset.symbol] = 0.0

        # Normalize Sharpe scores
        total_sharpe = sum(max(0, score) for score in sharpe_scores.values())

        if total_sharpe == 0:
            raise OptimizationError("All assets have non-positive risk-adjusted returns")

        for symbol, sharpe in sharpe_scores.items():
            weights[symbol] = max(0, sharpe) / total_sharpe

        # Create portfolio
        portfolio = PortfolioWeights(
            weights=weights,
            method=PositionSizingMethod.VOLATILITY_WEIGHTED
        )

        # Scale to target volatility
        current_volatility = sum(
            weights.get(asset.symbol, 0) * asset.volatility
            for asset in assets
        )

        if current_volatility > 0:
            scaling_factor = target_volatility / current_volatility
            # Apply scaling, but respect max position size
            for symbol in weights:
                weights[symbol] = min(
                    weights[symbol] * scaling_factor,
                    self.constraints.max_position_size
                )

        portfolio.weights = weights
        portfolio.normalize()

        # Calculate metrics
        portfolio.expected_return = sum(
            asset.expected_return * weights.get(asset.symbol, 0)
            for asset in assets
        )
        portfolio.expected_volatility = target_volatility

        if target_volatility > 0:
            portfolio.sharpe_ratio = (
                (portfolio.expected_return - self.risk_free_rate) /
                target_volatility
            )

        return portfolio

    # ========================================================================
    # Mean-Variance Optimization
    # ========================================================================

    def mean_variance_optimization(
        self,
        assets: List[AssetMetrics],
        correlation_matrix: pd.DataFrame,
        target_return: Optional[float] = None,
        objective: str = 'max_sharpe'
    ) -> PortfolioWeights:
        """
        Mean-Variance Optimization (Markowitz)

        Finds optimal portfolio weights that either:
        - Maximize Sharpe ratio (default)
        - Minimize variance for a target return
        - Maximize return for a target variance

        Args:
            assets: List of asset metrics
            correlation_matrix: Asset correlation matrix
            target_return: Target return (for min variance objective)
            objective: 'max_sharpe', 'min_variance', or 'max_return'

        Returns:
            PortfolioWeights with optimal allocations
        """
        if not assets:
            raise InsufficientDataError("No assets provided")

        if len(assets) < 2:
            raise InsufficientDataError("Need at least 2 assets for optimization")

        # Prepare data
        symbols = [asset.symbol for asset in assets]
        returns = np.array([asset.expected_return for asset in assets])
        volatilities = np.array([asset.volatility for asset in assets])

        # Build covariance matrix from correlation matrix and volatilities
        n = len(assets)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if symbols[i] in correlation_matrix.index and symbols[j] in correlation_matrix.columns:
                    corr = correlation_matrix.loc[symbols[i], symbols[j]]
                else:
                    corr = 0.0 if i != j else 1.0

                cov_matrix[i, j] = corr * volatilities[i] * volatilities[j]

        # Define objective functions
        def portfolio_return(weights):
            return np.dot(weights, returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 1e10

        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]

        if objective == 'min_variance' and target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
            )

        # Bounds: 0 <= weight <= max_position_size
        bounds = tuple(
            (0, self.constraints.max_position_size) for _ in range(n)
        )

        # Initial guess: equal weight
        x0 = np.array([1.0 / n] * n)

        # Optimize
        if objective == 'max_sharpe':
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            method = PositionSizingMethod.MAX_SHARPE

        elif objective == 'min_variance':
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            method = PositionSizingMethod.MIN_VARIANCE

        else:
            raise InvalidParameterError(f"Unknown objective: {objective}")

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Extract weights
        optimal_weights = result.x
        weights = {symbols[i]: optimal_weights[i] for i in range(n)}

        # Create portfolio
        portfolio = PortfolioWeights(
            weights=weights,
            method=method
        )

        # Calculate metrics
        portfolio.expected_return = portfolio_return(optimal_weights)
        portfolio.expected_volatility = portfolio_volatility(optimal_weights)

        if portfolio.expected_volatility > 0:
            portfolio.sharpe_ratio = (
                (portfolio.expected_return - self.risk_free_rate) /
                portfolio.expected_volatility
            )

        portfolio.metadata = {
            'optimization_success': result.success,
            'optimization_message': result.message,
            'objective': objective
        }

        return portfolio

    # ========================================================================
    # Risk Management Methods
    # ========================================================================

    def apply_risk_constraints(
        self,
        portfolio: PortfolioWeights,
        assets: List[AssetMetrics],
        current_portfolio_value: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PortfolioWeights:
        """
        Apply comprehensive risk management constraints

        Args:
            portfolio: Initial portfolio weights
            assets: List of asset metrics
            current_portfolio_value: Current total portfolio value
            correlation_matrix: Asset correlation matrix (optional)

        Returns:
            Risk-adjusted portfolio weights
        """
        weights = portfolio.weights.copy()

        # 1. Position size limits
        for symbol in weights:
            weights[symbol] = min(weights[symbol], self.constraints.max_position_size)

        # 2. Minimum positions constraint
        non_zero_positions = sum(1 for w in weights.values() if w > 0.001)
        if non_zero_positions < self.constraints.min_positions:
            logger.warning(
                f"Portfolio has {non_zero_positions} positions, "
                f"minimum is {self.constraints.min_positions}"
            )

        # 3. Maximum positions constraint
        if non_zero_positions > self.constraints.max_positions:
            # Keep only top N positions by weight
            sorted_weights = sorted(
                weights.items(), key=lambda x: x[1], reverse=True
            )
            new_weights = dict(sorted_weights[:self.constraints.max_positions])
            weights = new_weights

        # 4. Concentration limits (top 3 positions)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_3_concentration = sum(w for _, w in sorted_weights[:3])
        max_allowed = MAX_CONCENTRATION[self.risk_level]

        if top_3_concentration > max_allowed:
            # Scale down proportionally
            scale_factor = max_allowed / top_3_concentration
            weights = {symbol: w * scale_factor for symbol, w in weights.items()}

        # 5. Correlation constraints
        if correlation_matrix is not None:
            weights = self._apply_correlation_constraints(
                weights, correlation_matrix
            )

        # 6. Maximum portfolio volatility
        if self.constraints.max_portfolio_volatility is not None and correlation_matrix is not None:
            portfolio_vol = self._calculate_portfolio_volatility(
                weights, assets, correlation_matrix
            )

            if portfolio_vol > self.constraints.max_portfolio_volatility:
                # Scale down all positions proportionally
                scale_factor = self.constraints.max_portfolio_volatility / portfolio_vol
                weights = {symbol: w * scale_factor for symbol, w in weights.items()}

        # Create adjusted portfolio
        adjusted_portfolio = PortfolioWeights(
            weights=weights,
            method=portfolio.method
        )
        adjusted_portfolio.normalize()

        # Recalculate metrics
        adjusted_portfolio.expected_return = sum(
            asset.expected_return * weights.get(asset.symbol, 0)
            for asset in assets
        )

        if correlation_matrix is not None:
            adjusted_portfolio.expected_volatility = self._calculate_portfolio_volatility(
                weights, assets, correlation_matrix
            )

            if adjusted_portfolio.expected_volatility > 0:
                adjusted_portfolio.sharpe_ratio = (
                    (adjusted_portfolio.expected_return - self.risk_free_rate) /
                    adjusted_portfolio.expected_volatility
                )

        return adjusted_portfolio

    def _apply_correlation_constraints(
        self,
        weights: Dict[str, float],
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Reduce weights of highly correlated positions"""
        adjusted_weights = weights.copy()
        symbols = list(weights.keys())

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:
                    continue

                # Check correlation
                if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[symbol1, symbol2])

                    if corr > self.constraints.max_correlation:
                        # Reduce weight of smaller position
                        if weights[symbol1] < weights[symbol2]:
                            reduction = (corr - self.constraints.max_correlation) * 0.5
                            adjusted_weights[symbol1] *= (1 - reduction)
                        else:
                            reduction = (corr - self.constraints.max_correlation) * 0.5
                            adjusted_weights[symbol2] *= (1 - reduction)

        return adjusted_weights

    def _calculate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        assets: List[AssetMetrics],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility from weights and correlation matrix"""
        symbols = [asset.symbol for asset in assets]
        weight_array = np.array([weights.get(symbol, 0) for symbol in symbols])
        vol_array = np.array([asset.volatility for asset in assets])

        # Build covariance matrix
        n = len(assets)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if symbols[i] in correlation_matrix.index and symbols[j] in correlation_matrix.columns:
                    corr = correlation_matrix.loc[symbols[i], symbols[j]]
                else:
                    corr = 0.0 if i != j else 1.0

                cov_matrix[i, j] = corr * vol_array[i] * vol_array[j]

        # Portfolio variance
        portfolio_var = np.dot(weight_array, np.dot(cov_matrix, weight_array))

        return np.sqrt(portfolio_var)

    def calculate_position_sizes(
        self,
        portfolio_weights: PortfolioWeights,
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Convert portfolio weights to actual share quantities

        Args:
            portfolio_weights: Portfolio allocation weights
            portfolio_value: Total portfolio value
            current_prices: Current prices for each asset

        Returns:
            Dictionary mapping symbols to number of shares to purchase
        """
        positions = {}

        for symbol, weight in portfolio_weights.weights.items():
            if symbol not in current_prices:
                logger.warning(f"No price available for {symbol}, skipping")
                continue

            # Calculate dollar amount to invest
            dollar_amount = portfolio_value * weight

            # Calculate number of shares (integer)
            price = current_prices[symbol]
            shares = int(dollar_amount / price)

            positions[symbol] = shares

        return positions

    # ========================================================================
    # Rebalancing Methods
    # ========================================================================

    def calculate_rebalancing_trades(
        self,
        target_weights: PortfolioWeights,
        current_positions: Dict[str, float],  # Current shares
        current_prices: Dict[str, float],
        portfolio_value: float,
        threshold: float = 0.05
    ) -> Dict[str, int]:
        """
        Calculate trades needed to rebalance portfolio

        Args:
            target_weights: Target portfolio weights
            current_positions: Current positions (shares)
            current_prices: Current prices
            portfolio_value: Total portfolio value
            threshold: Minimum deviation to trigger rebalance (0.05 = 5%)

        Returns:
            Dictionary of trades (positive = buy, negative = sell)
        """
        trades = {}

        # Calculate current weights
        current_values = {
            symbol: current_positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in set(list(current_positions.keys()) + list(target_weights.weights.keys()))
        }

        total_value = sum(current_values.values())
        current_weights = {
            symbol: value / total_value if total_value > 0 else 0
            for symbol, value in current_values.items()
        }

        # Calculate trades for each position
        for symbol in set(list(current_weights.keys()) + list(target_weights.weights.keys())):
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.weights.get(symbol, 0)

            weight_diff = target_weight - current_weight

            # Only trade if deviation exceeds threshold
            if abs(weight_diff) < threshold:
                continue

            # Calculate dollar amount to trade
            dollar_amount = weight_diff * portfolio_value

            # Convert to shares
            price = current_prices.get(symbol, 0)
            if price > 0:
                shares_to_trade = int(dollar_amount / price)
                if shares_to_trade != 0:
                    trades[symbol] = shares_to_trade

        return trades


# ============================================================================
# Convenience Functions
# ============================================================================

def create_optimizer(
    risk_level: str = "MODERATE",
    max_position_size: float = 0.20,
    risk_free_rate: float = 0.0
) -> PortfolioOptimizer:
    """
    Create a portfolio optimizer with common settings

    Args:
        risk_level: Risk tolerance ('CONSERVATIVE', 'MODERATE', 'AGGRESSIVE', 'VERY_AGGRESSIVE')
        max_position_size: Maximum size per position
        risk_free_rate: Risk-free rate

    Returns:
        PortfolioOptimizer instance
    """
    risk_enum = RiskLevel(risk_level)
    constraints = RiskConstraints(max_position_size=max_position_size)

    return PortfolioOptimizer(
        risk_level=risk_enum,
        risk_free_rate=risk_free_rate,
        constraints=constraints
    )


def optimize_portfolio(
    assets: List[AssetMetrics],
    method: str = "FRACTIONAL_KELLY",
    risk_level: str = "MODERATE",
    correlation_matrix: Optional[pd.DataFrame] = None,
    **kwargs
) -> PortfolioWeights:
    """
    High-level portfolio optimization function

    Args:
        assets: List of asset metrics
        method: Optimization method
        risk_level: Risk tolerance level
        correlation_matrix: Asset correlation matrix
        **kwargs: Additional parameters for specific methods

    Returns:
        PortfolioWeights with optimal allocations
    """
    optimizer = create_optimizer(risk_level=risk_level)
    method_enum = PositionSizingMethod(method)

    if method_enum == PositionSizingMethod.KELLY_CRITERION:
        return optimizer.kelly_portfolio(assets, fractional=False)

    elif method_enum == PositionSizingMethod.FRACTIONAL_KELLY:
        return optimizer.kelly_portfolio(assets, fractional=True)

    elif method_enum == PositionSizingMethod.RISK_PARITY:
        return optimizer.risk_parity(assets, correlation_matrix)

    elif method_enum == PositionSizingMethod.EQUAL_WEIGHT:
        return optimizer.equal_weight(assets, kwargs.get('min_threshold'))

    elif method_enum == PositionSizingMethod.VOLATILITY_WEIGHTED:
        return optimizer.volatility_weighted(
            assets, kwargs.get('target_volatility', 0.15)
        )

    elif method_enum in [PositionSizingMethod.MAX_SHARPE, PositionSizingMethod.MIN_VARIANCE]:
        if correlation_matrix is None:
            raise InvalidParameterError(
                f"{method} requires correlation_matrix"
            )

        objective = 'max_sharpe' if method_enum == PositionSizingMethod.MAX_SHARPE else 'min_variance'
        return optimizer.mean_variance_optimization(
            assets, correlation_matrix, objective=objective
        )

    else:
        raise InvalidParameterError(f"Unknown method: {method}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BIST Portfolio Optimization - Example Usage")
    print("=" * 80)

    # Create sample assets
    assets = [
        AssetMetrics(
            symbol="THYAO",
            expected_return=0.15,
            volatility=0.25,
            win_rate=0.60,
            current_price=250.0
        ),
        AssetMetrics(
            symbol="GARAN",
            expected_return=0.12,
            volatility=0.20,
            win_rate=0.55,
            current_price=85.0
        ),
        AssetMetrics(
            symbol="AKBNK",
            expected_return=0.10,
            volatility=0.18,
            win_rate=0.58,
            current_price=45.0
        ),
        AssetMetrics(
            symbol="EREGL",
            expected_return=0.08,
            volatility=0.22,
            win_rate=0.52,
            current_price=35.0
        )
    ]

    # Create correlation matrix
    correlation_matrix = pd.DataFrame(
        [
            [1.00, 0.40, 0.45, 0.30],
            [0.40, 1.00, 0.60, 0.35],
            [0.45, 0.60, 1.00, 0.25],
            [0.30, 0.35, 0.25, 1.00]
        ],
        index=["THYAO", "GARAN", "AKBNK", "EREGL"],
        columns=["THYAO", "GARAN", "AKBNK", "EREGL"]
    )

    # Create optimizer
    optimizer = create_optimizer(
        risk_level="MODERATE",
        max_position_size=0.30
    )

    print("\n1. Kelly Criterion (Fractional)")
    print("-" * 80)
    kelly_portfolio = optimizer.kelly_portfolio(assets, fractional=True)
    print(f"Method: {kelly_portfolio.method.value}")
    print(f"Expected Return: {kelly_portfolio.expected_return:.2%}")
    print("Weights:")
    for symbol, weight in kelly_portfolio.weights.items():
        print(f"  {symbol}: {weight:.2%}")

    print("\n2. Risk Parity")
    print("-" * 80)
    risk_parity_portfolio = optimizer.risk_parity(assets, correlation_matrix)
    print(f"Method: {risk_parity_portfolio.method.value}")
    print(f"Expected Return: {risk_parity_portfolio.expected_return:.2%}")
    print(f"Expected Volatility: {risk_parity_portfolio.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {risk_parity_portfolio.sharpe_ratio:.2f}")
    print("Weights:")
    for symbol, weight in risk_parity_portfolio.weights.items():
        print(f"  {symbol}: {weight:.2%}")

    print("\n3. Equal Weight")
    print("-" * 80)
    equal_weight_portfolio = optimizer.equal_weight(assets)
    print(f"Method: {equal_weight_portfolio.method.value}")
    print(f"Expected Return: {equal_weight_portfolio.expected_return:.2%}")
    print(f"Expected Volatility: {equal_weight_portfolio.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {equal_weight_portfolio.sharpe_ratio:.2f}")
    print("Weights:")
    for symbol, weight in equal_weight_portfolio.weights.items():
        print(f"  {symbol}: {weight:.2%}")

    print("\n4. Maximum Sharpe Ratio")
    print("-" * 80)
    max_sharpe_portfolio = optimizer.mean_variance_optimization(
        assets, correlation_matrix, objective='max_sharpe'
    )
    print(f"Method: {max_sharpe_portfolio.method.value}")
    print(f"Expected Return: {max_sharpe_portfolio.expected_return:.2%}")
    print(f"Expected Volatility: {max_sharpe_portfolio.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio.sharpe_ratio:.2f}")
    print("Weights:")
    for symbol, weight in max_sharpe_portfolio.weights.items():
        print(f"  {symbol}: {weight:.2%}")

    print("\n5. Apply Risk Constraints")
    print("-" * 80)
    constrained_portfolio = optimizer.apply_risk_constraints(
        max_sharpe_portfolio,
        assets,
        current_portfolio_value=100000.0,
        correlation_matrix=correlation_matrix
    )
    print(f"Constrained Expected Return: {constrained_portfolio.expected_return:.2%}")
    print(f"Constrained Volatility: {constrained_portfolio.expected_volatility:.2%}")
    print("Constrained Weights:")
    for symbol, weight in constrained_portfolio.weights.items():
        print(f"  {symbol}: {weight:.2%}")

    print("\n6. Calculate Position Sizes")
    print("-" * 80)
    current_prices = {
        "THYAO": 250.0,
        "GARAN": 85.0,
        "AKBNK": 45.0,
        "EREGL": 35.0
    }

    positions = optimizer.calculate_position_sizes(
        constrained_portfolio,
        portfolio_value=100000.0,
        current_prices=current_prices
    )

    print("Recommended Positions:")
    for symbol, shares in positions.items():
        price = current_prices[symbol]
        value = shares * price
        print(f"  {symbol}: {shares} shares @ {price:.2f} TRY = {value:.2f} TRY")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
