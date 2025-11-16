"""
Portfolio Optimization Integration Example

This example demonstrates how to integrate the portfolio optimization module
with the portfolio manager to create an intelligent trading system.

The workflow includes:
1. Creating asset metrics from predictions
2. Optimizing portfolio weights using various strategies
3. Calculating position sizes
4. Executing trades via portfolio manager
5. Applying risk management constraints
6. Rebalancing the portfolio

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime

# Import portfolio management modules
from src.portfolio import (
    PortfolioManager,
    PortfolioOptimizer,
    AssetMetrics,
    PortfolioWeights,
    RiskConstraints,
    RiskLevel,
    PositionSizingMethod,
    create_optimizer,
    optimize_portfolio
)


def create_sample_asset_metrics():
    """Create sample asset metrics from model predictions"""
    # In a real scenario, these would come from your ML models
    assets = [
        AssetMetrics(
            symbol="THYAO",
            expected_return=0.15,      # 15% expected annual return
            volatility=0.25,            # 25% annual volatility
            win_rate=0.60,              # 60% win rate from backtesting
            current_price=250.0,
            max_drawdown=0.20,
            correlation_to_market=0.75
        ),
        AssetMetrics(
            symbol="GARAN",
            expected_return=0.12,
            volatility=0.20,
            win_rate=0.55,
            current_price=85.0,
            max_drawdown=0.18,
            correlation_to_market=0.80
        ),
        AssetMetrics(
            symbol="AKBNK",
            expected_return=0.10,
            volatility=0.18,
            win_rate=0.58,
            current_price=45.0,
            max_drawdown=0.15,
            correlation_to_market=0.85
        ),
        AssetMetrics(
            symbol="EREGL",
            expected_return=0.08,
            volatility=0.22,
            win_rate=0.52,
            current_price=35.0,
            max_drawdown=0.22,
            correlation_to_market=0.60
        ),
        AssetMetrics(
            symbol="PETKM",
            expected_return=0.18,
            volatility=0.30,
            win_rate=0.62,
            current_price=120.0,
            max_drawdown=0.25,
            correlation_to_market=0.55
        )
    ]
    return assets


def create_correlation_matrix(assets):
    """Create correlation matrix for assets"""
    symbols = [asset.symbol for asset in assets]

    # Sample correlation matrix (would be calculated from historical data)
    corr_data = pd.DataFrame(
        [
            [1.00, 0.40, 0.45, 0.30, 0.35],
            [0.40, 1.00, 0.60, 0.35, 0.25],
            [0.45, 0.60, 1.00, 0.25, 0.30],
            [0.30, 0.35, 0.25, 1.00, 0.50],
            [0.35, 0.25, 0.30, 0.50, 1.00]
        ],
        index=symbols,
        columns=symbols
    )

    return corr_data


def example_1_kelly_criterion():
    """Example 1: Kelly Criterion Position Sizing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Kelly Criterion Position Sizing")
    print("=" * 80)

    # Create optimizer with moderate risk
    optimizer = create_optimizer(
        risk_level="MODERATE",
        max_position_size=0.20
    )

    # Create sample assets
    assets = create_sample_asset_metrics()

    # Calculate Kelly-optimal portfolio
    kelly_portfolio = optimizer.kelly_portfolio(assets, fractional=True)

    print("\nKelly Criterion Results (Fractional - 50%):")
    print("-" * 80)
    print(f"Expected Return: {kelly_portfolio.expected_return:.2%}")
    print(f"\nPosition Weights:")
    for symbol, weight in sorted(kelly_portfolio.weights.items(), key=lambda x: x[1], reverse=True):
        asset = next(a for a in assets if a.symbol == symbol)
        print(f"  {symbol:8s}: {weight:6.2%} (Expected Return: {asset.expected_return:.2%}, Vol: {asset.volatility:.2%})")

    return kelly_portfolio, assets


def example_2_risk_parity():
    """Example 2: Risk Parity Allocation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Risk Parity Allocation")
    print("=" * 80)

    # Create optimizer
    optimizer = create_optimizer(risk_level="MODERATE")

    # Create assets and correlation matrix
    assets = create_sample_asset_metrics()
    correlation_matrix = create_correlation_matrix(assets)

    # Calculate risk parity portfolio
    risk_parity = optimizer.risk_parity(assets, correlation_matrix)

    print("\nRisk Parity Results:")
    print("-" * 80)
    print(f"Expected Return: {risk_parity.expected_return:.2%}")
    print(f"Expected Volatility: {risk_parity.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {risk_parity.sharpe_ratio:.2f}")
    print(f"\nPosition Weights:")
    for symbol, weight in sorted(risk_parity.weights.items(), key=lambda x: x[1], reverse=True):
        asset = next(a for a in assets if a.symbol == symbol)
        print(f"  {symbol:8s}: {weight:6.2%} (Volatility: {asset.volatility:.2%})")

    return risk_parity, assets


def example_3_compare_strategies():
    """Example 3: Compare Different Optimization Strategies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Compare Optimization Strategies")
    print("=" * 80)

    # Create assets and correlation matrix
    assets = create_sample_asset_metrics()
    correlation_matrix = create_correlation_matrix(assets)

    # Test different strategies
    strategies = {
        "Kelly Criterion": PositionSizingMethod.FRACTIONAL_KELLY,
        "Risk Parity": PositionSizingMethod.RISK_PARITY,
        "Equal Weight": PositionSizingMethod.EQUAL_WEIGHT,
        "Max Sharpe": PositionSizingMethod.MAX_SHARPE
    }

    results = []

    for name, method in strategies.items():
        try:
            if method in [PositionSizingMethod.MAX_SHARPE]:
                portfolio = optimize_portfolio(
                    assets,
                    method=method.value,
                    risk_level="MODERATE",
                    correlation_matrix=correlation_matrix
                )
            else:
                portfolio = optimize_portfolio(
                    assets,
                    method=method.value,
                    risk_level="MODERATE",
                    correlation_matrix=correlation_matrix
                )

            results.append({
                'Strategy': name,
                'Expected Return': portfolio.expected_return or 0,
                'Expected Vol': portfolio.expected_volatility or 0,
                'Sharpe Ratio': portfolio.sharpe_ratio or 0,
                'Top 3 Concentration': sum(sorted(portfolio.weights.values(), reverse=True)[:3])
            })

        except Exception as e:
            print(f"Error with {name}: {e}")

    # Display comparison
    df = pd.DataFrame(results)
    print("\nStrategy Comparison:")
    print("-" * 80)
    print(df.to_string(index=False))

    # Find best Sharpe ratio
    best_sharpe = df.loc[df['Sharpe Ratio'].idxmax()]
    print(f"\nBest Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")


def example_4_portfolio_integration():
    """Example 4: Integration with Portfolio Manager"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Integration with Portfolio Manager")
    print("=" * 80)

    # 1. Create portfolio manager
    portfolio_manager = PortfolioManager(
        name="BIST Optimized Portfolio",
        initial_cash=100000.0,  # 100,000 TRY
        currency="TRY"
    )

    print(f"\nInitial Portfolio:")
    print(f"  Cash: {portfolio_manager.cash:,.2f} TRY")

    # 2. Optimize portfolio weights
    assets = create_sample_asset_metrics()
    correlation_matrix = create_correlation_matrix(assets)

    optimizer = create_optimizer(
        risk_level="MODERATE",
        max_position_size=0.25
    )

    # Use Max Sharpe strategy
    optimal_weights = optimizer.mean_variance_optimization(
        assets,
        correlation_matrix,
        objective='max_sharpe'
    )

    # 3. Apply risk constraints
    constrained_weights = optimizer.apply_risk_constraints(
        optimal_weights,
        assets,
        current_portfolio_value=portfolio_manager.cash,
        correlation_matrix=correlation_matrix
    )

    print(f"\nOptimal Portfolio Weights (Max Sharpe with Constraints):")
    print("-" * 80)
    for symbol, weight in sorted(constrained_weights.weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight:.2%}")

    # 4. Calculate position sizes
    current_prices = {asset.symbol: asset.current_price for asset in assets}
    positions = optimizer.calculate_position_sizes(
        constrained_weights,
        portfolio_value=portfolio_manager.cash,
        current_prices=current_prices
    )

    print(f"\nRecommended Positions:")
    print("-" * 80)
    for symbol, shares in sorted(positions.items(), key=lambda x: x[1] * current_prices[x[0]], reverse=True):
        price = current_prices[symbol]
        value = shares * price
        pct = value / portfolio_manager.cash
        print(f"  {symbol}: {shares:4d} shares @ {price:7.2f} TRY = {value:10,.2f} TRY ({pct:5.1%})")

    # 5. Execute trades
    print(f"\nExecuting Trades:")
    print("-" * 80)

    total_invested = 0
    for symbol, shares in positions.items():
        if shares > 0:
            price = current_prices[symbol]
            # Apply 0.2% commission
            commission = shares * price * 0.002

            portfolio_manager.buy(
                symbol=symbol,
                shares=shares,
                price=price,
                commission=commission,
                notes="Initial allocation based on Max Sharpe optimization"
            )

            total_invested += shares * price + commission
            print(f"  BUY {shares} shares of {symbol} @ {price:.2f} TRY")

    # 6. Portfolio summary
    print(f"\nPortfolio Summary:")
    print("-" * 80)
    summary = portfolio_manager.get_portfolio_summary(current_prices)
    print(f"  Total Value: {summary['total_value']:,.2f} TRY")
    print(f"  Cash: {summary['cash']:,.2f} TRY")
    print(f"  Invested: {summary['positions_value']:,.2f} TRY")
    print(f"  Positions: {summary['num_positions']}")
    print(f"  Commissions: {summary['total_commissions']:,.2f} TRY")

    print(f"\nCurrent Positions:")
    for pos in summary['positions']:
        print(f"  {pos['symbol']:8s}: {pos['shares']:4.0f} shares, "
              f"Value: {pos['market_value']:10,.2f} TRY ({pos['market_value']/summary['total_value']*100:5.1f}%)")

    return portfolio_manager, optimizer, assets


def example_5_risk_management():
    """Example 5: Risk Management Rules"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Risk Management Rules")
    print("=" * 80)

    # Create assets with varying risk profiles
    assets = create_sample_asset_metrics()
    correlation_matrix = create_correlation_matrix(assets)

    # Test different risk levels
    risk_levels = [RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE]

    print("\nPosition Sizing by Risk Level:")
    print("-" * 80)

    for risk_level in risk_levels:
        optimizer = PortfolioOptimizer(risk_level=risk_level)

        # Use Kelly Criterion
        portfolio = optimizer.kelly_portfolio(assets, fractional=True)

        # Get top position
        top_symbol = max(portfolio.weights.items(), key=lambda x: x[1])

        print(f"\n{risk_level.value}:")
        print(f"  Max Position Size: {optimizer.constraints.max_position_size:.1%}")
        print(f"  Kelly Fraction: {KELLY_FRACTIONS[risk_level]:.1%}")
        print(f"  Top Position: {top_symbol[0]} = {top_symbol[1]:.2%}")
        print(f"  Top 3 Concentration Limit: {MAX_CONCENTRATION[risk_level]:.1%}")


def example_6_rebalancing():
    """Example 6: Portfolio Rebalancing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Portfolio Rebalancing")
    print("=" * 80)

    # Create portfolio manager with existing positions
    portfolio_manager = PortfolioManager(
        name="Existing Portfolio",
        initial_cash=100000.0,
        currency="TRY"
    )

    # Simulate existing positions
    current_prices = {
        "THYAO": 250.0,
        "GARAN": 85.0,
        "AKBNK": 45.0,
        "EREGL": 35.0
    }

    # Buy initial positions (equal weight)
    for symbol, price in current_prices.items():
        shares = int(25000 / price)  # 25% each
        portfolio_manager.buy(symbol, shares, price, commission=shares * price * 0.002)

    print("\nCurrent Portfolio:")
    print("-" * 80)
    current_positions = {
        pos.symbol: pos.shares
        for pos in portfolio_manager.get_all_positions().values()
    }

    for symbol, shares in current_positions.items():
        value = shares * current_prices[symbol]
        print(f"  {symbol}: {shares:4.0f} shares = {value:10,.2f} TRY")

    # Calculate new optimal weights
    assets = create_sample_asset_metrics()
    correlation_matrix = create_correlation_matrix(assets)

    optimizer = create_optimizer(risk_level="MODERATE")
    optimal_weights = optimizer.mean_variance_optimization(
        assets,
        correlation_matrix,
        objective='max_sharpe'
    )

    # Calculate rebalancing trades
    portfolio_value = portfolio_manager.get_portfolio_value(current_prices)
    trades = optimizer.calculate_rebalancing_trades(
        target_weights=optimal_weights,
        current_positions=current_positions,
        current_prices=current_prices,
        portfolio_value=portfolio_value,
        threshold=0.05  # 5% threshold
    )

    print(f"\nRebalancing Trades (5% threshold):")
    print("-" * 80)

    if not trades:
        print("  No rebalancing needed - portfolio is within tolerance")
    else:
        for symbol, shares_to_trade in trades.items():
            if shares_to_trade > 0:
                value = shares_to_trade * current_prices[symbol]
                print(f"  BUY  {abs(shares_to_trade):4d} shares of {symbol} (${value:10,.2f})")
            else:
                value = abs(shares_to_trade) * current_prices[symbol]
                print(f"  SELL {abs(shares_to_trade):4d} shares of {symbol} (${value:10,.2f})")


def main():
    """Run all examples"""
    print("\n")
    print("=" * 80)
    print("BIST AI Trading System - Portfolio Optimization Examples")
    print("=" * 80)

    try:
        # Run examples
        example_1_kelly_criterion()
        example_2_risk_parity()
        example_3_compare_strategies()
        example_4_portfolio_integration()
        example_5_risk_management()
        example_6_rebalancing()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
