"""
Example: Using Backtesting Metrics Module

This example demonstrates how to use the comprehensive metrics module
to calculate trading strategy performance metrics.

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.metrics import (
    calculate_all_metrics,
    calculate_win_rate,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    compare_strategies,
    PerformanceMetrics
)


def example_1_basic_metrics():
    """Example 1: Calculate basic metrics from trade returns"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Metrics Calculation")
    print("=" * 80)

    # Sample trade returns (as decimals, e.g., 0.02 = 2% return)
    returns = np.array([
        0.025, -0.010, 0.030, -0.015, 0.020,
        0.015, -0.005, 0.018, -0.012, 0.022,
        -0.008, 0.028, 0.012, -0.020, 0.035
    ])

    print(f"\nTotal trades: {len(returns)}")
    print(f"Sample returns: {returns[:5]}")

    # Calculate individual metrics
    win_rate = calculate_win_rate(returns)
    print(f"\nWin Rate: {win_rate:.2f}%")

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    print(f"Sharpe Ratio: {sharpe:.3f}")

    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
    print(f"Sortino Ratio: {sortino:.3f}")

    profit_factor = calculate_profit_factor(returns)
    print(f"Profit Factor: {profit_factor:.3f}")


def example_2_comprehensive_analysis():
    """Example 2: Comprehensive performance analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Comprehensive Performance Analysis")
    print("=" * 80)

    # Generate realistic trading returns
    np.random.seed(42)
    n_trades = 100

    # Simulate a strategy with 65% win rate
    winning_returns = np.random.normal(0.025, 0.015, 65)
    losing_returns = np.random.normal(-0.018, 0.012, 35)
    returns = np.concatenate([winning_returns, losing_returns])
    np.random.shuffle(returns)

    # Calculate all metrics at once
    metrics = calculate_all_metrics(
        returns=returns,
        risk_free_rate=0.02,  # 2% annual risk-free rate
        periods_per_year=252,  # Daily trading
        initial_capital=100000.0
    )

    # Display comprehensive results
    print(metrics)

    # Access individual metrics
    print("\nKey Performance Indicators:")
    print(f"  Total Return: {metrics.total_return_pct:.2f}%")
    print(f"  Win Rate: {metrics.win_rate:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.3f}")


def example_3_equity_curve_analysis():
    """Example 3: Analyzing equity curve and drawdowns"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Equity Curve Analysis")
    print("=" * 80)

    # Simulate returns over time
    np.random.seed(123)
    daily_returns = np.random.normal(0.001, 0.015, 252)  # 1 year of daily returns

    # Calculate equity curve
    initial_capital = 100000
    equity_curve = initial_capital * (1 + pd.Series(daily_returns)).cumprod()

    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {((equity_curve.iloc[-1] / initial_capital) - 1) * 100:.2f}%")

    # Calculate maximum drawdown with drawdown series
    max_dd, dd_series = calculate_max_drawdown(equity_curve, return_series=True)

    print(f"\nMaximum Drawdown: {max_dd * 100:.2f}%")
    print(f"Worst drawdown day: {dd_series.idxmin()}")
    print(f"Drawdown at that point: {dd_series.min() * 100:.2f}%")

    # Calculate metrics
    metrics = calculate_all_metrics(
        returns=daily_returns,
        equity_curve=equity_curve,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    print(f"\nAnnualized Return: {metrics.annualized_return * 100:.2f}%")
    print(f"Annualized Volatility: {metrics.annualized_volatility * 100:.2f}%")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"Recovery Factor: {metrics.recovery_factor:.3f}")


def example_4_strategy_comparison():
    """Example 4: Comparing multiple strategies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Strategy Comparison")
    print("=" * 80)

    np.random.seed(42)
    n_trades = 100

    # Strategy 1: Aggressive (higher returns, higher volatility)
    strategy_1_returns = np.random.normal(0.02, 0.025, n_trades)

    # Strategy 2: Conservative (lower returns, lower volatility)
    strategy_2_returns = np.random.normal(0.012, 0.012, n_trades)

    # Strategy 3: Balanced
    strategy_3_returns = np.random.normal(0.015, 0.018, n_trades)

    # Compare strategies
    comparison = compare_strategies(
        {
            'Aggressive': strategy_1_returns,
            'Conservative': strategy_2_returns,
            'Balanced': strategy_3_returns
        },
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Display key metrics
    print("\nStrategy Comparison:")
    print("-" * 80)
    key_metrics = [
        'total_return_pct',
        'win_rate',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown_pct',
        'profit_factor',
        'calmar_ratio'
    ]

    print(comparison[key_metrics].round(3).to_string())

    # Determine best strategy by Sharpe ratio
    best_strategy = comparison['sharpe_ratio'].idxmax()
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy}")
    print(f"Sharpe Ratio: {comparison.loc[best_strategy, 'sharpe_ratio']:.3f}")


def example_5_risk_metrics():
    """Example 5: Focus on risk metrics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Risk Metrics Analysis")
    print("=" * 80)

    # Simulate a strategy with occasional large losses
    np.random.seed(456)
    returns = []

    # 80 small wins
    returns.extend(np.random.normal(0.01, 0.005, 80).tolist())

    # 15 small losses
    returns.extend(np.random.normal(-0.008, 0.004, 15).tolist())

    # 5 large losses (tail risk)
    returns.extend(np.random.normal(-0.05, 0.015, 5).tolist())

    returns = np.array(returns)
    np.random.shuffle(returns)

    # Calculate metrics
    metrics = calculate_all_metrics(
        returns=returns,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    print("\nRisk Analysis:")
    print("-" * 80)
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print(f"\nReturn Metrics:")
    print(f"  Average Win: {metrics.avg_winning_trade * 100:.2f}%")
    print(f"  Average Loss: {metrics.avg_losing_trade * 100:.2f}%")
    print(f"  Largest Win: {metrics.largest_win * 100:.2f}%")
    print(f"  Largest Loss: {metrics.largest_loss * 100:.2f}%")
    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"  (Sortino > Sharpe indicates asymmetric risk)")
    print(f"\nDrawdown Analysis:")
    print(f"  Maximum Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"  Recovery Factor: {metrics.recovery_factor:.3f}")
    print(f"\nOther Metrics:")
    print(f"  Risk/Reward Ratio: {metrics.risk_reward_ratio:.3f}")
    print(f"  Expectancy: {metrics.expectancy * 100:.3f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.3f}")


def example_6_export_metrics():
    """Example 6: Exporting metrics to different formats"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Exporting Metrics")
    print("=" * 80)

    # Generate sample data
    np.random.seed(42)
    returns = np.random.normal(0.015, 0.02, 100)

    # Calculate metrics
    metrics = calculate_all_metrics(
        returns=returns,
        risk_free_rate=0.02,
        periods_per_year=252
    )

    # Export to dictionary
    metrics_dict = metrics.to_dict()
    print("\nMetrics as Dictionary:")
    print(f"  Keys: {list(metrics_dict.keys())[:5]}... (total: {len(metrics_dict)})")

    # Export to DataFrame
    metrics_df = metrics.to_dataframe()
    print("\nMetrics as DataFrame (first 10 rows):")
    print(metrics_df.head(10).to_string())

    # Could save to CSV
    # metrics_df.to_csv('backtest_metrics.csv')
    print("\n(Metrics can be saved to CSV with: metrics_df.to_csv('metrics.csv'))")


def main():
    """Run all examples"""
    print("\n")
    print("*" * 80)
    print("BACKTESTING METRICS MODULE - COMPREHENSIVE EXAMPLES")
    print("*" * 80)

    # Run all examples
    example_1_basic_metrics()
    example_2_comprehensive_analysis()
    example_3_equity_curve_analysis()
    example_4_strategy_comparison()
    example_5_risk_metrics()
    example_6_export_metrics()

    print("\n" + "*" * 80)
    print("All examples completed successfully!")
    print("*" * 80)
    print("\nFor more information, see:")
    print("  - Module documentation: src/backtesting/metrics.py")
    print("  - API reference: Use help(calculate_all_metrics)")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
