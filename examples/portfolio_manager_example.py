"""
Portfolio Manager Example - Complete Usage Guide

This example demonstrates all key features of the PortfolioManager:
1. Creating and initializing portfolios
2. Executing buy and sell transactions
3. Tracking positions and cost basis
4. Calculating P&L (realized and unrealized)
5. Portfolio analytics and metrics
6. Saving and loading portfolios
7. Different cost basis methods (FIFO, LIFO, Average)

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.portfolio.manager import (
    PortfolioManager,
    CostBasisMethod,
    TransactionType,
    create_portfolio
)
from datetime import datetime, timedelta
import json


def example_1_basic_portfolio():
    """Example 1: Basic portfolio operations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Portfolio Operations")
    print("=" * 80)

    # Create a portfolio with 100,000 TRY
    portfolio = PortfolioManager(
        name="BIST Growth Portfolio",
        initial_cash=100000.0,
        cost_basis_method=CostBasisMethod.AVERAGE,
        currency="TRY"
    )

    print(f"\nInitial Portfolio:")
    print(f"  Name: {portfolio.name}")
    print(f"  Cash: {portfolio.cash:,.2f} {portfolio.currency}")
    print(f"  Positions: {len(portfolio.positions)}")

    # Buy some stocks
    print("\n--- Executing Buy Orders ---")

    portfolio.buy(
        symbol="THYAO",
        shares=100,
        price=250.50,
        commission=12.50,
        notes="Turkish Airlines - Strong fundamentals"
    )
    print(f"Bought 100 THYAO @ 250.50 TRY")

    portfolio.buy(
        symbol="GARAN",
        shares=200,
        price=85.75,
        commission=17.15,
        notes="Garanti Bank - Banking sector play"
    )
    print(f"Bought 200 GARAN @ 85.75 TRY")

    portfolio.buy(
        symbol="EREGL",
        shares=150,
        price=42.30,
        commission=6.35,
        notes="Erdemir - Steel sector"
    )
    print(f"Bought 150 EREGL @ 42.30 TRY")

    # View current positions
    print("\n--- Current Positions ---")
    for symbol, position in portfolio.get_all_positions().items():
        print(f"{symbol}:")
        print(f"  Shares: {position.shares:,.2f}")
        print(f"  Cost Basis: {position.cost_basis:,.2f} TRY")
        print(f"  Total Cost: {position.total_cost:,.2f} TRY")

    print(f"\nRemaining Cash: {portfolio.cash:,.2f} TRY")
    print(f"Total Commissions Paid: {portfolio.total_commissions:,.2f} TRY")

    return portfolio


def example_2_selling_and_pnl(portfolio):
    """Example 2: Selling positions and calculating P&L"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Selling Positions and P&L Calculation")
    print("=" * 80)

    # Current market prices (simulated)
    current_prices = {
        "THYAO": 265.80,  # Up from 250.50
        "GARAN": 88.40,   # Up from 85.75
        "EREGL": 41.20    # Down from 42.30
    }

    # Get portfolio summary before selling
    print("\n--- Portfolio Value Before Selling ---")
    summary = portfolio.get_portfolio_summary(current_prices)

    print(f"Total Portfolio Value: {summary['total_value']:,.2f} TRY")
    print(f"Positions Value: {summary['positions_value']:,.2f} TRY")
    print(f"Cash: {summary['cash']:,.2f} TRY")
    print(f"Unrealized P&L: {summary['unrealized_pnl']:,.2f} TRY ({summary['unrealized_pnl'] / summary['total_cost_basis'] * 100:.2f}%)")

    print("\nPosition Details:")
    for pos in summary['positions']:
        print(f"  {pos['symbol']}: {pos['unrealized_pnl']:,.2f} TRY ({pos['unrealized_pnl_pct']:.2f}%)")

    # Sell some positions
    print("\n--- Executing Sell Orders ---")

    # Take profit on THYAO
    portfolio.sell(
        symbol="THYAO",
        shares=50,
        price=265.80,
        commission=6.65,
        notes="Taking partial profit on THYAO"
    )
    print(f"Sold 50 THYAO @ 265.80 TRY")

    # Cut loss on EREGL
    portfolio.sell(
        symbol="EREGL",
        shares=150,
        price=41.20,
        commission=6.18,
        notes="Cutting loss on EREGL"
    )
    print(f"Sold 150 EREGL @ 41.20 TRY")

    # Get updated summary
    print("\n--- Portfolio After Selling ---")
    summary = portfolio.get_portfolio_summary(current_prices)

    print(f"Total Portfolio Value: {summary['total_value']:,.2f} TRY")
    print(f"Cash: {summary['cash']:,.2f} TRY")
    print(f"Realized P&L: {summary['realized_pnl']:,.2f} TRY")
    print(f"Unrealized P&L: {summary['unrealized_pnl']:,.2f} TRY")
    print(f"Total P&L: {summary['total_pnl']:,.2f} TRY")
    print(f"Total Return: {summary['total_return_pct']:.2f}%")

    print("\nRemaining Positions:")
    for pos in summary['positions']:
        print(f"  {pos['symbol']}: {pos['shares']:.0f} shares, "
              f"P&L: {pos['unrealized_pnl']:,.2f} TRY ({pos['unrealized_pnl_pct']:.2f}%)")

    return portfolio


def example_3_allocation_analysis(portfolio):
    """Example 3: Portfolio allocation analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Portfolio Allocation Analysis")
    print("=" * 80)

    current_prices = {
        "THYAO": 265.80,
        "GARAN": 88.40,
        "EREGL": 41.20
    }

    # Get allocation breakdown
    allocation = portfolio.get_allocation(current_prices)

    print("\n--- Portfolio Allocation ---")
    print(allocation.to_string(index=False))

    # Calculate some additional metrics
    total_value = portfolio.get_portfolio_value(current_prices)
    cash_allocation = (portfolio.cash / total_value * 100) if total_value > 0 else 0

    print(f"\nCash Allocation: {cash_allocation:.2f}%")
    print(f"Equity Allocation: {100 - cash_allocation:.2f}%")

    # Check diversification
    if len(allocation) > 0:
        max_position = allocation['allocation_pct'].max()
        print(f"\nLargest Position: {max_position:.2f}%")
        print(f"Number of Positions: {len(allocation)}")

        if max_position > 30:
            print("⚠️  Warning: Concentrated position (>30%)")
        else:
            print("✓ Portfolio is reasonably diversified")


def example_4_transaction_history(portfolio):
    """Example 4: Analyzing transaction history"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Transaction History Analysis")
    print("=" * 80)

    # Get all transactions
    history = portfolio.get_transaction_history()

    print("\n--- Complete Transaction History ---")
    if not history.empty:
        print(history[['timestamp', 'symbol', 'transaction_type', 'shares', 'price', 'commission']].to_string(index=False))

        # Calculate total volume traded
        buy_volume = history[history['transaction_type'] == 'BUY']['shares'].sum()
        sell_volume = history[history['transaction_type'] == 'SELL']['shares'].sum()

        print(f"\n--- Trading Statistics ---")
        print(f"Total Transactions: {len(history)}")
        print(f"Buy Orders: {len(history[history['transaction_type'] == 'BUY'])}")
        print(f"Sell Orders: {len(history[history['transaction_type'] == 'SELL'])}")
        print(f"Total Buy Volume: {buy_volume:,.0f} shares")
        print(f"Total Sell Volume: {sell_volume:,.0f} shares")

    # Get transactions for specific symbol
    print("\n--- THYAO Transactions ---")
    thyao_history = portfolio.get_transaction_history(symbol="THYAO")
    if not thyao_history.empty:
        print(thyao_history[['timestamp', 'transaction_type', 'shares', 'price']].to_string(index=False))


def example_5_performance_metrics(portfolio):
    """Example 5: Performance metrics and analytics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Performance Metrics and Analytics")
    print("=" * 80)

    current_prices = {
        "THYAO": 265.80,
        "GARAN": 88.40,
        "EREGL": 41.20
    }

    # Calculate performance metrics
    benchmark_return = 5.0  # Assume BIST-100 returned 5%
    metrics = portfolio.calculate_performance_metrics(current_prices, benchmark_return)

    print("\n--- Performance Metrics ---")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profitable Positions: {metrics['profitable_positions']}/{metrics['total_positions']}")
    print(f"Days Invested: {metrics['days_invested']}")
    print(f"Total Commissions: {metrics['total_commissions']:,.2f} TRY")
    print(f"Commission % of Investment: {metrics['commission_pct_of_invested']:.2f}%")

    if 'excess_return' in metrics:
        print(f"\n--- Benchmark Comparison ---")
        print(f"Benchmark Return: {metrics['benchmark_return']:.2f}%")
        print(f"Excess Return (Alpha): {metrics['excess_return']:.2f}%")

        if metrics['excess_return'] > 0:
            print("✓ Outperforming benchmark")
        else:
            print("⚠️  Underperforming benchmark")


def example_6_save_and_load():
    """Example 6: Saving and loading portfolios"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Saving and Loading Portfolios")
    print("=" * 80)

    # Create a new portfolio
    portfolio = create_portfolio(
        name="Saved Portfolio Example",
        initial_cash=50000.0,
        cost_basis_method="AVERAGE"
    )

    # Add some positions
    portfolio.buy(symbol="AKBNK", shares=500, price=45.30)
    portfolio.buy(symbol="ISCTR", shares=300, price=18.75)

    print(f"\n--- Original Portfolio ---")
    print(f"Name: {portfolio.name}")
    print(f"Cash: {portfolio.cash:,.2f} TRY")
    print(f"Positions: {len(portfolio.positions)}")

    # Save to JSON
    json_file = "portfolio_backup.json"
    portfolio.save_to_json(json_file)
    print(f"\n✓ Saved to {json_file}")

    # Load from JSON
    loaded_portfolio = PortfolioManager.load_from_json(json_file)
    print(f"\n--- Loaded Portfolio ---")
    print(f"Name: {loaded_portfolio.name}")
    print(f"Cash: {loaded_portfolio.cash:,.2f} TRY")
    print(f"Positions: {len(loaded_portfolio.positions)}")

    for symbol, position in loaded_portfolio.get_all_positions().items():
        print(f"  {symbol}: {position.shares} shares @ {position.cost_basis:.2f} TRY")

    # Export positions to CSV
    csv_file = "portfolio_positions.csv"
    current_prices = {"AKBNK": 46.50, "ISCTR": 19.20}
    loaded_portfolio.export_to_csv(csv_file, current_prices)
    print(f"\n✓ Positions exported to {csv_file}")

    # Export transactions to CSV
    txn_file = "portfolio_transactions.csv"
    loaded_portfolio.export_transactions_to_csv(txn_file)
    print(f"✓ Transactions exported to {txn_file}")

    # Clean up
    import os
    for f in [json_file, csv_file, txn_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")


def example_7_cost_basis_methods():
    """Example 7: Comparing different cost basis methods"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Cost Basis Methods Comparison")
    print("=" * 80)

    # Create portfolios with different methods
    methods = [
        ("AVERAGE", CostBasisMethod.AVERAGE),
        ("FIFO", CostBasisMethod.FIFO),
        ("LIFO", CostBasisMethod.LIFO)
    ]

    results = {}

    for method_name, method in methods:
        portfolio = PortfolioManager(
            name=f"{method_name} Portfolio",
            initial_cash=50000.0,
            cost_basis_method=method
        )

        # Same transactions for all
        portfolio.buy(symbol="THYAO", shares=100, price=240.0)
        portfolio.buy(symbol="THYAO", shares=100, price=250.0)
        portfolio.buy(symbol="THYAO", shares=100, price=260.0)

        # Sell same amount
        portfolio.sell(symbol="THYAO", shares=150, price=270.0)

        results[method_name] = {
            'realized_pnl': portfolio.realized_pnl,
            'remaining_shares': portfolio.positions.get("THYAO").shares if "THYAO" in portfolio.positions else 0,
            'cost_basis': portfolio.positions.get("THYAO").cost_basis if "THYAO" in portfolio.positions else 0
        }

    print("\n--- Cost Basis Method Comparison ---")
    print("Scenario: Buy 100 @ 240, 100 @ 250, 100 @ 260, then Sell 150 @ 270")
    print()

    for method_name in ["AVERAGE", "FIFO", "LIFO"]:
        r = results[method_name]
        print(f"{method_name}:")
        print(f"  Realized P&L: {r['realized_pnl']:,.2f} TRY")
        print(f"  Remaining Shares: {r['remaining_shares']:.0f}")
        print(f"  Remaining Cost Basis: {r['cost_basis']:,.2f} TRY")
        print()

    print("Note: Different cost basis methods affect realized P&L and tax implications")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("BIST PORTFOLIO MANAGER - COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    # Run examples sequentially
    portfolio = example_1_basic_portfolio()
    portfolio = example_2_selling_and_pnl(portfolio)
    example_3_allocation_analysis(portfolio)
    example_4_transaction_history(portfolio)
    example_5_performance_metrics(portfolio)
    example_6_save_and_load()
    example_7_cost_basis_methods()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("✓ Portfolio creation and initialization")
    print("✓ Buy and sell transactions")
    print("✓ Position tracking and cost basis calculation")
    print("✓ Realized and unrealized P&L")
    print("✓ Portfolio allocation and diversification")
    print("✓ Transaction history analysis")
    print("✓ Performance metrics and benchmarking")
    print("✓ Save/load functionality (JSON, CSV)")
    print("✓ Multiple cost basis methods (FIFO, LIFO, Average)")
    print("\nFor production use, integrate with:")
    print("  - BISTCollector for real-time prices")
    print("  - Database storage for persistence")
    print("  - Risk management systems")
    print("  - Trading signals and strategies")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
