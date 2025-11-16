"""
Example Usage of Backtesting Simulator

This script demonstrates how to use the comprehensive backtesting simulator
for BIST AI Trading System, including:
- Historical backtesting
- Walk-forward analysis
- Monte Carlo simulation

Author: BIST AI Trading System
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from simulator import (
    BacktestSimulator,
    BacktestConfig,
    WalkForwardConfig,
    MonteCarloConfig,
    BacktestMode,
    ExecutionModel,
    run_quick_backtest,
    compare_strategies
)


# ============================================================================
# Example 1: Basic Historical Backtest
# ============================================================================

def example_historical_backtest():
    """Run a basic historical backtest"""
    print("=" * 80)
    print("Example 1: Historical Backtest")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')

    # Simulate price data for multiple stocks
    symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'SAHOL']
    data_frames = []

    for symbol in symbols:
        # Random walk price simulation
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        })
        data_frames.append(df)

    price_data = pd.concat(data_frames, ignore_index=True)
    price_data.set_index('timestamp', inplace=True)

    # Simple signal generator (example)
    def simple_signal_generator(timestamp, market_data):
        """Generate simple momentum-based signals"""
        signals = []

        for _, row in market_data.iterrows():
            # Simple example: buy if price is above 100, sell if below 95
            if row['close'] > 105:
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY',
                    'confidence_score': 0.7,
                    'target_price': row['close'] * 1.05
                })
            elif row['close'] < 95:
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'SELL',
                    'confidence_score': 0.6
                })

        return signals

    # Configure backtest
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        commission_rate=0.001,  # 0.1% commission
        slippage_bps=5.0,  # 5 basis points slippage
        position_size_pct=0.2,  # 20% per position
        max_positions=5,
        execution_model=ExecutionModel.REALISTIC,
        timeframe='1h'
    )

    # Create simulator
    simulator = BacktestSimulator(
        config=config,
        signal_generator=simple_signal_generator,
        price_data=price_data
    )

    # Run backtest
    result = simulator.run_historical_backtest()

    # Display results
    print("\nBacktest Results:")
    print(result.summary())

    # Plot equity curve
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)

    simulator.plot_equity_curve(
        result,
        save_path=str(output_dir / 'equity_curve.png'),
        show_drawdown=True
    )

    # Export results
    simulator.export_results(
        result,
        output_dir=output_dir,
        formats=['csv', 'json']
    )

    print(f"\nResults exported to {output_dir}/")

    return result


# ============================================================================
# Example 2: Walk-Forward Analysis
# ============================================================================

def example_walk_forward_analysis():
    """Run walk-forward analysis"""
    print("\n" + "=" * 80)
    print("Example 2: Walk-Forward Analysis")
    print("=" * 80)

    # Create sample price data (longer period for WF)
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='1D')

    symbols = ['THYAO', 'GARAN', 'AKBNK']
    data_frames = []

    for symbol in symbols:
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'close': prices,
            'volume': np.random.randint(500000, 2000000, len(dates))
        })
        data_frames.append(df)

    price_data = pd.concat(data_frames, ignore_index=True)
    price_data.set_index('timestamp', inplace=True)

    # Signal generator
    def momentum_signal_generator(timestamp, market_data):
        """Momentum-based signal generator"""
        signals = []

        for _, row in market_data.iterrows():
            # Calculate simple momentum (this is just an example)
            if np.random.random() > 0.7:  # Random signal for demo
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'confidence_score': np.random.uniform(0.5, 0.9)
                })

        return signals

    # Configure backtest
    config = BacktestConfig(
        start_date='2021-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        commission_rate=0.001,
        timeframe='1d'
    )

    # Configure walk-forward
    wf_config = WalkForwardConfig(
        train_period_days=365,  # 1 year training
        test_period_days=90,    # 3 months testing
        step_size_days=90,      # Move forward 3 months
        min_train_samples=100,
        reoptimize=True,
        anchored=False          # Rolling window
    )

    # Create simulator
    simulator = BacktestSimulator(
        config=config,
        signal_generator=momentum_signal_generator,
        price_data=price_data
    )

    # Strategy optimizer (example)
    def optimize_strategy(train_data):
        """Optimize strategy parameters on training data"""
        # In real implementation, this would optimize parameters
        # For now, return dummy parameters
        return {
            'lookback_period': np.random.randint(10, 50),
            'threshold': np.random.uniform(0.5, 0.9)
        }

    # Run walk-forward analysis
    wf_result = simulator.run_walk_forward_analysis(
        wf_config=wf_config,
        strategy_optimizer=optimize_strategy,
        progress_callback=lambda p: print(f"Progress: {p:.1f}%") if p % 20 == 0 else None
    )

    # Display results
    print(f"\nWalk-Forward Analysis Complete")
    print(f"Number of periods: {wf_result['combined_metrics']['n_periods']}")
    print(f"Average Sharpe Ratio: {wf_result['combined_metrics']['avg_sharpe_ratio']:.3f}")
    print(f"Average Return: {wf_result['combined_metrics']['avg_return']:.2%}")
    print(f"Consistency Score: {wf_result['combined_metrics']['consistency_score']:.2%}")

    # Create summary DataFrame
    periods_summary = []
    for period in wf_result['periods']:
        periods_summary.append({
            'Period': period['period'],
            'Test Start': period['test_start'].strftime('%Y-%m-%d'),
            'Test End': period['test_end'].strftime('%Y-%m-%d'),
            'N Trades': period['n_trades'],
            'Return': f"{period['total_return']:.2%}",
            'Sharpe': f"{period['sharpe_ratio']:.3f}",
            'Max DD': f"{period['max_drawdown']:.2%}"
        })

    summary_df = pd.DataFrame(periods_summary)
    print("\nPeriod-by-Period Results:")
    print(summary_df.to_string(index=False))

    return wf_result


# ============================================================================
# Example 3: Monte Carlo Simulation
# ============================================================================

def example_monte_carlo_simulation():
    """Run Monte Carlo simulation"""
    print("\n" + "=" * 80)
    print("Example 3: Monte Carlo Simulation")
    print("=" * 80)

    # First, run a base backtest
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')

    symbols = ['THYAO', 'GARAN']
    data_frames = []

    for symbol in symbols:
        returns = np.random.normal(0.0002, 0.018, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices * (1 + np.random.uniform(-0.008, 0.008, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.015, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.015, len(dates))),
            'close': prices,
            'volume': np.random.randint(200000, 1500000, len(dates))
        })
        data_frames.append(df)

    price_data = pd.concat(data_frames, ignore_index=True)
    price_data.set_index('timestamp', inplace=True)

    def example_signal_generator(timestamp, market_data):
        """Example signal generator"""
        signals = []

        for _, row in market_data.iterrows():
            if np.random.random() > 0.85:
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if np.random.random() > 0.4 else 'SELL',
                    'confidence_score': np.random.uniform(0.6, 0.95)
                })

        return signals

    # Configure backtest
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        timeframe='1h'
    )

    # Create simulator
    simulator = BacktestSimulator(
        config=config,
        signal_generator=example_signal_generator,
        price_data=price_data
    )

    # Run base backtest
    print("\nRunning base backtest...")
    base_result = simulator.run_historical_backtest()

    print(f"Base backtest complete: {len(base_result.trades)} trades")
    print(f"Base return: {base_result.metrics['total_return']:.2%}")
    print(f"Base Sharpe: {base_result.metrics['sharpe_ratio']:.3f}")

    # Configure Monte Carlo
    mc_config = MonteCarloConfig(
        n_simulations=1000,
        randomization_method='shuffle',  # or 'bootstrap', 'block_bootstrap'
        resample_trades=True,
        shuffle_returns=True,
        block_length=20,
        confidence_levels=[0.05, 0.25, 0.50, 0.75, 0.95],
        random_seed=42
    )

    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    mc_result = simulator.run_monte_carlo_simulation(
        mc_config=mc_config,
        base_result=base_result,
        progress_callback=lambda p: print(f"MC Progress: {p:.0f}%") if p % 10 == 0 else None
    )

    # Display results
    stats = mc_result['statistics']
    ci = stats['confidence_intervals']

    print("\nMonte Carlo Simulation Results:")
    print(f"Number of simulations: {stats['n_simulations']}")
    print(f"\nReturn Statistics:")
    print(f"  Mean: {stats['mean_return']:.2%}")
    print(f"  Median: {stats['median_return']:.2%}")
    print(f"  Std Dev: {stats['std_return']:.2%}")
    print(f"  Best: {stats['best_return']:.2%}")
    print(f"  Worst: {stats['worst_return']:.2%}")
    print(f"\nProbability of Profit: {stats['prob_profit']:.2%}")

    print("\nConfidence Intervals (Return):")
    for level in [5, 25, 50, 75, 95]:
        key = f'return_{level}pct'
        if key in ci:
            print(f"  {level}th percentile: {ci[key]:.2%}")

    print("\nSharpe Ratio Statistics:")
    print(f"  Mean: {stats['mean_sharpe']:.3f}")
    print(f"  Median: {stats['median_sharpe']:.3f}")
    print(f"  Best: {stats['best_sharpe']:.3f}")
    print(f"  Worst: {stats['worst_sharpe']:.3f}")

    # Plot distribution
    output_dir = Path('backtest_results')
    simulator.plot_monte_carlo_distribution(
        mc_result,
        metric='total_return',
        save_path=str(output_dir / 'mc_distribution.png')
    )

    print(f"\nMonte Carlo distribution plot saved to {output_dir}/mc_distribution.png")

    return mc_result


# ============================================================================
# Example 4: Strategy Comparison
# ============================================================================

def example_strategy_comparison():
    """Compare multiple trading strategies"""
    print("\n" + "=" * 80)
    print("Example 4: Strategy Comparison")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')

    symbols = ['THYAO', 'GARAN', 'AKBNK']
    data_frames = []

    for symbol in symbols:
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'close': prices,
            'volume': np.random.randint(300000, 1500000, len(dates))
        })
        data_frames.append(df)

    price_data = pd.concat(data_frames, ignore_index=True)
    price_data.set_index('timestamp', inplace=True)

    # Define multiple strategies
    def conservative_strategy(timestamp, market_data):
        """Conservative strategy with high confidence threshold"""
        signals = []
        for _, row in market_data.iterrows():
            if np.random.random() > 0.9:  # Less frequent signals
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if row['close'] > 102 else 'SELL',
                    'confidence_score': 0.8
                })
        return signals

    def aggressive_strategy(timestamp, market_data):
        """Aggressive strategy with more frequent trades"""
        signals = []
        for _, row in market_data.iterrows():
            if np.random.random() > 0.7:  # More frequent signals
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if row['close'] > 100 else 'SELL',
                    'confidence_score': 0.6
                })
        return signals

    def balanced_strategy(timestamp, market_data):
        """Balanced strategy"""
        signals = []
        for _, row in market_data.iterrows():
            if np.random.random() > 0.8:
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if row['close'] > 101 else 'SELL',
                    'confidence_score': 0.7
                })
        return signals

    # Compare strategies
    strategies = {
        'Conservative': conservative_strategy,
        'Aggressive': aggressive_strategy,
        'Balanced': balanced_strategy
    }

    comparison_df = compare_strategies(
        strategies=strategies,
        price_data=price_data,
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        commission_rate=0.001,
        timeframe='1h'
    )

    print("\nStrategy Comparison Results:")
    print(comparison_df.to_string())

    # Highlight key metrics
    print("\nKey Metrics Comparison:")
    key_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate', 'n_trades']
    if all(m in comparison_df.columns for m in key_metrics):
        print(comparison_df[key_metrics].to_string())

    return comparison_df


# ============================================================================
# Example 5: Using Quick Backtest Function
# ============================================================================

def example_quick_backtest():
    """Demonstrate quick backtest convenience function"""
    print("\n" + "=" * 80)
    print("Example 5: Quick Backtest")
    print("=" * 80)

    # Create minimal price data
    dates = pd.date_range(start='2023-06-01', end='2023-12-31', freq='1D')

    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))

    price_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'THYAO',
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': 1000000
    })
    price_data.set_index('timestamp', inplace=True)

    # Simple signal generator
    def simple_signals(timestamp, market_data):
        signals = []
        for _, row in market_data.iterrows():
            if np.random.random() > 0.85:
                signals.append({
                    'stock_code': row['symbol'],
                    'signal': 'BUY' if np.random.random() > 0.5 else 'SELL'
                })
        return signals

    # Run quick backtest
    result = run_quick_backtest(
        start_date='2023-06-01',
        end_date='2023-12-31',
        signal_generator=simple_signals,
        price_data=price_data,
        initial_capital=50000.0,
        commission_rate=0.0015
    )

    print("\nQuick Backtest Results:")
    print(f"Total Return: {result.metrics['total_return_pct']:.2f}%")
    print(f"Number of Trades: {result.metrics['n_trades']}")
    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown_pct']:.2f}%")

    return result


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("BIST AI Trading System - Backtesting Simulator Examples")
    print("=" * 80)
    print()

    # Run examples
    try:
        # Example 1: Historical Backtest
        hist_result = example_historical_backtest()

        # Example 2: Walk-Forward Analysis
        # wf_result = example_walk_forward_analysis()

        # Example 3: Monte Carlo Simulation
        # mc_result = example_monte_carlo_simulation()

        # Example 4: Strategy Comparison
        # comparison = example_strategy_comparison()

        # Example 5: Quick Backtest
        # quick_result = example_quick_backtest()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
