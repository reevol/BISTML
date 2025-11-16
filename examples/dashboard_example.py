"""
Dashboard Integration Example

This example demonstrates how to integrate the Streamlit dashboard with real
trading data, signals, and portfolio information from the BIST AI Trading System.

Usage:
    python examples/dashboard_example.py

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals.generator import SignalGenerator, create_model_output, SignalType
from src.portfolio.manager import PortfolioManager, create_portfolio
from src.backtesting.engine import BacktestEngine, BacktestConfig, PositionSizing
from src.backtesting.metrics import calculate_all_metrics


def generate_sample_signals():
    """
    Generate sample trading signals for dashboard demonstration.

    In production, these would come from:
    - Signal generator running on real-time data
    - Database queries for historical signals
    - Model predictions on current market conditions
    """
    print("Generating sample trading signals...")

    # Initialize signal generator
    generator = SignalGenerator(
        enable_dynamic_thresholds=True,
        risk_adjustment=True,
        min_confidence=0.5
    )

    # Sample stocks
    stocks = ['THYAO', 'GARAN', 'EREGL', 'KCHOL', 'AKBNK']
    signals_data = []

    for stock in stocks:
        # Create sample model outputs
        model_outputs = [
            create_model_output(
                model_name='lstm_forecaster',
                model_type='regression',
                prediction=np.random.uniform(90, 110),
                confidence=np.random.uniform(0.6, 0.9)
            ),
            create_model_output(
                model_name='random_forest',
                model_type='classification',
                prediction=np.random.randint(0, 5),
                confidence=np.random.uniform(0.5, 0.8),
                probabilities=np.random.dirichlet(np.ones(5))
            ),
            create_model_output(
                model_name='sentiment_analyzer',
                model_type='nlp',
                prediction=np.random.uniform(-0.5, 0.5),
                confidence=np.random.uniform(0.6, 0.85)
            )
        ]

        # Generate signal
        current_price = np.random.uniform(50, 200)
        historical_prices = pd.Series(
            current_price * (1 + np.random.randn(50) * 0.02)
        )

        signal = generator.generate_signal(
            stock_code=stock,
            model_outputs=model_outputs,
            current_price=current_price,
            historical_prices=historical_prices
        )

        # Store signal data
        signals_data.append({
            'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 120)),
            'stock_code': stock,
            'signal': signal.signal.name,
            'confidence_score': signal.confidence_score,
            'current_price': signal.current_price,
            'target_price': signal.target_price,
            'expected_return': signal.expected_return * 100 if signal.expected_return else 0,
            'position_size': signal.position_size,
            'risk_score': signal.risk_score,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        })

    signals_df = pd.DataFrame(signals_data)
    print(f"Generated {len(signals_df)} signals")

    return signals_df


def create_sample_portfolio():
    """
    Create a sample portfolio with positions and transactions.

    In production, this would load from:
    - Portfolio manager state file
    - Database with transaction history
    - Real-time position tracking system
    """
    print("\nCreating sample portfolio...")

    # Initialize portfolio
    portfolio = create_portfolio(
        name="BIST AI Portfolio",
        initial_cash=100000.0,
        cost_basis_method="AVERAGE",
        currency="TRY"
    )

    # Execute sample trades
    trades = [
        ('THYAO', 100, 250.00, 10.0),
        ('GARAN', 200, 85.00, 15.0),
        ('EREGL', 500, 42.00, 20.0),
        ('KCHOL', 150, 120.00, 12.0)
    ]

    for symbol, shares, price, commission in trades:
        portfolio.buy(
            symbol=symbol,
            shares=shares,
            price=price,
            timestamp=datetime.now() - timedelta(days=np.random.randint(10, 90)),
            commission=commission,
            notes=f"Sample trade for {symbol}"
        )

    # Get portfolio summary
    current_prices = {
        'THYAO': 265.50,
        'GARAN': 92.30,
        'EREGL': 45.80,
        'KCHOL': 125.40
    }

    summary = portfolio.get_portfolio_summary(current_prices)

    print(f"Portfolio created with {len(portfolio.positions)} positions")
    print(f"Total value: {summary['total_value']:,.2f} TRY")
    print(f"Total return: {summary['total_return_pct']:.2f}%")

    return portfolio, summary


def run_sample_backtest():
    """
    Run a sample backtest for dashboard demonstration.

    In production, backtests would:
    - Use historical BIST data from database
    - Apply real trading signals from models
    - Include realistic transaction costs
    - Run over various time periods
    """
    print("\nRunning sample backtest...")

    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)

    base_price = 100.0
    returns = np.random.randn(n_days) * 0.02 + 0.0005
    prices = base_price * (1 + returns).cumprod()

    price_data = pd.DataFrame({
        'symbol': 'THYAO',
        'date': dates,
        'open': prices * (1 + np.random.randn(n_days) * 0.005),
        'high': prices * (1 + abs(np.random.randn(n_days)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(n_days)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    price_data.set_index('date', inplace=True)

    # Generate sample signals (simple moving average crossover)
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

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        position_sizing=PositionSizing.PERCENT_EQUITY,
        position_size_value=0.20,
        use_stop_loss=True,
        use_take_profit=True,
        min_confidence=0.6
    )

    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run(price_data, signals)

    print(f"Backtest completed:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Win rate: {results.win_rate:.2f}%")
    print(f"  Total return: {results.total_return_pct:.2f}%")
    print(f"  Sharpe ratio: {results.sharpe_ratio:.3f}")
    print(f"  Max drawdown: {results.max_drawdown:.2f}%")

    return results


def save_dashboard_data(signals_df, portfolio_summary, backtest_results):
    """
    Save data in formats that the dashboard can load.

    In production, this would:
    - Save to database
    - Update cache
    - Trigger real-time updates
    """
    print("\nSaving data for dashboard...")

    output_dir = project_root / "data" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save signals
    signals_path = output_dir / "signals.csv"
    signals_df.to_csv(signals_path, index=False)
    print(f"Signals saved to: {signals_path}")

    # Save portfolio
    portfolio_path = output_dir / "portfolio.json"
    with open(portfolio_path, 'w') as f:
        import json
        json.dump(portfolio_summary, f, indent=2, default=str)
    print(f"Portfolio saved to: {portfolio_path}")

    # Save backtest results
    backtest_path = output_dir / "backtest.json"
    with open(backtest_path, 'w') as f:
        json.dump(backtest_results.to_dict(), f, indent=2, default=str)
    print(f"Backtest results saved to: {backtest_path}")

    # Save equity curve
    equity_path = output_dir / "equity_curve.csv"
    backtest_results.equity_curve.to_csv(equity_path)
    print(f"Equity curve saved to: {equity_path}")

    print("\nData saved successfully!")
    print(f"You can now run the dashboard with: python run_dashboard.py")


def main():
    """Main execution function."""
    print("=" * 80)
    print("BIST AI Trading System - Dashboard Data Generator")
    print("=" * 80)

    # Generate signals
    signals_df = generate_sample_signals()

    # Create portfolio
    portfolio, portfolio_summary = create_sample_portfolio()

    # Run backtest
    backtest_results = run_sample_backtest()

    # Save data
    save_dashboard_data(signals_df, portfolio_summary, backtest_results)

    print("\n" + "=" * 80)
    print("Dashboard data generation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run the dashboard: python run_dashboard.py")
    print("2. Open your browser to: http://localhost:8501")
    print("3. Explore the four main tabs:")
    print("   - Live Signals: View current trading signals")
    print("   - Portfolio: Monitor positions and P&L")
    print("   - Backtesting: Analyze historical performance")
    print("   - Performance: Review comprehensive metrics")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
