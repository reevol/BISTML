"""
Example: Portfolio Alert System Usage

This example demonstrates how to use the Portfolio Alert System to:
1. Create an alert manager with a portfolio
2. Process new trading signals
3. Generate alerts for holdings
4. Send notifications via email/Telegram
5. Track alert history

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.portfolio.manager import PortfolioManager, create_portfolio
from src.portfolio.alerts import (
    PortfolioAlertManager,
    create_alert_manager,
    AlertType,
    AlertPriority,
    NotificationChannel
)
from src.signals.generator import (
    SignalGenerator,
    create_signal_generator,
    create_model_output,
    SignalType as GenSignalType
)
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("Portfolio Alert System - Complete Example")
    print("=" * 80)

    # ========================================================================
    # Step 1: Create a portfolio with some holdings
    # ========================================================================
    print("\n1. Creating Portfolio with Holdings")
    print("-" * 80)

    portfolio = create_portfolio(
        name="Example Trading Portfolio",
        initial_cash=100000.0,
        cost_basis_method="AVERAGE",
        currency="TRY"
    )

    # Add some holdings
    portfolio.buy(symbol="THYAO", shares=100, price=250.0, commission=10.0)
    portfolio.buy(symbol="GARAN", shares=200, price=85.0, commission=15.0)
    portfolio.buy(symbol="AKBNK", shares=150, price=48.0, commission=12.0)

    print(f"Portfolio created with {len(portfolio.positions)} positions:")
    for symbol, position in portfolio.get_all_positions().items():
        print(f"  {symbol}: {position.shares} shares @ {position.cost_basis:.2f} TRY")

    # ========================================================================
    # Step 2: Create Alert Manager
    # ========================================================================
    print("\n2. Creating Alert Manager")
    print("-" * 80)

    # Watchlist of stocks we're interested in (but don't own yet)
    watchlist = ['EREGL', 'SAHOL', 'SISE']

    alert_manager = create_alert_manager(
        portfolio_manager=portfolio,
        watchlist=watchlist
    )

    print(f"Alert manager created")
    print(f"  Portfolio positions being monitored: {len(portfolio.positions)}")
    print(f"  Watchlist symbols: {len(alert_manager.watchlist)}")
    print(f"  Active alert rules: {len(alert_manager.alert_rules)}")

    for rule in alert_manager.alert_rules:
        print(f"    - {rule.name} ({rule.priority.value})")

    # ========================================================================
    # Step 3: Generate Trading Signals
    # ========================================================================
    print("\n3. Generating Trading Signals")
    print("-" * 80)

    # Create signal generator
    signal_generator = create_signal_generator(
        enable_dynamic_thresholds=True,
        risk_adjustment=True
    )

    # Current prices
    current_prices = {
        'THYAO': 265.0,   # Up from our purchase
        'GARAN': 82.0,    # Down from our purchase
        'AKBNK': 52.0,    # Up from our purchase
        'EREGL': 45.0,    # On watchlist
        'SAHOL': 38.0,    # On watchlist
    }

    # Simulate signals for our holdings and watchlist
    signals = []

    # Signal 1: STRONG SELL for GARAN (we hold this - should trigger CRITICAL alert)
    model_outputs_garan = [
        create_model_output('lstm', 'regression', prediction=78.0, confidence=0.75),
        create_model_output('random_forest', 'classification', prediction=1, confidence=0.80),  # SELL
        create_model_output('sentiment', 'nlp', prediction=-0.6, confidence=0.70)
    ]

    signal_garan = signal_generator.generate_signal(
        stock_code='GARAN',
        model_outputs=model_outputs_garan,
        current_price=current_prices['GARAN'],
        historical_prices=pd.Series([85, 84, 83, 82, 81, 82] * 5)
    )
    signals.append(signal_garan)

    print(f"\nSignal generated for GARAN (we own this):")
    print(f"  Signal: {signal_garan.signal.name}")
    print(f"  Confidence: {signal_garan.confidence.name} ({signal_garan.confidence_score:.2%})")
    print(f"  Target Price: {signal_garan.target_price:.2f} TRY")

    # Signal 2: STRONG BUY for THYAO (we hold this - adding opportunity)
    model_outputs_thyao = [
        create_model_output('lstm', 'regression', prediction=280.0, confidence=0.82),
        create_model_output('random_forest', 'classification', prediction=4, confidence=0.78),  # STRONG_BUY
        create_model_output('sentiment', 'nlp', prediction=0.7, confidence=0.75)
    ]

    signal_thyao = signal_generator.generate_signal(
        stock_code='THYAO',
        model_outputs=model_outputs_thyao,
        current_price=current_prices['THYAO'],
        historical_prices=pd.Series([250, 252, 255, 258, 262, 265] * 5)
    )
    signals.append(signal_thyao)

    print(f"\nSignal generated for THYAO (we own this):")
    print(f"  Signal: {signal_thyao.signal.name}")
    print(f"  Confidence: {signal_thyao.confidence.name} ({signal_thyao.confidence_score:.2%})")
    print(f"  Target Price: {signal_thyao.target_price:.2f} TRY")

    # Signal 3: STRONG BUY for EREGL (watchlist - new opportunity)
    model_outputs_eregl = [
        create_model_output('lstm', 'regression', prediction=50.0, confidence=0.85),
        create_model_output('random_forest', 'classification', prediction=4, confidence=0.83),
        create_model_output('sentiment', 'nlp', prediction=0.65, confidence=0.78)
    ]

    signal_eregl = signal_generator.generate_signal(
        stock_code='EREGL',
        model_outputs=model_outputs_eregl,
        current_price=current_prices['EREGL'],
        historical_prices=pd.Series([42, 43, 44, 44.5, 45, 45] * 5)
    )
    signals.append(signal_eregl)

    print(f"\nSignal generated for EREGL (on watchlist):")
    print(f"  Signal: {signal_eregl.signal.name}")
    print(f"  Confidence: {signal_eregl.confidence.name} ({signal_eregl.confidence_score:.2%})")
    print(f"  Target Price: {signal_eregl.target_price:.2f} TRY")

    # ========================================================================
    # Step 4: Process Signals and Generate Alerts
    # ========================================================================
    print("\n4. Processing Signals and Generating Alerts")
    print("-" * 80)

    alerts = alert_manager.process_signals(signals, current_prices)

    print(f"\nGenerated {len(alerts)} alerts:")

    for i, alert in enumerate(alerts, 1):
        print(f"\n  Alert {i}:")
        print(f"    Type: {alert.alert_type.value}")
        print(f"    Priority: {alert.priority.value}")
        print(f"    Symbol: {alert.symbol}")
        print(f"    Signal: {alert.signal_direction.value}")
        print(f"    Confidence: {alert.confidence_score:.1f}%")
        print(f"    Title: {alert.title}")

        if alert.position_size:
            print(f"    Your Position: {alert.position_size:.0f} shares")
            if alert.unrealized_pnl is not None:
                pnl_symbol = "+" if alert.unrealized_pnl > 0 else ""
                print(f"    Current P&L: {pnl_symbol}{alert.unrealized_pnl:,.2f} TRY ({alert.unrealized_pnl_pct:+.2f}%)")

        if alert.expected_return is not None:
            print(f"    Expected Return: {alert.expected_return*100:+.2f}%")

    # ========================================================================
    # Step 5: Display Alert Messages
    # ========================================================================
    print("\n5. Alert Message Examples")
    print("-" * 80)

    for alert in alerts[:2]:  # Show first 2 alerts
        print(f"\n{'-' * 80}")
        print(f"ALERT: {alert.title}")
        print('-' * 80)
        print(alert.message)

    # ========================================================================
    # Step 6: Send Notifications (Demo Mode - Won't Actually Send)
    # ========================================================================
    print("\n6. Sending Notifications (Demo Mode)")
    print("-" * 80)

    # Note: This won't actually send emails unless SMTP is configured
    results = alert_manager.send_alerts(alerts)

    print(f"\nNotification Results:")
    print(f"  Total alerts: {results['total']}")
    print(f"  Email attempts: {results['email']}")
    print(f"  Telegram attempts: {results['telegram']}")
    print(f"  SMS attempts: {results['sms']}")

    # ========================================================================
    # Step 7: Alert History and Summary
    # ========================================================================
    print("\n7. Alert History and Summary")
    print("-" * 80)

    history = alert_manager.get_alert_history(limit=10)
    print(f"\nAlert History: {len(history)} alerts")

    summary = alert_manager.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Total Alerts: {summary['total_alerts']}")
    print(f"  By Priority: {summary['alerts_by_priority']}")
    print(f"  By Type: {summary['alerts_by_type']}")

    if summary['most_alerted_symbols']:
        print(f"\n  Most Alerted Symbols:")
        for symbol, count in summary['most_alerted_symbols'][:5]:
            print(f"    {symbol}: {count} alerts")

    # ========================================================================
    # Step 8: Portfolio Summary with Alerts
    # ========================================================================
    print("\n8. Portfolio Summary with Current Alerts")
    print("-" * 80)

    portfolio_summary = portfolio.get_portfolio_summary(current_prices)

    print(f"\nPortfolio: {portfolio_summary['portfolio_name']}")
    print(f"Total Value: {portfolio_summary['total_value']:,.2f} TRY")
    print(f"Cash: {portfolio_summary['cash']:,.2f} TRY")
    print(f"Positions Value: {portfolio_summary['positions_value']:,.2f} TRY")
    print(f"Unrealized P&L: {portfolio_summary['unrealized_pnl']:+,.2f} TRY")
    print(f"Total Return: {portfolio_summary['total_return_pct']:+.2f}%")

    print(f"\nPositions with Active Alerts:")
    for position in portfolio_summary['positions']:
        # Check if this position has alerts
        position_alerts = [a for a in alerts if a.symbol == position['symbol']]
        if position_alerts:
            print(f"\n  {position['symbol']}:")
            print(f"    Shares: {position['shares']:.0f}")
            print(f"    Current Price: {position['current_price']:.2f} TRY")
            print(f"    P&L: {position['unrealized_pnl']:+,.2f} TRY ({position['unrealized_pnl_pct']:+.2f}%)")
            print(f"    Active Alerts: {len(position_alerts)}")
            for alert in position_alerts:
                print(f"      - {alert.priority.value}: {alert.signal_direction.value} ({alert.confidence_score:.0f}%)")

    # ========================================================================
    # Finish
    # ========================================================================
    print("\n" + "=" * 80)
    print("Portfolio Alert System Example Completed Successfully!")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("  1. Alerts are automatically generated when signals match your holdings")
    print("  2. Different alert priorities help you focus on critical signals")
    print("  3. Alert messages include position details and P&L information")
    print("  4. Notifications can be sent via Email, Telegram, and SMS")
    print("  5. Alert throttling prevents notification spam")
    print("  6. Alert history helps track signal effectiveness")

    print("\nNext Steps:")
    print("  - Configure SMTP settings in .env for email notifications")
    print("  - Set up Telegram bot for instant mobile alerts")
    print("  - Customize alert rules for your trading strategy")
    print("  - Integrate with signal scheduler for automated monitoring")


if __name__ == "__main__":
    main()
