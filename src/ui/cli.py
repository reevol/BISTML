"""
Command-Line Interface for BIST AI Trading System

This module provides a comprehensive CLI for interacting with the AI trading system.
Supports commands for signal generation, backtesting, model training, data collection,
and portfolio management.

Commands:
- run-signals: Generate trading signals for BIST stocks
- backtest: Run backtesting on historical data
- train-models: Train or retrain machine learning models
- collect-data: Collect market data from various sources
- show-portfolio: Display portfolio status and positions

Author: BIST AI Trading System
Date: 2025-11-16
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Command: run-signals
# ============================================================================

def cmd_run_signals(args):
    """
    Run signal generation for BIST stocks

    This command generates trading signals by:
    1. Loading trained models
    2. Fetching current market data
    3. Running predictions
    4. Generating aggregated signals
    5. Optionally saving results to file
    """
    from src.signals.generator import SignalGenerator, create_signal_generator, create_model_output
    from src.models.trainer import ModelTrainer
    import pandas as pd
    import numpy as np

    logger.info("=" * 80)
    logger.info("SIGNAL GENERATION")
    logger.info("=" * 80)

    # Parse symbols
    symbols = args.symbols.split(',') if args.symbols else ['THYAO', 'GARAN', 'AKBNK']
    logger.info(f"Generating signals for: {', '.join(symbols)}")

    # Initialize signal generator
    generator = create_signal_generator(
        enable_dynamic_thresholds=not args.no_dynamic_thresholds,
        risk_adjustment=not args.no_risk_adjustment,
        min_confidence=args.min_confidence
    )

    # Load models (in production, load actual trained models)
    logger.info("Loading models...")
    # This is a simplified version - in production, load actual trained models

    results = {}

    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")

        # In production, you would:
        # 1. Fetch current price and historical data
        # 2. Load trained models
        # 3. Generate predictions from each model
        # For demo purposes, we'll create dummy model outputs

        current_price = 100.0  # Replace with actual price fetch
        historical_prices = pd.Series([95, 96, 97, 98, 99, 100] * 5)  # Replace with actual data

        # Create dummy model outputs (replace with actual model predictions)
        model_outputs = [
            create_model_output(
                model_name='lstm_forecaster',
                model_type='regression',
                prediction=current_price * 1.05,  # 5% predicted increase
                confidence=0.75
            ),
            create_model_output(
                model_name='random_forest',
                model_type='classification',
                prediction=3,  # BUY
                confidence=0.70,
                probabilities=np.array([0.05, 0.10, 0.15, 0.70, 0.00])
            ),
            create_model_output(
                model_name='sentiment_analyzer',
                model_type='nlp',
                prediction=0.3,  # Positive sentiment
                confidence=0.65
            )
        ]

        # Generate signal
        signal = generator.generate_signal(
            stock_code=symbol,
            model_outputs=model_outputs,
            current_price=current_price,
            historical_prices=historical_prices
        )

        results[symbol] = signal

        # Display signal
        logger.info(f"Signal: {signal.signal.name}")
        logger.info(f"Confidence: {signal.confidence.name} ({signal.confidence_score:.2%})")
        logger.info(f"Expected Return: {signal.expected_return:.2%}" if signal.expected_return else "Expected Return: N/A")
        logger.info(f"Position Size: {signal.position_size:.2%}" if signal.position_size else "Position Size: N/A")

    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert signals to serializable format
        signals_data = {
            symbol: signal.to_dict()
            for symbol, signal in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(signals_data, f, indent=2)

        logger.info(f"\nSignals saved to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Signal generation completed successfully!")
    logger.info("=" * 80)


# ============================================================================
# Command: backtest
# ============================================================================

def cmd_backtest(args):
    """
    Run backtesting on historical data

    This command:
    1. Loads historical price data
    2. Loads or generates trading signals
    3. Simulates trading based on signals
    4. Calculates performance metrics
    5. Generates backtest report
    """
    from src.backtesting.engine import BacktestEngine, BacktestConfig, PositionSizing, export_results
    import pandas as pd
    import numpy as np

    logger.info("=" * 80)
    logger.info("BACKTESTING")
    logger.info("=" * 80)

    # Parse date range
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    else:
        start_date = pd.to_datetime('2023-01-01')

    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    else:
        end_date = pd.to_datetime('2023-12-31')

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")

    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        position_sizing=PositionSizing.PERCENT_EQUITY,
        position_size_value=args.position_size,
        use_stop_loss=not args.no_stop_loss,
        use_take_profit=not args.no_take_profit,
        min_confidence=args.min_confidence
    )

    logger.info(f"Initial capital: {config.initial_capital:,.2f} TRY")
    logger.info(f"Commission rate: {config.commission_rate*100:.3f}%")
    logger.info(f"Position sizing: {args.position_size*100:.1f}% per trade")

    # Load or generate data
    logger.info("\nLoading data...")

    # In production, load actual historical data
    # For demo, generate sample data
    dates = pd.date_range(start_date, end_date, freq='D')
    np.random.seed(42)

    symbol = args.symbol or 'THYAO'
    base_price = 100.0
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * (1 + returns).cumprod()

    price_data = pd.DataFrame({
        'symbol': symbol,
        'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': prices * (1 + abs(np.random.randn(len(dates)) * 0.01)),
        'low': prices * (1 - abs(np.random.randn(len(dates)) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    logger.info(f"Loaded {len(price_data)} days of price data for {symbol}")

    # Generate or load signals
    if args.signals_file:
        logger.info(f"Loading signals from {args.signals_file}")
        signals = pd.read_csv(args.signals_file)
        signals['date'] = pd.to_datetime(signals['date'])
        signals = signals.set_index('date')
    else:
        logger.info("Generating sample signals (SMA crossover strategy)")
        # Generate simple signals based on moving average crossover
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
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence
            })

        signals = pd.DataFrame(signals_list, index=price_data.index[50:])

    logger.info(f"Loaded {len(signals)} trading signals")

    # Run backtest
    logger.info("\nRunning backtest...")
    engine = BacktestEngine(config)
    results = engine.run(price_data, signals)

    # Display results
    print("\n" + str(results))

    # Save results if output directory specified
    if args.output:
        output_dir = Path(args.output)
        export_results(results, str(output_dir))
        logger.info(f"\nBacktest results saved to {output_dir}")

    logger.info("\n" + "=" * 80)
    logger.info("Backtesting completed successfully!")
    logger.info("=" * 80)


# ============================================================================
# Command: train-models
# ============================================================================

def cmd_train_models(args):
    """
    Train or retrain machine learning models

    This command:
    1. Loads training data
    2. Prepares features
    3. Trains specified models
    4. Evaluates performance
    5. Saves trained models
    """
    from src.models.trainer import ModelTrainer, TrainingConfig, ModelType, TaskType, SplitStrategy
    import pandas as pd
    import numpy as np

    logger.info("=" * 80)
    logger.info("MODEL TRAINING")
    logger.info("=" * 80)

    # Parse model types
    if args.model_types:
        model_types = [ModelType(mt.strip()) for mt in args.model_types.split(',')]
    else:
        model_types = [ModelType.XGBOOST_REGRESSOR]

    logger.info(f"Training models: {', '.join([mt.value for mt in model_types])}")

    # Load data
    logger.info("\nLoading training data...")

    if args.data_file:
        # Load from file
        data = pd.read_csv(args.data_file)
        logger.info(f"Loaded {len(data)} samples from {args.data_file}")

        # Assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        # Generate sample data for demonstration
        logger.info("Generating sample training data...")
        np.random.seed(42)
        n_samples = args.n_samples or 1000
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Create target based on task type
        if args.task_type == 'regression':
            y = X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5
        else:  # classification
            y = (X.iloc[:, :5].sum(axis=1) > 0).astype(int)

        logger.info(f"Generated {n_samples} samples with {n_features} features")

    # Train each model
    results = {}

    for model_type in model_types:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training {model_type.value}")
        logger.info(f"{'=' * 80}")

        # Create configuration
        config = TrainingConfig(
            model_type=model_type,
            task_type=TaskType(args.task_type),
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            split_strategy=SplitStrategy(args.split_strategy),
            use_cross_validation=not args.no_cross_validation,
            cv_folds=args.cv_folds,
            tune_hyperparameters=args.tune_hyperparameters,
            save_model=not args.no_save,
            model_dir=args.model_dir,
            verbose=1 if args.verbose else 0
        )

        # Create trainer
        trainer = ModelTrainer(config)

        # Train model
        result = trainer.train(X, y)
        results[model_type.value] = result

        # Display results
        logger.info(f"\nTraining completed in {result.training_time:.2f} seconds")
        logger.info(f"Test Metrics:")
        for metric, value in result.test_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")

        if result.cv_scores:
            logger.info(f"Cross-Validation: {result.cv_scores['mean_score']:.6f} (+/- {result.cv_scores['std_score']:.6f})")

    # Compare models if multiple were trained
    if len(results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)

        comparison = []
        for model_name, result in results.items():
            primary_metric = 'r2' if args.task_type == 'regression' else 'accuracy'
            comparison.append({
                'model': model_name,
                'test_score': result.test_metrics.get(primary_metric, 0),
                'training_time': result.training_time
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('test_score', ascending=False)

        print("\n" + comparison_df.to_string(index=False))

        best_model = comparison_df.iloc[0]['model']
        logger.info(f"\nBest model: {best_model}")

    logger.info("\n" + "=" * 80)
    logger.info("Model training completed successfully!")
    logger.info("=" * 80)


# ============================================================================
# Command: collect-data
# ============================================================================

def cmd_collect_data(args):
    """
    Collect market data from various sources

    This command:
    1. Connects to data sources
    2. Fetches historical or real-time data
    3. Validates and cleans data
    4. Stores data in database
    """
    logger.info("=" * 80)
    logger.info("DATA COLLECTION")
    logger.info("=" * 80)

    # Parse sources
    sources = args.sources.split(',') if args.sources else ['all']
    logger.info(f"Data sources: {', '.join(sources)}")

    # Parse symbols
    symbols = args.symbols.split(',') if args.symbols else ['THYAO', 'GARAN', 'AKBNK']
    logger.info(f"Symbols: {', '.join(symbols)}")

    # Parse date range
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    else:
        start_date = pd.to_datetime('today') - pd.Timedelta(days=30)

    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    else:
        end_date = pd.to_datetime('today')

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    collected_data = {}

    # Collect BIST price data
    if 'bist' in sources or 'all' in sources:
        logger.info("\nCollecting BIST price data...")
        try:
            from src.data.collectors.bist_collector import BISTCollector

            collector = BISTCollector()

            for symbol in symbols:
                logger.info(f"  Fetching data for {symbol}...")
                try:
                    data = collector.fetch_stock_data(
                        symbol=symbol,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )

                    if data is not None and not data.empty:
                        collected_data[f'bist_{symbol}'] = data
                        logger.info(f"    Collected {len(data)} records")
                    else:
                        logger.warning(f"    No data collected for {symbol}")

                except Exception as e:
                    logger.error(f"    Error collecting data for {symbol}: {e}")

        except ImportError as e:
            logger.warning(f"BIST collector not available: {e}")
        except Exception as e:
            logger.error(f"Error in BIST data collection: {e}")

    # Collect fundamental data
    if 'fundamental' in sources or 'all' in sources:
        logger.info("\nCollecting fundamental data...")
        try:
            from src.data.collectors.fundamental_collector import FundamentalDataCollector

            collector = FundamentalDataCollector()

            for symbol in symbols:
                logger.info(f"  Fetching fundamentals for {symbol}...")
                try:
                    data = collector.fetch_financials(symbol)

                    if data:
                        collected_data[f'fundamental_{symbol}'] = data
                        logger.info(f"    Collected fundamental data")
                    else:
                        logger.warning(f"    No fundamental data for {symbol}")

                except Exception as e:
                    logger.error(f"    Error collecting fundamentals for {symbol}: {e}")

        except ImportError as e:
            logger.warning(f"Fundamental collector not available: {e}")
        except Exception as e:
            logger.error(f"Error in fundamental data collection: {e}")

    # Collect macro data
    if 'macro' in sources or 'all' in sources:
        logger.info("\nCollecting macroeconomic data...")
        try:
            from src.data.collectors.macro_collector import MacroDataCollector

            collector = MacroDataCollector()
            logger.info("  Fetching macro indicators...")

            try:
                data = collector.collect_macro_data(
                    start_date=start_date,
                    end_date=end_date
                )

                if data:
                    collected_data['macro'] = data
                    logger.info(f"    Collected macro data")

            except Exception as e:
                logger.error(f"    Error collecting macro data: {e}")

        except ImportError as e:
            logger.warning(f"Macro collector not available: {e}")
        except Exception as e:
            logger.error(f"Error in macro data collection: {e}")

    # Collect news data
    if 'news' in sources or 'all' in sources:
        logger.info("\nCollecting news data...")
        try:
            from src.data.collectors.news_collector import NewsCollector

            collector = NewsCollector()

            for symbol in symbols:
                logger.info(f"  Fetching news for {symbol}...")
                try:
                    news = collector.fetch_news(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )

                    if news:
                        collected_data[f'news_{symbol}'] = news
                        logger.info(f"    Collected {len(news)} news articles")
                    else:
                        logger.warning(f"    No news data for {symbol}")

                except Exception as e:
                    logger.error(f"    Error collecting news for {symbol}: {e}")

        except ImportError as e:
            logger.warning(f"News collector not available: {e}")
        except Exception as e:
            logger.error(f"Error in news data collection: {e}")

    # Save collected data
    if args.output and collected_data:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving collected data to {output_path}...")

        for name, data in collected_data.items():
            try:
                if isinstance(data, pd.DataFrame):
                    file_path = output_path / f"{name}.csv"
                    data.to_csv(file_path)
                    logger.info(f"  Saved {name} to {file_path}")
                else:
                    file_path = output_path / f"{name}.json"
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                    logger.info(f"  Saved {name} to {file_path}")

            except Exception as e:
                logger.error(f"  Error saving {name}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Data collection completed! Collected {len(collected_data)} datasets")
    logger.info("=" * 80)


# ============================================================================
# Command: show-portfolio
# ============================================================================

def cmd_show_portfolio(args):
    """
    Display portfolio status and positions

    This command:
    1. Loads portfolio from file
    2. Fetches current market prices
    3. Calculates P&L and metrics
    4. Displays portfolio summary
    """
    from src.portfolio.manager import PortfolioManager
    import pandas as pd

    logger.info("=" * 80)
    logger.info("PORTFOLIO STATUS")
    logger.info("=" * 80)

    # Load portfolio
    if args.portfolio_file:
        logger.info(f"Loading portfolio from {args.portfolio_file}")

        try:
            if args.portfolio_file.endswith('.json'):
                portfolio = PortfolioManager.load_from_json(args.portfolio_file)
            elif args.portfolio_file.endswith('.pkl'):
                portfolio = PortfolioManager.load_from_pickle(args.portfolio_file)
            else:
                logger.error("Unsupported file format. Use .json or .pkl")
                return

        except FileNotFoundError:
            logger.error(f"Portfolio file not found: {args.portfolio_file}")
            return
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return
    else:
        # Create demo portfolio
        logger.info("Creating demo portfolio...")
        from src.portfolio.manager import create_portfolio

        portfolio = create_portfolio(
            name="Demo BIST Portfolio",
            initial_cash=100000.0,
            cost_basis_method="AVERAGE",
            currency="TRY"
        )

        # Add some demo positions
        portfolio.buy(symbol="THYAO", shares=100, price=250.0, commission=10.0)
        portfolio.buy(symbol="GARAN", shares=200, price=85.0, commission=15.0)
        portfolio.buy(symbol="AKBNK", shares=150, price=45.0, commission=12.0)

    logger.info(f"\nPortfolio: {portfolio.name}")
    logger.info(f"Created: {portfolio.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get current prices (in production, fetch from API)
    current_prices = {}

    if args.prices_file:
        # Load prices from file
        try:
            prices_df = pd.read_csv(args.prices_file)
            current_prices = dict(zip(prices_df['symbol'], prices_df['price']))
            logger.info(f"Loaded prices from {args.prices_file}")
        except Exception as e:
            logger.error(f"Error loading prices: {e}")

    # Use default prices for demo
    if not current_prices:
        logger.info("Using demo prices...")
        for symbol in portfolio.positions.keys():
            # Demo prices (in production, fetch real prices)
            demo_prices = {
                'THYAO': 265.0,
                'GARAN': 90.0,
                'AKBNK': 48.0
            }
            current_prices[symbol] = demo_prices.get(symbol, 100.0)

    # Get portfolio summary
    summary = portfolio.get_portfolio_summary(current_prices)

    # Display summary
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    print(f"Total Value:        {summary['total_value']:>15,.2f} {summary['currency']}")
    print(f"Cash:               {summary['cash']:>15,.2f} {summary['currency']}")
    print(f"Positions Value:    {summary['positions_value']:>15,.2f} {summary['currency']}")
    print(f"Total Cost Basis:   {summary['total_cost_basis']:>15,.2f} {summary['currency']}")
    print("")
    print(f"Unrealized P&L:     {summary['unrealized_pnl']:>15,.2f} {summary['currency']}")
    print(f"Realized P&L:       {summary['realized_pnl']:>15,.2f} {summary['currency']}")
    print(f"Total P&L:          {summary['total_pnl']:>15,.2f} {summary['currency']}")
    print(f"Total Return:       {summary['total_return_pct']:>15.2f}%")
    print(f"Total Commissions:  {summary['total_commissions']:>15,.2f} {summary['currency']}")
    print("")
    print(f"Number of Positions: {summary['num_positions']}")

    # Display positions
    if summary['positions']:
        print("\n" + "=" * 80)
        print("POSITIONS")
        print("=" * 80)
        print(f"{'Symbol':<10} {'Shares':>10} {'Cost Basis':>12} {'Current':>12} {'Market Value':>15} {'P&L':>12} {'P&L %':>10}")
        print("-" * 80)

        for pos in summary['positions']:
            print(f"{pos['symbol']:<10} "
                  f"{pos['shares']:>10.2f} "
                  f"{pos['cost_basis']:>12.2f} "
                  f"{pos['current_price']:>12.2f} "
                  f"{pos['market_value']:>15,.2f} "
                  f"{pos['unrealized_pnl']:>12,.2f} "
                  f"{pos['unrealized_pnl_pct']:>9.2f}%")

    # Display allocation
    if args.show_allocation:
        allocation = portfolio.get_allocation(current_prices)

        if not allocation.empty:
            print("\n" + "=" * 80)
            print("ALLOCATION")
            print("=" * 80)
            print(allocation.to_string(index=False))

    # Display transaction history
    if args.show_transactions:
        history = portfolio.get_transaction_history()

        if not history.empty:
            print("\n" + "=" * 80)
            print("RECENT TRANSACTIONS (Last 10)")
            print("=" * 80)
            recent = history.tail(10)
            print(recent[['timestamp', 'symbol', 'transaction_type', 'shares', 'price', 'commission']].to_string(index=False))

    # Calculate and display performance metrics
    if args.show_metrics:
        metrics = portfolio.calculate_performance_metrics(current_prices)

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        print(f"Total Return:           {metrics['total_return_pct']:.2f}%")
        print(f"Annualized Return:      {metrics['annualized_return_pct']:.2f}%")
        print(f"Win Rate:               {metrics['win_rate']:.2f}%")
        print(f"Profitable Positions:   {metrics['profitable_positions']}/{metrics['total_positions']}")
        print(f"Days Invested:          {metrics['days_invested']}")
        print(f"Commission % of Capital: {metrics['commission_pct_of_invested']:.3f}%")

    # Save portfolio if output specified
    if args.output:
        output_path = Path(args.output)

        try:
            if output_path.suffix == '.json':
                portfolio.save_to_json(str(output_path))
            elif output_path.suffix == '.pkl':
                portfolio.save_to_pickle(str(output_path))
            else:
                logger.warning("Unknown output format, saving as JSON")
                portfolio.save_to_json(str(output_path.with_suffix('.json')))

            logger.info(f"\nPortfolio saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Portfolio display completed!")
    logger.info("=" * 80)


# ============================================================================
# Main CLI Setup
# ============================================================================

def create_parser():
    """Create the argument parser with all commands and options"""

    parser = argparse.ArgumentParser(
        description='BIST AI Trading System - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate signals for specific symbols
  python -m src.ui.cli run-signals --symbols THYAO,GARAN,AKBNK --output signals.json

  # Run backtest with custom parameters
  python -m src.ui.cli backtest --symbol THYAO --start-date 2023-01-01 --initial-capital 100000

  # Train multiple models
  python -m src.ui.cli train-models --model-types xgboost_regressor,lightgbm_regressor --tune-hyperparameters

  # Collect all available data
  python -m src.ui.cli collect-data --sources all --symbols THYAO,GARAN --output data/

  # Show portfolio with full details
  python -m src.ui.cli show-portfolio --portfolio-file portfolio.json --show-allocation --show-metrics
        """
    )

    # Add version
    parser.add_argument('--version', action='version', version='BIST AI Trading System v1.0.0')

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ========================================================================
    # run-signals command
    # ========================================================================

    parser_signals = subparsers.add_parser(
        'run-signals',
        help='Generate trading signals',
        description='Generate trading signals for BIST stocks using AI models'
    )

    parser_signals.add_argument(
        '--symbols',
        type=str,
        default='THYAO,GARAN,AKBNK',
        help='Comma-separated list of stock symbols (default: THYAO,GARAN,AKBNK)'
    )

    parser_signals.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for signals (default: 0.5)'
    )

    parser_signals.add_argument(
        '--no-dynamic-thresholds',
        action='store_true',
        help='Disable dynamic threshold adjustment'
    )

    parser_signals.add_argument(
        '--no-risk-adjustment',
        action='store_true',
        help='Disable risk-based signal adjustment'
    )

    parser_signals.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file for generated signals (JSON format)'
    )

    parser_signals.set_defaults(func=cmd_run_signals)

    # ========================================================================
    # backtest command
    # ========================================================================

    parser_backtest = subparsers.add_parser(
        'backtest',
        help='Run backtesting',
        description='Backtest trading strategies on historical data'
    )

    parser_backtest.add_argument(
        '--symbol',
        type=str,
        default='THYAO',
        help='Stock symbol to backtest (default: THYAO)'
    )

    parser_backtest.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser_backtest.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser_backtest.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital in TRY (default: 100000)'
    )

    parser_backtest.add_argument(
        '--commission-rate',
        type=float,
        default=0.001,
        help='Commission rate as decimal (default: 0.001 = 0.1%%)'
    )

    parser_backtest.add_argument(
        '--slippage-rate',
        type=float,
        default=0.0005,
        help='Slippage rate as decimal (default: 0.0005 = 0.05%%)'
    )

    parser_backtest.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as fraction of equity (default: 0.1 = 10%%)'
    )

    parser_backtest.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence to execute trade (default: 0.5)'
    )

    parser_backtest.add_argument(
        '--no-stop-loss',
        action='store_true',
        help='Disable stop loss orders'
    )

    parser_backtest.add_argument(
        '--no-take-profit',
        action='store_true',
        help='Disable take profit orders'
    )

    parser_backtest.add_argument(
        '--signals-file',
        type=str,
        help='CSV file containing pre-computed signals'
    )

    parser_backtest.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output directory for backtest results'
    )

    parser_backtest.set_defaults(func=cmd_backtest)

    # ========================================================================
    # train-models command
    # ========================================================================

    parser_train = subparsers.add_parser(
        'train-models',
        help='Train ML models',
        description='Train or retrain machine learning models'
    )

    parser_train.add_argument(
        '--model-types',
        type=str,
        default='xgboost_regressor',
        help='Comma-separated model types to train (lstm, gru, xgboost_regressor, lightgbm_regressor, random_forest_classifier, ann_classifier)'
    )

    parser_train.add_argument(
        '--task-type',
        type=str,
        choices=['regression', 'classification', 'time_series_forecasting'],
        default='regression',
        help='Type of ML task (default: regression)'
    )

    parser_train.add_argument(
        '--data-file',
        type=str,
        help='CSV file containing training data'
    )

    parser_train.add_argument(
        '--n-samples',
        type=int,
        help='Number of samples to generate if no data file provided'
    )

    parser_train.add_argument(
        '--train-size',
        type=float,
        default=0.7,
        help='Training set size as fraction (default: 0.7)'
    )

    parser_train.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Validation set size as fraction (default: 0.15)'
    )

    parser_train.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Test set size as fraction (default: 0.15)'
    )

    parser_train.add_argument(
        '--split-strategy',
        type=str,
        choices=['time_series', 'random', 'stratified'],
        default='time_series',
        help='Data splitting strategy (default: time_series)'
    )

    parser_train.add_argument(
        '--no-cross-validation',
        action='store_true',
        help='Disable cross-validation'
    )

    parser_train.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser_train.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Enable hyperparameter tuning'
    )

    parser_train.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )

    parser_train.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )

    parser_train.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser_train.set_defaults(func=cmd_train_models)

    # ========================================================================
    # collect-data command
    # ========================================================================

    parser_collect = subparsers.add_parser(
        'collect-data',
        help='Collect market data',
        description='Collect market data from various sources'
    )

    parser_collect.add_argument(
        '--sources',
        type=str,
        default='all',
        help='Comma-separated data sources (bist, fundamental, macro, news, whale, all) (default: all)'
    )

    parser_collect.add_argument(
        '--symbols',
        type=str,
        default='THYAO,GARAN,AKBNK',
        help='Comma-separated stock symbols (default: THYAO,GARAN,AKBNK)'
    )

    parser_collect.add_argument(
        '--start-date',
        type=str,
        help='Start date for data collection (YYYY-MM-DD)'
    )

    parser_collect.add_argument(
        '--end-date',
        type=str,
        help='End date for data collection (YYYY-MM-DD)'
    )

    parser_collect.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output directory for collected data'
    )

    parser_collect.set_defaults(func=cmd_collect_data)

    # ========================================================================
    # show-portfolio command
    # ========================================================================

    parser_portfolio = subparsers.add_parser(
        'show-portfolio',
        help='Display portfolio status',
        description='Display portfolio positions, P&L, and performance metrics'
    )

    parser_portfolio.add_argument(
        '--portfolio-file',
        type=str,
        help='Portfolio file to load (.json or .pkl)'
    )

    parser_portfolio.add_argument(
        '--prices-file',
        type=str,
        help='CSV file with current prices (columns: symbol, price)'
    )

    parser_portfolio.add_argument(
        '--show-allocation',
        action='store_true',
        help='Show portfolio allocation breakdown'
    )

    parser_portfolio.add_argument(
        '--show-transactions',
        action='store_true',
        help='Show recent transaction history'
    )

    parser_portfolio.add_argument(
        '--show-metrics',
        action='store_true',
        help='Show performance metrics'
    )

    parser_portfolio.add_argument(
        '--output',
        '-o',
        type=str,
        help='Save portfolio to file (.json or .pkl)'
    )

    parser_portfolio.set_defaults(func=cmd_show_portfolio)

    return parser


def main():
    """Main entry point for the CLI"""

    # Create parser
    parser = create_parser()

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
