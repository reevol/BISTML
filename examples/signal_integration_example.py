"""
Complete Signal Pipeline Integration Example

This example demonstrates the full signal generation and prioritization pipeline:
1. Generate signals from multiple ML models
2. Collect whale activity (WAI) data
3. Analyze news sentiment
4. Prioritize signals using multi-factor ranking
5. Generate actionable trading recommendations

This represents the complete workflow for the BIST AI Trading System.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.prioritizer import (
    SignalPrioritizer,
    create_signal_input,
    SignalDirection,
    PrioritizationStrategy
)


def simulate_ml_model_predictions(symbol: str) -> dict:
    """
    Simulate predictions from multiple ML models.
    In production, these would be actual LSTM, GRU, XGBoost, LightGBM predictions.
    """
    # Simulate different models with slight variations
    base_prediction = np.random.uniform(-0.05, 0.05)
    noise = 0.01

    return {
        'LSTM': base_prediction + np.random.normal(0, noise),
        'GRU': base_prediction + np.random.normal(0, noise),
        'XGBoost': base_prediction + np.random.normal(0, noise),
        'LightGBM': base_prediction + np.random.normal(0, noise)
    }


def simulate_wai_calculation(symbol: str) -> float:
    """
    Simulate WAI (Whale Activity Index) calculation.
    In production, this would use actual broker distribution data.
    """
    # Simulate WAI scores (0-100)
    return np.random.uniform(20, 95)


def simulate_sentiment_analysis(symbol: str) -> float:
    """
    Simulate news sentiment analysis.
    In production, this would use actual news articles and NLP models.
    """
    # Simulate sentiment scores (-1 to 1)
    return np.random.uniform(-0.8, 0.8)


def calculate_confidence_score(model_predictions: dict) -> float:
    """
    Calculate confidence score based on model predictions.
    Higher agreement and stronger magnitude = higher confidence.
    """
    predictions = list(model_predictions.values())

    # Check agreement (all same sign)
    signs = [np.sign(p) for p in predictions]
    agreement = len(set(signs)) == 1 and signs[0] != 0

    # Calculate magnitude
    avg_magnitude = np.mean(np.abs(predictions))

    # Calculate consistency (inverse of coefficient of variation)
    cv = np.std(predictions) / (np.abs(np.mean(predictions)) + 1e-10)
    consistency = 1 / (1 + cv)

    # Combine into confidence score
    if agreement:
        confidence = min(100, 50 + avg_magnitude * 1000 + consistency * 30)
    else:
        confidence = max(30, 50 - cv * 20)

    return confidence


def determine_signal_direction(model_predictions: dict, threshold: float = 0.015) -> SignalDirection:
    """
    Determine signal direction from model predictions.
    """
    avg_prediction = np.mean(list(model_predictions.values()))

    if avg_prediction > threshold * 1.5:
        return SignalDirection.STRONG_BUY
    elif avg_prediction > threshold:
        return SignalDirection.BUY
    elif avg_prediction < -threshold * 1.5:
        return SignalDirection.STRONG_SELL
    elif avg_prediction < -threshold:
        return SignalDirection.SELL
    else:
        return SignalDirection.HOLD


def calculate_target_price(current_price: float, predictions: dict) -> float:
    """Calculate target price from model predictions"""
    avg_return = np.mean(list(predictions.values()))
    return current_price * (1 + avg_return)


def generate_signal_for_symbol(
    symbol: str,
    current_price: float,
    include_wai: bool = True,
    include_sentiment: bool = True
) -> dict:
    """
    Generate a complete signal for a symbol with all components.

    This simulates the full data pipeline:
    - ML model predictions
    - Whale activity analysis
    - News sentiment analysis
    - Confidence calculation
    """
    # Step 1: Get model predictions
    model_predictions = simulate_ml_model_predictions(symbol)

    # Step 2: Calculate confidence
    confidence_score = calculate_confidence_score(model_predictions)

    # Step 3: Determine signal direction
    signal_direction = determine_signal_direction(model_predictions)

    # Step 4: Calculate target price
    target_price = calculate_target_price(current_price, model_predictions)

    # Step 5: Get WAI score (if enabled)
    wai_score = simulate_wai_calculation(symbol) if include_wai else None

    # Step 6: Get sentiment (if enabled)
    news_sentiment = simulate_sentiment_analysis(symbol) if include_sentiment else None

    return {
        'symbol': symbol,
        'current_price': current_price,
        'target_price': target_price,
        'signal_direction': signal_direction,
        'confidence_score': confidence_score,
        'wai_score': wai_score,
        'news_sentiment': news_sentiment,
        'model_predictions': model_predictions
    }


def example_1_full_pipeline():
    """Example 1: Complete signal generation and prioritization pipeline"""
    print("=" * 80)
    print("Example 1: Full Signal Generation and Prioritization Pipeline")
    print("=" * 80)

    # BIST 30 stocks with simulated current prices
    stocks = {
        'THYAO': 100.50,
        'AKBNK': 50.25,
        'GARAN': 95.75,
        'ISCTR': 8.45,
        'EREGL': 45.30,
        'ASELS': 68.75,
        'KCHOL': 125.50,
        'SAHOL': 75.20,
        'TUPRS': 125.50,
        'PETKM': 85.40
    }

    print("\nStep 1: Generating signals for 10 BIST stocks...")
    print("-" * 80)

    # Generate signals for all stocks
    raw_signals = []
    for symbol, price in stocks.items():
        signal_data = generate_signal_for_symbol(symbol, price)
        raw_signals.append(signal_data)

        print(f"{symbol}: {signal_data['signal_direction'].name:12} "
              f"(Conf: {signal_data['confidence_score']:.1f}, "
              f"WAI: {signal_data['wai_score']:.1f}, "
              f"Sent: {signal_data['news_sentiment']:+.2f})")

    print("\nStep 2: Converting to SignalInput objects...")
    print("-" * 80)

    signal_inputs = []
    for signal_data in raw_signals:
        signal_input = create_signal_input(
            symbol=signal_data['symbol'],
            signal_direction=signal_data['signal_direction'].name,
            confidence_score=signal_data['confidence_score'],
            wai_score=signal_data['wai_score'],
            news_sentiment=signal_data['news_sentiment'],
            model_predictions=signal_data['model_predictions'],
            current_price=signal_data['current_price'],
            target_price=signal_data['target_price']
        )
        signal_inputs.append(signal_input)

    print(f"Created {len(signal_inputs)} signal inputs")

    print("\nStep 3: Prioritizing signals with BALANCED strategy...")
    print("-" * 80)

    prioritizer = SignalPrioritizer(
        strategy=PrioritizationStrategy.BALANCED,
        min_confidence_threshold=50.0,
        min_wai_threshold=30.0,
        min_agreement_threshold=0.5
    )

    prioritized_signals = prioritizer.prioritize_signals(signal_inputs, return_all=False)

    print(f"Prioritized {len(prioritized_signals)} actionable signals")

    print("\nStep 4: Top Trading Recommendations:")
    print("-" * 80)

    df = prioritizer.to_dataframe(prioritized_signals)
    print(df[['rank', 'symbol', 'signal', 'risk_adjusted_score',
              'signal_strength', 'expected_return_pct', 'model_agreement_pct']].to_string(index=False))

    # Get top BUY and SELL signals
    buy_signals = [s for s in prioritized_signals
                   if s.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]]
    sell_signals = [s for s in prioritized_signals
                    if s.signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]]

    print("\n" + "=" * 80)
    print("TRADING RECOMMENDATIONS")
    print("=" * 80)

    if buy_signals:
        top_buy = buy_signals[0]
        print(f"\n🟢 PRIMARY BUY RECOMMENDATION: {top_buy.symbol}")
        print(f"   Signal: {top_buy.signal_direction.name}")
        print(f"   Priority Score: {top_buy.priority_score:.2f}")
        print(f"   Risk-Adjusted Score: {top_buy.risk_adjusted_score:.2f}")
        print(f"   Signal Strength: {top_buy.signal_strength}")
        print(f"   Expected Return: {top_buy.expected_return:.2f}%")
        print(f"   Current Price: {top_buy.current_price:.2f}")
        print(f"   Target Price: {top_buy.target_price:.2f}")
        print(f"   Model Agreement: {top_buy.model_agreement_pct*100:.1f}%")
        print(f"   WAI Score: {top_buy.raw_scores['wai']:.1f}")
        print(f"   News Sentiment: {top_buy.raw_scores['sentiment']:+.2f}")

    if sell_signals:
        top_sell = sell_signals[0]
        print(f"\n🔴 PRIMARY SELL RECOMMENDATION: {top_sell.symbol}")
        print(f"   Signal: {top_sell.signal_direction.name}")
        print(f"   Priority Score: {top_sell.priority_score:.2f}")
        print(f"   Risk-Adjusted Score: {top_sell.risk_adjusted_score:.2f}")
        print(f"   Expected Return: {top_sell.expected_return:.2f}%")

    print("\n" + "=" * 80)


def example_2_strategy_comparison():
    """Example 2: Compare different strategies on same signals"""
    print("\n" + "=" * 80)
    print("Example 2: Strategy Comparison for Trading Decisions")
    print("=" * 80)

    # Generate signals for 5 stocks
    stocks = {
        'THYAO': 100.0,
        'AKBNK': 50.0,
        'EREGL': 45.0,
        'ASELS': 68.0,
        'GARAN': 95.0
    }

    signal_inputs = []
    for symbol, price in stocks.items():
        signal_data = generate_signal_for_symbol(symbol, price)
        signal_input = create_signal_input(
            symbol=signal_data['symbol'],
            signal_direction=signal_data['signal_direction'].name,
            confidence_score=signal_data['confidence_score'],
            wai_score=signal_data['wai_score'],
            news_sentiment=signal_data['news_sentiment'],
            model_predictions=signal_data['model_predictions'],
            current_price=signal_data['current_price'],
            target_price=signal_data['target_price']
        )
        signal_inputs.append(signal_input)

    # Test different strategies
    strategies = [
        ('BALANCED', PrioritizationStrategy.BALANCED),
        ('WHALE_FOCUSED', PrioritizationStrategy.WHALE_FOCUSED),
        ('CONSERVATIVE', PrioritizationStrategy.CONSERVATIVE),
        ('AGGRESSIVE', PrioritizationStrategy.AGGRESSIVE)
    ]

    results = {}
    for strategy_name, strategy in strategies:
        prioritizer = SignalPrioritizer(strategy=strategy, verbose=False)
        prioritized = prioritizer.prioritize_signals(signal_inputs, return_all=False)

        # Get top signal
        if prioritized:
            top_signal = prioritized[0]
            results[strategy_name] = {
                'top_symbol': top_signal.symbol,
                'signal': top_signal.signal_direction.name,
                'score': top_signal.risk_adjusted_score,
                'strength': top_signal.signal_strength
            }

    print("\nTop Signal by Strategy:")
    print("-" * 80)
    for strategy_name, result in results.items():
        print(f"{strategy_name:20} -> {result['top_symbol']:6} "
              f"{result['signal']:12} (Score: {result['score']:.2f}, "
              f"Strength: {result['strength']})")

    print("\n" + "=" * 80)


def example_3_time_series_monitoring():
    """Example 3: Simulate continuous signal monitoring"""
    print("\n" + "=" * 80)
    print("Example 3: Real-Time Signal Monitoring Simulation")
    print("=" * 80)

    symbol = 'THYAO'
    current_price = 100.0

    print(f"\nMonitoring {symbol} over 5 time periods (30-min intervals)...")
    print("-" * 80)

    prioritizer = SignalPrioritizer(strategy=PrioritizationStrategy.BALANCED)

    for i in range(5):
        timestamp = datetime.now() + timedelta(minutes=i*30)

        # Generate new signal
        signal_data = generate_signal_for_symbol(symbol, current_price)
        signal_input = create_signal_input(
            symbol=signal_data['symbol'],
            signal_direction=signal_data['signal_direction'].name,
            confidence_score=signal_data['confidence_score'],
            wai_score=signal_data['wai_score'],
            news_sentiment=signal_data['news_sentiment'],
            model_predictions=signal_data['model_predictions'],
            current_price=signal_data['current_price'],
            target_price=signal_data['target_price'],
            timestamp=timestamp
        )

        # Prioritize
        prioritized = prioritizer.prioritize_signal(signal_input)

        print(f"\n{timestamp.strftime('%H:%M:%S')} - {symbol}")
        print(f"  Signal: {prioritized.signal_direction.name:12} | "
              f"Score: {prioritized.risk_adjusted_score:5.2f} | "
              f"Strength: {prioritized.signal_strength:12} | "
              f"Agreement: {prioritized.model_agreement_pct*100:5.1f}%")

        # Simulate price movement
        avg_prediction = np.mean(list(signal_data['model_predictions'].values()))
        current_price *= (1 + avg_prediction + np.random.normal(0, 0.005))

    print("\n" + "=" * 80)


def example_4_portfolio_integration():
    """Example 4: Integration with portfolio management"""
    print("\n" + "=" * 80)
    print("Example 4: Portfolio-Aware Signal Prioritization")
    print("=" * 80)

    # Existing portfolio
    portfolio = {
        'THYAO': {'shares': 100, 'avg_price': 98.0},
        'AKBNK': {'shares': 200, 'avg_price': 48.0},
        'GARAN': {'shares': 50, 'avg_price': 90.0}
    }

    # Current market prices
    current_prices = {
        'THYAO': 100.5,
        'AKBNK': 50.25,
        'GARAN': 95.75,
        'EREGL': 45.30,
        'ASELS': 68.75
    }

    print("\nCurrent Portfolio:")
    print("-" * 80)
    for symbol, position in portfolio.items():
        current_price = current_prices[symbol]
        pl_pct = ((current_price - position['avg_price']) / position['avg_price']) * 100
        print(f"{symbol}: {position['shares']} shares @ {position['avg_price']:.2f} "
              f"(Current: {current_price:.2f}, P/L: {pl_pct:+.2f}%)")

    # Generate signals for all stocks
    print("\nGenerating signals for portfolio and watchlist...")
    print("-" * 80)

    signal_inputs = []
    for symbol, price in current_prices.items():
        signal_data = generate_signal_for_symbol(symbol, price)
        signal_input = create_signal_input(
            symbol=signal_data['symbol'],
            signal_direction=signal_data['signal_direction'].name,
            confidence_score=signal_data['confidence_score'],
            wai_score=signal_data['wai_score'],
            news_sentiment=signal_data['news_sentiment'],
            model_predictions=signal_data['model_predictions'],
            current_price=signal_data['current_price'],
            target_price=signal_data['target_price'],
            additional_metadata={'in_portfolio': symbol in portfolio}
        )
        signal_inputs.append(signal_input)

    # Prioritize
    prioritizer = SignalPrioritizer(strategy=PrioritizationStrategy.CONSERVATIVE)
    prioritized_signals = prioritizer.prioritize_signals(signal_inputs, return_all=False)

    # Separate portfolio and new opportunities
    portfolio_signals = [s for s in prioritized_signals
                        if s.additional_metadata.get('in_portfolio')]
    new_opportunities = [s for s in prioritized_signals
                        if not s.additional_metadata.get('in_portfolio')]

    print("\nPortfolio Signals (SELL alerts):")
    print("-" * 80)
    for signal in portfolio_signals:
        if signal.signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
            print(f"⚠️  {signal.symbol}: {signal.signal_direction.name} "
                  f"(Score: {signal.risk_adjusted_score:.2f}, "
                  f"Expected Return: {signal.expected_return:+.2f}%)")

    print("\nNew Opportunities (BUY signals):")
    print("-" * 80)
    for signal in new_opportunities:
        if signal.signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            print(f"✨ {signal.symbol}: {signal.signal_direction.name} "
                  f"(Score: {signal.risk_adjusted_score:.2f}, "
                  f"Target: {signal.target_price:.2f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPLETE SIGNAL PIPELINE INTEGRATION")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run all examples
    example_1_full_pipeline()
    example_2_strategy_comparison()
    example_3_time_series_monitoring()
    example_4_portfolio_integration()

    print("\n" + "=" * 80)
    print("Integration examples completed successfully!")
    print("=" * 80)
