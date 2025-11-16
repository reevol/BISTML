#!/usr/bin/env python3
"""
Example: Using LLM Integration for Turkish Financial News Analysis

This script demonstrates how to use the LLM integration module to analyze
Turkish financial news and generate nuanced impact scores for BIST stocks.

Usage:
    python examples/llm_news_analysis_example.py

Requirements:
    - OpenAI API key set in environment (OPENAI_API_KEY)
    OR
    - Local LLM model downloaded

Author: BIST AI Trading System
Date: 2025-11-16
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nlp.llm_integration import (
    LLMNewsAnalyzer,
    create_analyzer,
    NewsImpactScore,
    SentimentImpact
)
from src.data.collectors.news_collector import (
    NewsCollectorManager,
    NewsArticle,
    NewsSource
)


def example_1_basic_analysis():
    """Example 1: Basic news analysis with OpenAI"""
    print("\n" + "=" * 70)
    print("Example 1: Basic News Analysis with OpenAI GPT")
    print("=" * 70)

    # Create analyzer
    analyzer = create_analyzer(
        provider="openai",
        model_name="gpt-3.5-turbo",  # or "gpt-4" for better results
        temperature=0.3,
        language="turkish"
    )

    # Sample news for AKBNK (Akbank)
    news_articles = [
        {
            'title': 'Akbank\'ın 9 aylık karı beklentileri aştı',
            'source': 'Bloomberg HT',
            'published_date': '2024-11-15',
            'content': '''Akbank\'ın 9 aylık net karı 45 milyar TL olarak açıklandı.
                Piyasa beklentisi 42 milyar TL idi. Kredilerde %15 büyüme kaydedildi.
                Mevduat büyümesi ise %18 seviyesinde gerçekleşti.''',
            'summary': 'Güçlü finansal performans'
        },
        {
            'title': 'BDDK\'dan yeni kredi düzenlemesi',
            'source': 'KAP',
            'published_date': '2024-11-14',
            'content': '''BDDK, konut kredisi faiz oranlarına yeni sınırlamalar getirdi.
                Bu düzenleme bankaların karlılığını etkileyebilir.''',
            'summary': 'Düzenleyici risk faktörü'
        },
        {
            'title': 'Yabancı yatırımcılar bankacılık hisselerinde alımda',
            'source': 'Investing.com',
            'published_date': '2024-11-13',
            'content': '''Son bir haftada yabancı yatırımcılar bankacılık sektöründe
                net 2 milyar TL alım yaptı. Akbank en çok alım yapılan hisseler arasında.''',
            'summary': 'Güçlü yabancı ilgisi'
        }
    ]

    # Analyze
    print("\nAnalyzing news for AKBNK...")
    result = analyzer.analyze_news_impact("AKBNK", news_articles)

    # Display results
    print_analysis_result(result)


def example_2_real_news_collection():
    """Example 2: Collect real news and analyze"""
    print("\n" + "=" * 70)
    print("Example 2: Real News Collection + LLM Analysis")
    print("=" * 70)

    # Collect real news
    print("\nCollecting news for THYAO (Turkish Airlines)...")
    news_manager = NewsCollectorManager()

    try:
        articles = news_manager.collect_for_stock("THYAO", days_back=7, limit_per_source=5)
        print(f"Collected {len(articles)} articles")

        if not articles:
            print("No articles found. Using sample data instead.")
            return example_1_basic_analysis()

        # Convert NewsArticle objects to dicts
        article_dicts = [article.to_dict() for article in articles]

        # Analyze with LLM
        print("\nAnalyzing with LLM...")
        analyzer = create_analyzer(provider="openai", language="turkish")
        result = analyzer.analyze_news_impact("THYAO", article_dicts)

        # Display results
        print_analysis_result(result)

    except Exception as e:
        print(f"Error collecting news: {e}")
        print("Falling back to sample data...")
        return example_1_basic_analysis()


def example_3_batch_analysis():
    """Example 3: Batch analysis for multiple stocks"""
    print("\n" + "=" * 70)
    print("Example 3: Batch Analysis for Multiple Stocks")
    print("=" * 70)

    # Sample news for multiple stocks
    stock_news_map = {
        'AKBNK': [
            {
                'title': 'Akbank kar açıkladı',
                'content': 'Güçlü finansal sonuçlar',
                'source': 'Bloomberg HT',
                'published_date': '2024-11-15'
            }
        ],
        'THYAO': [
            {
                'title': 'THY yeni uçak siparişi',
                'content': '15 milyar dolarlık sipariş',
                'source': 'Bloomberg HT',
                'published_date': '2024-11-15'
            }
        ],
        'EREGL': [
            {
                'title': 'Erdemir üretimi artırdı',
                'content': 'Çelik üretimi %10 arttı',
                'source': 'Bloomberg HT',
                'published_date': '2024-11-14'
            }
        ]
    }

    # Create analyzer
    analyzer = create_analyzer(provider="openai", temperature=0.3)

    # Batch analyze
    print("\nAnalyzing news for multiple stocks...")
    results = analyzer.batch_analyze(stock_news_map)

    # Display results
    print("\n" + "-" * 70)
    print("Batch Analysis Results:")
    print("-" * 70)

    for stock_code, result in results.items():
        print(f"\n{stock_code}:")
        print(f"  Impact: {result.impact_category.value} ({result.impact_score:+.2f})")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Key Factor: {result.key_factors[0] if result.key_factors else 'N/A'}")


def example_4_comparison_analysis():
    """Example 4: Compare different LLM models"""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Different LLM Models")
    print("=" * 70)

    # Sample news
    news = [
        {
            'title': 'Garanti BBVA\'dan güçlü kar açıklaması',
            'content': '''Garanti BBVA 3. çeyrek net karını 18.5 milyar TL olarak açıkladı.
                Beklentiler 17 milyar TL civarındaydı. ROE %25 seviyesine ulaştı.''',
            'source': 'Bloomberg HT',
            'published_date': '2024-11-15'
        }
    ]

    models = ["gpt-3.5-turbo", "gpt-4"]  # Add more models as needed

    print(f"\nAnalyzing same news with different models:")
    print(f"Stock: GARAN")

    for model_name in models:
        try:
            print(f"\n{'-' * 70}")
            print(f"Model: {model_name}")
            print(f"{'-' * 70}")

            analyzer = create_analyzer(
                provider="openai",
                model_name=model_name,
                temperature=0.3
            )

            result = analyzer.analyze_news_impact("GARAN", news)

            print(f"Impact Score: {result.impact_score:+.2f}")
            print(f"Category: {result.impact_category.value}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"Reasoning: {result.reasoning[:200]}...")

        except Exception as e:
            print(f"Error with {model_name}: {e}")


def example_5_trading_signal_integration():
    """Example 5: Integrate news impact with trading signals"""
    print("\n" + "=" * 70)
    print("Example 5: Trading Signal Integration")
    print("=" * 70)

    # Simulate trading scenario
    stocks_to_analyze = ['AKBNK', 'THYAO', 'GARAN', 'EREGL', 'TUPRS']

    print("\nGenerating trading signals based on news analysis...")
    print(f"Analyzing {len(stocks_to_analyze)} stocks")

    analyzer = create_analyzer(provider="openai", temperature=0.2)

    # Sample news (in practice, fetch real news)
    sample_news = {
        'AKBNK': [{'title': 'Güçlü kar', 'content': 'Beklentileri aştı',
                   'source': 'Test', 'published_date': '2024-11-15'}],
        'THYAO': [{'title': 'Yolcu sayısı arttı', 'content': 'Pozitif trend',
                   'source': 'Test', 'published_date': '2024-11-15'}],
        'GARAN': [{'title': 'ROE arttı', 'content': 'Güçlü performans',
                   'source': 'Test', 'published_date': '2024-11-15'}],
        'EREGL': [{'title': 'Üretim düştü', 'content': 'Zayıf talep',
                   'source': 'Test', 'published_date': '2024-11-14'}],
        'TUPRS': [{'title': 'Petrol fiyatları düştü', 'content': 'Marj baskısı',
                   'source': 'Test', 'published_date': '2024-11-14'}],
    }

    results = analyzer.batch_analyze(sample_news)

    # Generate trading signals
    print("\n" + "-" * 70)
    print("Trading Signals (Based on News Sentiment):")
    print("-" * 70)
    print(f"{'Stock':<10} {'Signal':<15} {'Score':<10} {'Confidence':<12} {'Action':<10}")
    print("-" * 70)

    for stock_code, result in sorted(results.items(), key=lambda x: x[1].impact_score, reverse=True):
        # Determine signal
        if result.impact_score > 0.5 and result.confidence > 0.7:
            signal = "STRONG BUY"
            action = "Buy"
        elif result.impact_score > 0.2 and result.confidence > 0.6:
            signal = "BUY"
            action = "Consider Buy"
        elif result.impact_score < -0.5 and result.confidence > 0.7:
            signal = "STRONG SELL"
            action = "Sell"
        elif result.impact_score < -0.2 and result.confidence > 0.6:
            signal = "SELL"
            action = "Consider Sell"
        else:
            signal = "HOLD"
            action = "Hold"

        print(f"{stock_code:<10} {signal:<15} {result.impact_score:+.2f}      {result.confidence:.0%}          {action:<10}")


def print_analysis_result(result: NewsImpactScore):
    """Helper function to print analysis results in a formatted way"""
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nStock Code: {result.stock_code}")
    print(f"Analysis Timestamp: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Articles Analyzed: {result.analyzed_articles}")
    print("\n" + "-" * 70)
    print("IMPACT ASSESSMENT")
    print("-" * 70)
    print(f"Impact Score: {result.impact_score:+.2f} (Range: -1.0 to +1.0)")
    print(f"Impact Category: {result.impact_category.value.upper()}")
    print(f"Confidence Level: {result.confidence:.0%}")
    print(f"Risk Level: {result.risk_level.upper() if result.risk_level else 'N/A'}")

    print("\n" + "-" * 70)
    print("REASONING")
    print("-" * 70)
    print(f"{result.reasoning}")

    if result.key_factors:
        print("\n" + "-" * 70)
        print("KEY FACTORS")
        print("-" * 70)
        for i, factor in enumerate(result.key_factors, 1):
            print(f"{i}. {factor}")

    print("\n" + "-" * 70)
    print("OUTLOOK")
    print("-" * 70)
    print(f"Short-term (1-7 days): {result.short_term_outlook}")
    if result.medium_term_outlook:
        print(f"Medium-term (1-3 months): {result.medium_term_outlook}")

    if result.sector_impact is not None:
        print(f"\nSector Impact: {result.sector_impact:+.2f}")

    if result.macro_factors:
        print("\nMacro Factors:")
        for factor in result.macro_factors:
            print(f"  - {factor}")

    print("\n" + "=" * 70)


def main():
    """Main function to run examples"""
    print("\n" + "=" * 70)
    print("LLM News Analysis Examples for BIST Trading System")
    print("=" * 70)

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it to run these examples:")
        print('  export OPENAI_API_KEY="sk-your-key-here"')
        print("\nRunning with dummy data for demonstration...")
        return

    # Run examples
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Real News Collection", example_2_real_news_collection),
        ("Batch Analysis", example_3_batch_analysis),
        ("Model Comparison", example_4_comparison_analysis),
        ("Trading Signals", example_5_trading_signal_integration),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...")

    for name, example_func in examples:
        try:
            example_func()
            print("\n✓ Example completed successfully")
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
