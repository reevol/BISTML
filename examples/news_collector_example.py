"""
Example usage of the News Collector module

This script demonstrates how to use the news collector to gather
Turkish financial news and KAP disclosures for BIST stocks.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.news_collector import (
    NewsCollectorManager,
    KAPCollector,
    BloombergHTCollector,
    NewsSource
)


def example_1_collect_for_specific_stock():
    """Example 1: Collect news for a specific stock"""
    print("=" * 80)
    print("Example 1: Collecting news for THYAO (Turkish Airlines)")
    print("=" * 80)

    manager = NewsCollectorManager()

    # Collect news for THYAO from the last 7 days
    articles = manager.collect_for_stock(
        stock_code="THYAO",
        days_back=7,
        limit_per_source=10
    )

    print(f"\nCollected {len(articles)} articles for THYAO\n")

    # Display first 3 articles
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. [{article.source}] {article.title}")
        print(f"   Published: {article.published_date}")
        print(f"   URL: {article.url}")
        print(f"   Stock codes mentioned: {', '.join(article.stock_codes)}")
        if article.content:
            print(f"   Preview: {article.content[:150]}...")


def example_2_collect_kap_disclosures():
    """Example 2: Collect only KAP disclosures"""
    print("\n" + "=" * 80)
    print("Example 2: Collecting KAP disclosures for multiple banks")
    print("=" * 80)

    kap_collector = KAPCollector()

    # Bank stock codes
    banks = ["AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN"]

    for bank in banks:
        articles = kap_collector.collect_news(
            stock_code=bank,
            start_date=datetime.now() - timedelta(days=30),
            limit=5
        )

        print(f"\n{bank}: {len(articles)} disclosures found")
        for article in articles[:2]:
            print(f"  - {article.title}")
            print(f"    Date: {article.published_date}")


def example_3_latest_market_news():
    """Example 3: Get latest general market news"""
    print("\n" + "=" * 80)
    print("Example 3: Getting latest market news from all sources")
    print("=" * 80)

    manager = NewsCollectorManager()

    # Get 20 latest news articles
    latest_news = manager.get_latest_news(limit=20)

    print(f"\nCollected {len(latest_news)} latest news articles\n")

    # Group by source
    by_source = {}
    for article in latest_news:
        if article.source not in by_source:
            by_source[article.source] = []
        by_source[article.source].append(article)

    for source, articles in by_source.items():
        print(f"\n{source}: {len(articles)} articles")
        for article in articles[:2]:
            print(f"  - {article.title}")


def example_4_collect_from_specific_sources():
    """Example 4: Collect from specific sources only"""
    print("\n" + "=" * 80)
    print("Example 4: Collecting only from KAP and Bloomberg HT")
    print("=" * 80)

    manager = NewsCollectorManager()

    # Collect only from KAP and Bloomberg HT
    articles = manager.collect_from_all_sources(
        start_date=datetime.now() - timedelta(days=3),
        limit_per_source=15,
        sources=[NewsSource.KAP, NewsSource.BLOOMBERG_HT]
    )

    print(f"\nCollected {len(articles)} articles from selected sources")

    # Show distribution
    kap_count = sum(1 for a in articles if a.source == NewsSource.KAP.value)
    bloomberg_count = sum(1 for a in articles if a.source == NewsSource.BLOOMBERG_HT.value)

    print(f"  - KAP: {kap_count} articles")
    print(f"  - Bloomberg HT: {bloomberg_count} articles")


def example_5_export_to_dict():
    """Example 5: Export articles to dictionary format"""
    print("\n" + "=" * 80)
    print("Example 5: Exporting articles to dictionary format")
    print("=" * 80)

    manager = NewsCollectorManager()

    articles = manager.collect_for_stock(
        stock_code="PETKM",
        days_back=7,
        limit_per_source=5
    )

    print(f"\nCollected {len(articles)} articles for PETKM")
    print("\nFirst article as dictionary:")

    if articles:
        article_dict = articles[0].to_dict()
        for key, value in article_dict.items():
            if key == 'content' and value:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


def example_6_filter_by_stock_codes():
    """Example 6: Find all articles mentioning specific stocks"""
    print("\n" + "=" * 80)
    print("Example 6: Finding articles mentioning technology stocks")
    print("=" * 80)

    manager = NewsCollectorManager()

    # Tech stocks in BIST
    tech_stocks = ["ASELS", "LOGO", "NETAS", "INDES", "KAREL"]

    # Collect general news
    all_articles = manager.get_latest_news(limit=50)

    # Filter for tech stocks
    tech_articles = []
    for article in all_articles:
        if any(stock in article.stock_codes for stock in tech_stocks):
            tech_articles.append(article)

    print(f"\nFound {len(tech_articles)} articles mentioning tech stocks:")
    for article in tech_articles[:5]:
        mentioned = [s for s in tech_stocks if s in article.stock_codes]
        print(f"\n  - {article.title}")
        print(f"    Stocks: {', '.join(mentioned)}")
        print(f"    Source: {article.source}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "BIST NEWS COLLECTOR EXAMPLES" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        example_1_collect_for_specific_stock()
        example_2_collect_kap_disclosures()
        example_3_latest_market_news()
        example_4_collect_from_specific_sources()
        example_5_export_to_dict()
        example_6_filter_by_stock_codes()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
