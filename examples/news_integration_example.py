"""
Integration Example: News Collector with Trading System

This example demonstrates how to integrate the news collector with
the broader BIST AI trading system for sentiment-based signal generation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.news_collector import (
    NewsCollectorManager,
    NewsSource,
    NewsArticle
)


class NewsDataPipeline:
    """
    Pipeline to integrate news collection into the trading system
    """

    def __init__(self):
        """Initialize the pipeline"""
        self.collector_manager = NewsCollectorManager()

    def collect_and_store_news(
        self,
        stock_codes: list,
        days_back: int = 7,
        limit_per_source: int = 20
    ):
        """
        Collect news for multiple stocks and prepare for storage

        Args:
            stock_codes: List of BIST stock codes
            days_back: Number of days to look back
            limit_per_source: Max articles per source per stock

        Returns:
            DataFrame with collected news
        """
        all_articles = []

        print(f"Collecting news for {len(stock_codes)} stocks...")

        for stock in stock_codes:
            print(f"  Processing {stock}...")

            articles = self.collector_manager.collect_for_stock(
                stock_code=stock,
                days_back=days_back,
                limit_per_source=limit_per_source
            )

            all_articles.extend(articles)

        # Convert to DataFrame
        df = self._articles_to_dataframe(all_articles)

        print(f"\nCollected {len(df)} articles total")
        return df

    def _articles_to_dataframe(self, articles: list) -> pd.DataFrame:
        """Convert articles to DataFrame"""
        data = [article.to_dict() for article in articles]
        df = pd.DataFrame(data)

        if not df.empty:
            # Convert date strings to datetime
            df['published_date'] = pd.to_datetime(df['published_date'])

            # Explode stock_codes (one row per stock-article pair)
            df = df.explode('stock_codes')

            # Add additional columns for analysis
            df['collected_at'] = datetime.now()
            df['has_content'] = df['content'].notna()
            df['content_length'] = df['content'].str.len()

        return df

    def prepare_for_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare news data for NLP sentiment analysis

        Args:
            df: DataFrame with news articles

        Returns:
            DataFrame ready for sentiment analysis
        """
        # Filter articles with content
        df_with_content = df[df['has_content']].copy()

        # Combine title and content for analysis
        df_with_content['full_text'] = (
            df_with_content['title'] + ' ' + df_with_content['content']
        )

        # Remove duplicates
        df_with_content = df_with_content.drop_duplicates(
            subset=['source', 'title']
        )

        return df_with_content[['stock_codes', 'source', 'published_date',
                               'title', 'full_text', 'url']]

    def aggregate_news_by_stock(
        self,
        df: pd.DataFrame,
        time_window: str = '1D'
    ) -> pd.DataFrame:
        """
        Aggregate news metrics by stock and time window

        Args:
            df: DataFrame with news articles
            time_window: Pandas time window (e.g., '1D', '1H')

        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df

        # Set index for resampling
        df = df.set_index('published_date')

        # Group by stock and time window
        aggregated = df.groupby([
            'stock_codes',
            pd.Grouper(freq=time_window)
        ]).agg({
            'title': 'count',  # Number of articles
            'source': lambda x: x.nunique(),  # Number of unique sources
            'sentiment_score': 'mean',  # Average sentiment (if available)
        }).rename(columns={
            'title': 'article_count',
            'source': 'source_count',
            'sentiment_score': 'avg_sentiment'
        })

        return aggregated.reset_index()


class NewsBasedSignalGenerator:
    """
    Generate trading signals based on news analysis
    (Placeholder for integration with src/signals/)
    """

    def __init__(self):
        """Initialize signal generator"""
        self.pipeline = NewsDataPipeline()

    def generate_news_impact_score(
        self,
        stock_code: str,
        hours_back: int = 24
    ) -> dict:
        """
        Generate a news impact score for a stock

        Args:
            stock_code: BIST stock code
            hours_back: Hours to look back

        Returns:
            Dictionary with impact metrics
        """
        # Collect recent news
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)

        manager = NewsCollectorManager()
        articles = manager.collect_from_all_sources(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            limit_per_source=50
        )

        if not articles:
            return {
                'stock_code': stock_code,
                'news_count': 0,
                'impact_score': 0.0,
                'sources': [],
                'latest_article': None
            }

        # Calculate impact metrics
        kap_count = sum(1 for a in articles if a.source == NewsSource.KAP.value)
        news_count = len(articles)
        sources = list(set(a.source for a in articles))

        # Simple impact score (can be enhanced with sentiment)
        # Higher score for KAP disclosures and multiple sources
        impact_score = (
            (news_count * 0.3) +
            (kap_count * 2.0) +  # KAP disclosures have higher weight
            (len(sources) * 0.5)
        )

        return {
            'stock_code': stock_code,
            'news_count': news_count,
            'kap_count': kap_count,
            'impact_score': min(impact_score, 10.0),  # Cap at 10
            'sources': sources,
            'latest_article': articles[0] if articles else None,
            'time_window': f'{hours_back}h',
        }

    def batch_generate_impact_scores(
        self,
        stock_codes: list,
        hours_back: int = 24
    ) -> pd.DataFrame:
        """
        Generate impact scores for multiple stocks

        Args:
            stock_codes: List of stock codes
            hours_back: Hours to look back

        Returns:
            DataFrame with impact scores
        """
        scores = []

        for stock in stock_codes:
            print(f"Analyzing news impact for {stock}...")
            score = self.generate_news_impact_score(stock, hours_back)
            scores.append(score)

        df = pd.DataFrame(scores)

        # Sort by impact score
        df = df.sort_values('impact_score', ascending=False)

        return df


def example_1_basic_integration():
    """Example 1: Basic news collection and storage"""
    print("=" * 80)
    print("Example 1: Basic News Collection and Storage")
    print("=" * 80)

    pipeline = NewsDataPipeline()

    # Collect news for major BIST stocks
    stocks = ["THYAO", "AKBNK", "GARAN", "TUPRS", "PETKM"]

    df = pipeline.collect_and_store_news(
        stock_codes=stocks,
        days_back=7,
        limit_per_source=10
    )

    print("\nDataFrame Info:")
    print(df.info())

    print("\nSample Articles:")
    print(df[['stock_codes', 'source', 'title', 'published_date']].head())


def example_2_sentiment_preparation():
    """Example 2: Prepare data for sentiment analysis"""
    print("\n" + "=" * 80)
    print("Example 2: Prepare for Sentiment Analysis")
    print("=" * 80)

    pipeline = NewsDataPipeline()

    # Collect news
    df = pipeline.collect_and_store_news(
        stock_codes=["THYAO"],
        days_back=3,
        limit_per_source=20
    )

    # Prepare for sentiment analysis
    sentiment_ready = pipeline.prepare_for_sentiment_analysis(df)

    print(f"\nPrepared {len(sentiment_ready)} articles for sentiment analysis")
    print("\nSample:")
    print(sentiment_ready[['stock_codes', 'source', 'title']].head())

    # This would be passed to the NLP module
    # from src.models.nlp.turkish_sentiment import analyze_sentiment
    # sentiment_ready['sentiment_score'] = sentiment_ready['full_text'].apply(analyze_sentiment)


def example_3_news_aggregation():
    """Example 3: Aggregate news by stock and time"""
    print("\n" + "=" * 80)
    print("Example 3: News Aggregation by Stock and Time")
    print("=" * 80)

    pipeline = NewsDataPipeline()

    # Collect news for multiple stocks
    stocks = ["THYAO", "AKBNK", "GARAN"]
    df = pipeline.collect_and_store_news(
        stock_codes=stocks,
        days_back=7
    )

    # Aggregate by day
    daily_stats = pipeline.aggregate_news_by_stock(df, time_window='1D')

    print("\nDaily News Statistics:")
    print(daily_stats)


def example_4_news_impact_signals():
    """Example 4: Generate news-based signals"""
    print("\n" + "=" * 80)
    print("Example 4: News Impact Signal Generation")
    print("=" * 80)

    signal_gen = NewsBasedSignalGenerator()

    # Generate impact scores for BIST 30 stocks (sample)
    bist30_sample = [
        "THYAO", "AKBNK", "GARAN", "TUPRS", "PETKM",
        "EREGL", "SAHOL", "KOZAL", "KCHOL", "ISCTR"
    ]

    impact_df = signal_gen.batch_generate_impact_scores(
        stock_codes=bist30_sample,
        hours_back=24
    )

    print("\nNews Impact Scores (Last 24 Hours):")
    print(impact_df[['stock_code', 'news_count', 'kap_count', 'impact_score']])

    print("\nTop 3 Stocks by News Impact:")
    top_3 = impact_df.head(3)
    for _, row in top_3.iterrows():
        print(f"\n{row['stock_code']}: Impact Score = {row['impact_score']:.2f}")
        print(f"  News Count: {row['news_count']}")
        print(f"  KAP Disclosures: {row['kap_count']}")
        print(f"  Sources: {', '.join(row['sources'])}")


def example_5_integration_with_signals():
    """Example 5: Integration with signal generation module"""
    print("\n" + "=" * 80)
    print("Example 5: Integration with Trading Signals")
    print("=" * 80)

    print("""
    This example shows how news data integrates with the signal generation module.

    Integration Flow:
    1. News Collector gathers articles for BIST stocks
    2. NLP Module analyzes sentiment (Turkish financial NLP)
    3. News features are generated:
       - Article count (last 24h, 7d, 30d)
       - Average sentiment score
       - KAP disclosure count
       - News impact score
    4. These features are combined with:
       - Technical indicators (from src/features/technical/)
       - Fundamental metrics (from src/features/fundamental/)
       - Whale activity (from src/features/whale/)
    5. ML models use combined features for signal generation
    6. Final signals prioritize based on news impact

    Example Signal Output:
    ┌──────────┬──────────┬────────────┬──────────────┬──────────────┐
    │ Stock    │ Signal   │ ML Price   │ Confidence   │ News Impact  │
    ├──────────┼──────────┼────────────┼──────────────┼──────────────┤
    │ THYAO    │ BUY      │ 285.50     │ 87%          │ 8.5/10       │
    │ AKBNK    │ HOLD     │ 55.20      │ 65%          │ 4.2/10       │
    │ GARAN    │ STRONG   │ 145.00     │ 92%          │ 9.1/10       │
    │          │ BUY      │            │              │              │
    └──────────┴──────────┴────────────┴──────────────┴──────────────┘

    Code example (pseudo-code for future implementation):

    from src.data.collectors.news_collector import NewsCollectorManager
    from src.models.nlp.turkish_sentiment import TurkishSentimentAnalyzer
    from src.signals.generator import SignalGenerator

    # Collect news
    news_manager = NewsCollectorManager()
    articles = news_manager.collect_for_stock("THYAO", days_back=7)

    # Analyze sentiment
    sentiment_analyzer = TurkishSentimentAnalyzer()
    for article in articles:
        article.sentiment_score = sentiment_analyzer.analyze(article.content)

    # Generate news features
    news_features = {
        'article_count_24h': len([a for a in articles if ...]),
        'avg_sentiment': np.mean([a.sentiment_score for a in articles]),
        'kap_count': len([a for a in articles if a.source == 'kap']),
        'news_impact': calculate_impact_score(articles)
    }

    # Combine with other features and generate signal
    signal_gen = SignalGenerator()
    signal = signal_gen.generate(
        stock_code="THYAO",
        technical_features={...},
        fundamental_features={...},
        whale_features={...},
        news_features=news_features
    )
    """)


def main():
    """Run all integration examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "NEWS COLLECTOR INTEGRATION EXAMPLES" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        example_1_basic_integration()
        example_2_sentiment_preparation()
        example_3_news_aggregation()
        example_4_news_impact_signals()
        example_5_integration_with_signals()

        print("\n" + "=" * 80)
        print("All integration examples completed!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
