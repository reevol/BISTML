"""
Example usage of the News Analyzer module

This script demonstrates how to use the news analyzer to extract entities,
calculate sentiment scores, identify key events, and aggregate multiple
articles for BIST stocks.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nlp.news_analyzer import (
    NewsAnalyzer,
    ArticleAnalysis,
    analyze_news_article,
    aggregate_news_for_stock
)
from src.data.collectors.news_collector import (
    NewsCollectorManager,
    NewsArticle
)


def example_1_analyze_single_article():
    """Example 1: Analyze a single Turkish news article"""
    print("=" * 80)
    print("Example 1: Analyzing a Single News Article")
    print("=" * 80)

    # Sample Turkish financial news
    title = "THYAO Hisseleri Yükselişte: Üçüncü Çeyrek Karı Beklentileri Aştı"
    content = """
    Türk Hava Yolları (THYAO) bugün yayınladığı finansal tablolarda üçüncü çeyrek
    net karının beklentilerin üzerinde gerçekleştiğini açıkladı. Şirket, 2.5 milyar
    TL net kar açıklarken, bu rakam analistlerin beklentilerinin %15 üzerinde kaldı.

    Genel Müdür Ahmet Yılmaz, sonuçları değerlendirirken "Yolcu sayısında rekor
    artış yaşadık ve gelecek dönem için olumlu beklentilerimiz var" dedi.

    GARAN ve AKBNK analistleri, THYAO için hedef fiyatı yükselterek hisse için
    'AL' tavsiyesi verdi. Şirketin hisse senedi bugün %3.5 artışla 285 TL'den
    işlem gördü.
    """

    # Analyze the article
    analysis = analyze_news_article(
        title=title,
        content=content,
        url="https://example.com/thyao-news",
        source="Example News",
        published_date=datetime.now()
    )

    # Display results
    print(f"\nTitle: {analysis.title}\n")

    print("EXTRACTED ENTITIES:")
    print("-" * 40)
    print(f"Stock Codes: {', '.join(analysis.stock_codes)}")
    print(f"Companies: {', '.join([e.text for e in analysis.companies])}")
    if analysis.people:
        print(f"People: {', '.join([e.text for e in analysis.people])}")
    if analysis.organizations:
        print(f"Organizations: {', '.join([e.text for e in analysis.organizations])}")

    print("\nSENTIMENT ANALYSIS:")
    print("-" * 40)
    if analysis.sentiment:
        print(f"Label: {analysis.sentiment.label.value}")
        print(f"Score: {analysis.sentiment.score:.3f} (range: -1 to +1)")
        print(f"Confidence: {analysis.sentiment.confidence:.3f}")
        print(f"Positive Probability: {analysis.sentiment.positive_prob:.3f}")
        print(f"Negative Probability: {analysis.sentiment.negative_prob:.3f}")
        print(f"Neutral Probability: {analysis.sentiment.neutral_prob:.3f}")

    print("\nDETECTED EVENTS:")
    print("-" * 40)
    if analysis.events:
        for i, event in enumerate(analysis.events, 1):
            print(f"{i}. {event.event_type.value.replace('_', ' ').title()}")
            print(f"   Impact Score: {event.impact_score:.2f}")
            print(f"   Confidence: {event.confidence:.2f}")
            print(f"   Description: {event.description[:100]}...")
    else:
        print("No specific events detected")

    print("\nKEYWORDS:")
    print("-" * 40)
    print(f"{', '.join(analysis.keywords[:15])}")

    print(f"\nRelevance Score: {analysis.relevance_score:.3f}")


def example_2_analyze_multiple_articles():
    """Example 2: Analyze multiple articles and aggregate"""
    print("\n" + "=" * 80)
    print("Example 2: Analyzing Multiple Articles for THYAO")
    print("=" * 80)

    # Sample articles about THYAO
    articles_data = [
        {
            "title": "THYAO Üçüncü Çeyrek Sonuçları Açıklandı",
            "content": "Türk Hava Yolları rekor kar açıkladı. Net kar 2.5 milyar TL olarak gerçekleşti.",
            "date": datetime.now() - timedelta(days=1)
        },
        {
            "title": "THY Yeni Uçak Siparişi Verdi",
            "content": "THYAO, filosunu genişletmek için 50 yeni uçak siparişi verdiğini duyurdu.",
            "date": datetime.now() - timedelta(days=2)
        },
        {
            "title": "Analistler THYAO için Hedef Fiyatı Yükseltti",
            "content": "Garanti ve Akbank analistleri THYAO için AL tavsiyesi verdi. Hedef fiyat 320 TL.",
            "date": datetime.now() - timedelta(days=3)
        },
        {
            "title": "THYAO Yolcu Sayısında Artış",
            "content": "Türk Hava Yolları yolcu sayısında %25 artış kaydetti. Avrupa hatlarında talep güçlü.",
            "date": datetime.now() - timedelta(days=4)
        },
        {
            "title": "THY Yeni Ortaklık Anlaşması İmzaladı",
            "content": "THYAO, Orta Doğu'da yeni bir havayolu ile stratejik ortaklık anlaşması imzaladı.",
            "date": datetime.now() - timedelta(days=5)
        }
    ]

    # Analyze all articles
    analyzer = NewsAnalyzer()
    analyzed_articles = []

    print("\nAnalyzing articles...")
    for i, article_data in enumerate(articles_data, 1):
        analysis = analyzer.analyze_article(
            title=article_data["title"],
            content=article_data["content"],
            published_date=article_data["date"],
            source="Example News",
            article_id=f"article_{i}"
        )
        analyzed_articles.append(analysis)
        print(f"  {i}. {article_data['title'][:60]}... "
              f"[Sentiment: {analysis.sentiment.score:.2f}]")

    # Aggregate articles for THYAO
    print("\n" + "=" * 80)
    print("AGGREGATED ANALYSIS FOR THYAO")
    print("=" * 80)

    aggregated = analyzer.aggregate_articles(analyzed_articles, "THYAO")

    print(f"\nNumber of Articles: {aggregated.num_articles}")
    print(f"Date Range: {aggregated.date_range[0].strftime('%Y-%m-%d')} to "
          f"{aggregated.date_range[1].strftime('%Y-%m-%d')}")

    print("\nSENTIMENT SUMMARY:")
    print("-" * 40)
    print(f"Average Sentiment: {aggregated.avg_sentiment:.3f}")
    print(f"Sentiment Trend: {aggregated.sentiment_trend}")
    print(f"Distribution: {aggregated.sentiment_distribution}")

    print("\nKEY EVENTS:")
    print("-" * 40)
    if aggregated.key_events:
        for i, event in enumerate(aggregated.key_events[:5], 1):
            print(f"{i}. {event.event_type.value.replace('_', ' ').title()} "
                  f"(impact: {event.impact_score:.2f})")
    else:
        print("No key events detected")

    print(f"\nEvent Distribution: {aggregated.event_distribution}")

    print("\nNEWS MOMENTUM:")
    print("-" * 40)
    print(f"Article Velocity: {aggregated.article_velocity:.2f} articles/day")
    print(f"Recent Spike: {'Yes' if aggregated.recent_spike else 'No'}")

    print("\nTOP KEYWORDS:")
    print("-" * 40)
    for keyword, count in aggregated.top_keywords[:10]:
        print(f"  {keyword}: {count}")

    print("\nOVERALL ASSESSMENT:")
    print("-" * 40)
    print(f"News Impact Score: {aggregated.news_impact_score:.3f}")
    print(f"Recommendation: {aggregated.recommendation.upper()}")


def example_3_integrate_with_news_collector():
    """Example 3: Integration with News Collector"""
    print("\n" + "=" * 80)
    print("Example 3: Integration with News Collector")
    print("=" * 80)

    print("\nThis example shows how to integrate the news analyzer with the news collector")
    print("to create a complete news analysis pipeline.\n")

    # Sample workflow code
    workflow_code = '''
    # Step 1: Collect news articles
    from src.data.collectors.news_collector import NewsCollectorManager
    from src.models.nlp.news_analyzer import NewsAnalyzer

    # Initialize collectors and analyzer
    news_manager = NewsCollectorManager()
    analyzer = NewsAnalyzer()

    # Collect news for THYAO
    raw_articles = news_manager.collect_for_stock("THYAO", days_back=7)

    # Step 2: Analyze each article
    analyzed_articles = []
    for article in raw_articles:
        analysis = analyzer.analyze_article(
            title=article.title,
            content=article.content or "",
            url=article.url,
            published_date=article.published_date,
            source=article.source
        )
        analyzed_articles.append(analysis)

        # Update the original article with sentiment score
        article.sentiment_score = analysis.sentiment.score if analysis.sentiment else 0.0

    # Step 3: Aggregate analysis
    aggregated = analyzer.aggregate_articles(analyzed_articles, "THYAO")

    # Step 4: Use in trading signals
    if aggregated.recommendation == "bullish" and aggregated.news_impact_score > 0.7:
        print(f"STRONG BUY signal for THYAO")
        print(f"  Sentiment: {aggregated.avg_sentiment:.2f}")
        print(f"  Impact: {aggregated.news_impact_score:.2f}")
        print(f"  Articles: {aggregated.num_articles}")
    '''

    print(workflow_code)


def example_4_sentiment_analysis_details():
    """Example 4: Detailed sentiment analysis"""
    print("\n" + "=" * 80)
    print("Example 4: Detailed Sentiment Analysis Comparison")
    print("=" * 80)

    test_texts = [
        ("VERY POSITIVE", "THYAO rekor kar açıkladı! Hisseler tarihi zirvede!"),
        ("POSITIVE", "AKBNK için olumlu görüş. Analistler hedef fiyatı yükseltti."),
        ("NEUTRAL", "GARAN bugün yatay seyretti. Hacim düşük kaldı."),
        ("NEGATIVE", "TUPRS üretimde sorun yaşıyor. Karlar düşebilir."),
        ("VERY NEGATIVE", "PETKM için büyük zarar bekleniyor! Kritik durum!")
    ]

    analyzer = NewsAnalyzer()

    print("\nAnalyzing sentiment for various text samples:\n")

    for expected, text in test_texts:
        analysis = analyzer.analyze_article(
            title=text,
            content="",
            source="test"
        )

        if analysis.sentiment:
            print(f"Text: {text}")
            print(f"  Expected: {expected}")
            print(f"  Detected: {analysis.sentiment.label.value.upper().replace('_', ' ')}")
            print(f"  Score: {analysis.sentiment.score:.3f}")
            print(f"  Confidence: {analysis.sentiment.confidence:.3f}")
            print()


def example_5_export_to_dataframe():
    """Example 5: Export analysis results to DataFrame"""
    print("\n" + "=" * 80)
    print("Example 5: Exporting Analysis to Pandas DataFrame")
    print("=" * 80)

    import pandas as pd

    # Analyze multiple articles
    articles_data = [
        ("THYAO Karı Arttı", "Türk Hava Yolları net karını %20 artırdı."),
        ("GARAN Temettü Dağıtacak", "Garanti Bankası yüksek temettü açıkladı."),
        ("AKBNK Hedef Fiyat", "Akbank için analist hedefi yükseldi."),
    ]

    analyzer = NewsAnalyzer()
    results = []

    for title, content in articles_data:
        analysis = analyzer.analyze_article(title=title, content=content)

        results.append({
            'title': analysis.title,
            'stock_codes': ', '.join(analysis.stock_codes),
            'sentiment_score': analysis.sentiment.score if analysis.sentiment else None,
            'sentiment_label': analysis.sentiment.label.value if analysis.sentiment else None,
            'num_events': len(analysis.events),
            'relevance_score': analysis.relevance_score,
            'keywords': ', '.join(analysis.keywords[:5])
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    print("\nAnalysis Results as DataFrame:")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to CSV (optional)
    # df.to_csv('news_analysis_results.csv', index=False)
    print("\n(Can be saved to CSV for further processing)")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "NEWS ANALYZER EXAMPLES" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        example_1_analyze_single_article()
        example_2_analyze_multiple_articles()
        example_3_integrate_with_news_collector()
        example_4_sentiment_analysis_details()
        example_5_export_to_dataframe()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

        print("\nNOTE: For production use:")
        print("  1. Install required dependencies: pip install -r requirements.txt")
        print("  2. Download spaCy Turkish model: python -m spacy download tr_core_news_lg")
        print("  3. The sentiment analyzer will download Turkish BERT model on first use")
        print("  4. Consider fine-tuning the model on Turkish financial news for better accuracy")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
