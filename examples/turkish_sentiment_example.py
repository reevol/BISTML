"""
Example: Turkish Financial Sentiment Analysis

This example demonstrates how to use the Turkish Financial Sentiment Analyzer
for analyzing BIST stock-related news and financial disclosures.

Prerequisites:
- pip install torch transformers scikit-learn
- Internet connection for downloading BERTurk model (first time only)

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nlp.turkish_sentiment import (
    TurkishFinancialSentimentAnalyzer,
    TurkishBERTModel,
    SentimentLabel,
    create_sentiment_analyzer,
    analyze_financial_text
)


def example_basic_usage():
    """Example 1: Basic sentiment analysis"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Sentiment Analysis")
    print("=" * 80)

    # Create analyzer with BERTurk
    analyzer = create_sentiment_analyzer(
        model_name=TurkishBERTModel.BERTURK_CASED
    )

    # Sample Turkish financial news
    texts = [
        "Türk Hava Yolları'nın 3. çeyrek net karı 500 milyon TL olarak açıklandı, beklentileri aştı.",
        "Akbank'ın kredi portföyü daralıyor, yatırımcılar endişeli.",
        "BIST 100 endeksi bugün %0.2 yükselişle kapandı.",
        "Garanti BBVA dijital dönüşüm yatırımlarını artırıyor, büyük fırsat!",
        "Petrol fiyatlarındaki düşüş enerji şirketlerini olumsuz etkiledi.",
    ]

    # Analyze
    predictions = analyzer.predict(texts)

    # Display results
    for text, pred in zip(texts, predictions):
        print(f"\nMetin: {text}")
        print(f"Duygu: {pred.label.name}")
        print(f"Güven: {pred.confidence:.3f}")
        print(f"Olasılıklar:")
        for label, prob in pred.probabilities.items():
            print(f"  {label}: {prob:.3f}")
        if pred.stock_codes:
            print(f"Tespit edilen hisse kodları: {', '.join(pred.stock_codes)}")


def example_batch_processing():
    """Example 2: Batch processing of news articles"""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing of News Articles")
    print("=" * 80)

    # Create analyzer
    analyzer = create_sentiment_analyzer()

    # Sample news articles (simulating data from news collector)
    news_articles = [
        {
            'title': 'THYAO hissesi rekor kırdı',
            'content': 'Türk Hava Yolları hisseleri bugün %5 yükselişle günü kapattı. Şirketin güçlü bilanço açıklaması piyasada olumlu karşılandı.',
            'source': 'Bloomberg HT',
            'published_date': '2025-11-16',
        },
        {
            'title': 'GARAN için satış önerisi',
            'content': 'Analistler Garanti Bankası için satış önerisi yayınladı. Karlılık beklentilerinin düşmesi endişe yarattı.',
            'source': 'Investing.com',
            'published_date': '2025-11-16',
        },
        {
            'title': 'BIST 100 yatay seyretti',
            'content': 'Borsa İstanbul 100 endeksi bugün dar bir bantta hareket etti. İşlem hacmi ortalama seviyede kaldı.',
            'source': 'KAP',
            'published_date': '2025-11-16',
        },
    ]

    # Analyze batch
    results_df = analyzer.analyze_news_batch(news_articles, text_field='content')

    # Display results
    print("\nAnaliz Sonuçları:")
    print(results_df[['title', 'sentiment', 'confidence', 'positive_prob', 'negative_prob']].to_string())


def example_fine_tuning():
    """Example 3: Fine-tuning on custom financial data"""
    print("\n" + "=" * 80)
    print("Example 3: Fine-tuning on Custom Financial Data")
    print("=" * 80)

    # Create analyzer
    analyzer = create_sentiment_analyzer()

    # Sample training data (in practice, you would have much more data)
    train_texts = [
        "Şirketin karı beklentileri aştı, harika bir performans",
        "Hisse senedi fiyatı düşüyor, yatırımcılar kaygılı",
        "Şirket yeni bir ürün lansmanı yapacak",
        "Borsa bugün yatay seyirde, önemli bir hareket yok",
        "CEO istifa etti, yönetimde belirsizlik var",
        "Temettü ödemesi açıklandı, hissedarlar memnun",
        "Karlılık oranı düştü, zor bir çeyrek geçirdik",
        "Şirket büyüme hedeflerini açıkladı",
    ]

    train_labels = [
        0,  # POSITIVE
        1,  # NEGATIVE
        2,  # NEUTRAL
        2,  # NEUTRAL
        1,  # NEGATIVE
        0,  # POSITIVE
        1,  # NEGATIVE
        2,  # NEUTRAL
    ]

    print("\nNot: Fine-tuning gerçek uygulamada çok daha fazla veri gerektirir.")
    print("Bu örnek, API kullanımını göstermek içindir.\n")
    print("Fine-tuning kodu:")
    print("""
    # Fine-tune the model
    history = analyzer.fine_tune(
        train_texts=train_texts,
        train_labels=train_labels,
        output_dir='./models/custom_turkish_sentiment',
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        early_stopping_patience=2
    )

    # Save the fine-tuned model
    analyzer.save_model('./models/my_sentiment_model')

    # Later, load the model
    analyzer.load_model('./models/my_sentiment_model')
    """)


def example_kap_news_integration():
    """Example 4: Integration with KAP news collector"""
    print("\n" + "=" * 80)
    print("Example 4: Integration with KAP News Collector")
    print("=" * 80)

    print("\nBu örnek, KAP haberlerini sentiment analizi ile birleştirmeyi gösterir:\n")
    print("""
from src.data.collectors.news_collector import KAPCollector
from src.models.nlp.turkish_sentiment import create_sentiment_analyzer
from datetime import datetime, timedelta

# Create collectors and analyzer
kap_collector = KAPCollector()
sentiment_analyzer = create_sentiment_analyzer()

# Collect recent news for a stock
stock_code = "THYAO"
articles = kap_collector.collect_news(
    stock_code=stock_code,
    start_date=datetime.now() - timedelta(days=7),
    limit=20
)

# Analyze sentiment for each article
for article in articles:
    if article.content:
        prediction = sentiment_analyzer.predict(article.content)

        print(f"\\nBaşlık: {article.title}")
        print(f"Tarih: {article.published_date}")
        print(f"Duygu: {prediction.label.name} ({prediction.confidence:.3f})")

        # Update article with sentiment
        article.sentiment_score = prediction.confidence
        if prediction.label == SentimentLabel.POSITIVE:
            article.sentiment_score *= 1
        elif prediction.label == SentimentLabel.NEGATIVE:
            article.sentiment_score *= -1
        else:
            article.sentiment_score = 0
    """)


def example_model_comparison():
    """Example 5: Comparing different Turkish BERT models"""
    print("\n" + "=" * 80)
    print("Example 5: Comparing Different Turkish BERT Models")
    print("=" * 80)

    sample_text = "AKBNK hisseleri güçlü bilanço sonrası yükselişe geçti, yatırımcılar mutlu."

    print(f"\nÖrnek metin: {sample_text}\n")
    print("Farklı modellerin sonuçları:")
    print("-" * 80)

    # Test different models
    models_to_test = [
        TurkishBERTModel.BERTURK_CASED,
        # TurkishBERTModel.BERTURK_UNCASED,  # Uncomment to test other models
        # TurkishBERTModel.DISTILBERT_TURKISH,
    ]

    for model_type in models_to_test:
        print(f"\nModel: {model_type.value}")
        try:
            analyzer = create_sentiment_analyzer(model_name=model_type)
            prediction = analyzer.predict(sample_text)
            print(f"Duygu: {prediction.label.name}")
            print(f"Güven: {prediction.confidence:.3f}")
        except Exception as e:
            print(f"Hata: {e}")


def example_evaluation():
    """Example 6: Evaluating model performance"""
    print("\n" + "=" * 80)
    print("Example 6: Model Evaluation")
    print("=" * 80)

    # Create analyzer
    analyzer = create_sentiment_analyzer()

    # Sample test data
    test_texts = [
        "Şirket rekor kar açıkladı, muhteşem!",
        "Hisse değeri çöktü, büyük kayıp",
        "Şirket yeni bir fabrika açacak",
        "Borsa yatay kapandı",
    ]

    test_labels = [0, 1, 2, 2]  # POSITIVE, NEGATIVE, NEUTRAL, NEUTRAL

    # Evaluate
    print("\nTest verileri üzerinde değerlendirme:")
    print("Not: Gerçek değerlendirme için daha fazla test verisi gereklidir.\n")

    # Get predictions
    predictions = analyzer.predict(test_texts)

    print("Tahminler:")
    for text, true_label, pred in zip(test_texts, test_labels, predictions):
        true_sentiment = SentimentLabel(true_label).name
        print(f"\nMetin: {text}")
        print(f"Gerçek: {true_sentiment}, Tahmin: {pred.label.name}")
        print(f"Doğru: {'✓' if pred.label.value == true_label else '✗'}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("TURKISH FINANCIAL SENTIMENT ANALYSIS EXAMPLES")
    print("BIST AI Trading System - BERTurk Sentiment Analyzer")
    print("=" * 80)

    try:
        # Run examples
        example_basic_usage()
        example_batch_processing()
        example_fine_tuning()
        example_kap_news_integration()
        example_model_comparison()
        example_evaluation()

        print("\n" + "=" * 80)
        print("Tüm örnekler başarıyla tamamlandı!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        print("\nGerekli kütüphaneleri yükleyin:")
        print("pip install torch transformers scikit-learn")


if __name__ == "__main__":
    main()
