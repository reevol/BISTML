# News Analyzer Module - Implementation Summary

## Created Files

### 1. `/home/user/BISTML/src/models/nlp/news_analyzer.py` (1,192 lines)
Complete implementation of the Turkish financial news analyzer.

### 2. `/home/user/BISTML/examples/news_analyzer_example.py` (353 lines)
Comprehensive usage examples and integration patterns.

### 3. `/home/user/BISTML/src/models/nlp/README.md`
Documentation for the NLP module.

---

## Core Features Implemented

### 1. Entity Extraction (`TurkishStockEntityExtractor`)
- **Stock Code Detection**: Extracts BIST stock codes (THYAO, GARAN, AKBNK, etc.)
  - Recognizes 80+ BIST 100 stocks
  - Filters Turkish stop words that match stock pattern
  - Confidence scoring based on known stocks
  
- **Company Name Extraction**: Identifies Turkish companies
  - Recognizes A.Ş., Holding, Bank, Sigorta, etc. suffixes
  - Regex patterns for Turkish company naming conventions
  
- **Entity Positioning**: Tracks entity positions in text for context analysis

### 2. Sentiment Analysis (`TurkishSentimentAnalyzer`)
- **Transformer-Based**: Uses Turkish BERT model (`savasy/bert-base-turkish-sentiment-cased`)
  - Automatic model download on first use
  - GPU acceleration support
  - Probability distributions (positive, negative, neutral)
  
- **Financial Keyword Boosting**: 
  - Positive keywords: artış, yükseliş, büyüme, kar, kazanç, başarı, rekor, etc.
  - Negative keywords: düşüş, azalış, zarar, kayıp, risk, kriz, etc.
  - Blends transformer predictions with keyword analysis (70/30 ratio)
  
- **Fallback System**: Keyword-based sentiment when transformers unavailable
  
- **Sentiment Scores**:
  - Numeric score: -1 (very negative) to +1 (very positive)
  - Labels: VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE
  - Confidence metrics for each prediction

### 3. Event Detection (`EventDetector`)
Detects 18+ types of financial events using Turkish pattern matching:

1. **Earnings Report** (Finansal sonuçlar, kar açıklama)
2. **Dividend Announcement** (Temettü dağıtım)
3. **Merger & Acquisition** (Birleşme, devralma, satın alma)
4. **Partnership** (İşbirliği, ortaklık)
5. **Product Launch** (Yeni ürün)
6. **Management Change** (Yönetim değişikliği, CEO atama)
7. **Legal Issue** (Hukuki sorun)
8. **Regulatory Change** (Düzenleme değişikliği)
9. **Credit Rating** (Kredi notu, Moody's, Fitch, S&P)
10. **Share Buyback** (Hisse geri alım)
11. **Capital Increase** (Sermaye artırım, bedelli/bedelsiz)
12. **Bankruptcy** (İflas)
13. **Contract Win** (İhale kazanma, sözleşme)
14. **Guidance Update** (Rehberlik güncellemesi)
15. **Analyst Upgrade** (Analist tavsiye yükseltme)
16. **Analyst Downgrade** (Analist tavsiye düşürme)
17. **Stock Split** (Hisse bölünmesi)
18. **IPO** (Halka arz)

Each event includes:
- Event type classification
- Impact score (0-1, estimated market impact)
- Context description
- Related entities
- Confidence score

### 4. Multi-Article Aggregation (`NewsAnalyzer.aggregate_articles`)

Combines analysis from multiple articles about a stock:

- **Sentiment Aggregation**:
  - Average sentiment score across all articles
  - Sentiment trend detection (improving/declining/stable)
  - Distribution by sentiment labels
  
- **Event Aggregation**:
  - Top events ranked by impact score
  - Event type distribution
  - Deduplication of similar events
  
- **Entity Aggregation**:
  - Most mentioned companies (with frequency)
  - Most mentioned people (with frequency)
  - Organization mentions
  
- **News Momentum Analysis**:
  - Article velocity (articles per day)
  - News spike detection (3x average threshold)
  - Date range coverage
  
- **Keyword Extraction**:
  - Top 15 keywords across all articles
  - Frequency-based ranking
  - Turkish stop word filtering
  
- **Impact Assessment**:
  - Overall news impact score (0-1)
  - Trading recommendation (bullish/bearish/neutral)
  - Relevance scoring for trading decisions

---

## Data Classes & Types

### ArticleAnalysis
Complete analysis result for a single article containing:
- Extracted entities (stock codes, companies, people, organizations)
- Sentiment scores and probabilities
- Detected events with impact scores
- Keywords and categories
- Relevance score for trading

### AggregatedAnalysis
Multi-article summary containing:
- Sentiment statistics and trends
- Key events and distributions
- Entity mentions with frequencies
- News momentum metrics
- Overall impact score and recommendation

### SentimentScore
Detailed sentiment information:
- Numeric score (-1 to +1)
- Categorical label
- Confidence score
- Probability distribution

### KeyEvent
Financial event information:
- Event type (enum)
- Description with context
- Related entities
- Impact score
- Confidence level

---

## Integration Points

### 1. With News Collector
```python
from src.data.collectors.news_collector import NewsCollectorManager
from src.models.nlp.news_analyzer import NewsAnalyzer

manager = NewsCollectorManager()
analyzer = NewsAnalyzer()

# Collect news
articles = manager.collect_for_stock("THYAO", days_back=7)

# Analyze each article
for article in articles:
    analysis = analyzer.analyze_article(
        title=article.title,
        content=article.content or "",
        url=article.url,
        published_date=article.published_date,
        source=article.source
    )
    article.sentiment_score = analysis.sentiment.score

# Aggregate for stock
summary = analyzer.aggregate_articles(analyzed_articles, "THYAO")
```

### 2. With Trading Signals
```python
# Use aggregated news analysis in signal generation
if (aggregated.recommendation == "bullish" and
    aggregated.news_impact_score > 0.7 and
    aggregated.avg_sentiment > 0.3):
    signal = "BUY"
    confidence = aggregated.news_impact_score
```

### 3. With Feature Engineering
```python
# Generate news-based features for ML models
features = {
    'news_sentiment_1d': aggregated.avg_sentiment,
    'news_velocity': aggregated.article_velocity,
    'news_impact': aggregated.news_impact_score,
    'news_trend_improving': 1 if aggregated.sentiment_trend == "improving" else 0,
    'high_impact_events': sum(1 for e in aggregated.key_events if e.impact_score > 0.7)
}
```

---

## Technical Implementation Details

### Turkish Language Processing
- **Character Support**: Full Turkish alphabet (ç, ğ, ı, ö, ş, ü)
- **Company Patterns**: Turkish naming conventions (A.Ş., Ltd. Şti., Holding)
- **Financial Terminology**: Turkish financial keywords and phrases
- **Stop Words**: Turkish-specific stop word filtering

### Model Architecture
- **Primary**: Turkish BERT for sentiment (`savasy/bert-base-turkish-sentiment-cased`)
- **Optional**: spaCy Turkish model for NER (`tr_core_news_lg`)
- **Fallback**: Keyword-based analysis when models unavailable

### Performance Optimizations
- Lazy model loading (only when needed)
- GPU acceleration support (CUDA)
- Efficient regex compilation
- Event deduplication
- Text truncation for long articles (512 tokens max)

### Error Handling
- Graceful degradation when dependencies unavailable
- Fallback to keyword-based sentiment
- Missing data handling (empty content, missing dates)
- Try-except blocks around external library calls

---

## Usage Examples

See `/home/user/BISTML/examples/news_analyzer_example.py` for:

1. **Single Article Analysis**: Complete entity, sentiment, and event extraction
2. **Multi-Article Aggregation**: Combine insights from 5+ articles
3. **News Collector Integration**: Full pipeline from collection to analysis
4. **Sentiment Comparison**: Test sentiment detection across positive/negative texts
5. **DataFrame Export**: Convert results to pandas for further analysis

---

## Dependencies

### Required
- numpy
- pandas

### NLP (Optional but Recommended)
- transformers (Turkish BERT)
- torch (PyTorch backend)
- spacy (NER)
- nltk (tokenization)

### Installation
```bash
pip install numpy pandas transformers torch spacy nltk
python -m spacy download tr_core_news_lg
```

---

## Testing & Validation

The module includes:
- Example Turkish financial news for testing
- Comprehensive docstrings with parameter descriptions
- Type hints throughout
- Validation of edge cases (empty text, missing data)
- Confidence scoring for quality assessment

---

## Future Enhancements

Potential improvements identified:
1. Fine-tune Turkish BERT on financial news corpus
2. Real-time streaming support
3. Multi-language support (English financial news)
4. Named entity linking (map codes to company names)
5. Causal relationship extraction
6. Article summarization
7. Source reliability scoring
8. Social media sentiment integration

---

## File Structure

```
src/models/nlp/
├── __init__.py
├── news_analyzer.py          # Main implementation (1,192 lines)
└── README.md                  # Documentation

examples/
└── news_analyzer_example.py   # Usage examples (353 lines)
```

---

## Quick Reference

### Analyze Single Article
```python
from src.models.nlp.news_analyzer import analyze_news_article

analysis = analyze_news_article(
    title="THYAO Karı Arttı",
    content="Türk Hava Yolları..."
)
```

### Aggregate Multiple Articles
```python
from src.models.nlp.news_analyzer import aggregate_news_for_stock

summary = aggregate_news_for_stock(analyzed_articles, "THYAO")
```

### Custom Sentiment Model
```python
analyzer = NewsAnalyzer(
    sentiment_model="custom-model-name",
    use_spacy=True
)
```

---

## Summary

The news analyzer module provides enterprise-grade NLP capabilities specifically designed for Turkish financial news analysis in the BIST trading system. It combines modern transformer-based models with domain-specific financial knowledge to extract actionable insights from news articles, enabling data-driven trading decisions based on sentiment, events, and news momentum.

**Key Strengths:**
- Turkish language optimization
- Financial domain expertise
- Comprehensive event detection
- Multi-article intelligence
- Production-ready error handling
- Flexible integration points

**Lines of Code:** 1,192 (main module) + 353 (examples) = 1,545 total
