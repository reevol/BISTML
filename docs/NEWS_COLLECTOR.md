# News Collector Module Documentation

## Overview

The News Collector module is designed to collect Turkish financial news from major sources and KAP (Kamuyu Aydınlatma Platformu - Public Disclosure Platform) for BIST (Borsa Istanbul) stocks.

## Features

- **Multi-Source Collection**: Gather news from multiple Turkish financial news sources
- **KAP Integration**: Collect official public disclosures from KAP
- **Web Scraping**: Built with BeautifulSoup and optional Scrapy support
- **Stock Code Extraction**: Automatically extract BIST stock codes from articles
- **Rate Limiting**: Respectful scraping with built-in rate limiting
- **Caching**: Optional caching to reduce redundant requests
- **Error Handling**: Robust error handling and retry logic
- **Flexible Filtering**: Filter by stock code, date range, and source

## Supported News Sources

1. **KAP (Public Disclosure Platform)**: https://www.kap.org.tr
   - Official regulatory disclosures
   - Material events
   - Financial statements
   - Special situations

2. **Bloomberg HT**: https://www.bloomberght.com
   - Turkish edition of Bloomberg
   - Stock market news
   - Company news

3. **Investing.com Turkey**: https://tr.investing.com
   - International financial news
   - Turkish market coverage

4. **Extensible Architecture**: Easy to add more sources

## Installation

### Required Dependencies

```bash
pip install requests beautifulsoup4 lxml html5lib python-dateutil
```

### Optional Dependencies

```bash
# For advanced scraping
pip install scrapy

# For caching
pip install redis

# For database storage
pip install sqlalchemy
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Quick Start

### Example 1: Collect News for a Specific Stock

```python
from src.data.collectors.news_collector import NewsCollectorManager

# Initialize the manager
manager = NewsCollectorManager()

# Collect news for THYAO (Turkish Airlines)
articles = manager.collect_for_stock(
    stock_code="THYAO",
    days_back=7,
    limit_per_source=20
)

# Display results
for article in articles:
    print(f"{article.source}: {article.title}")
    print(f"Published: {article.published_date}")
    print(f"URL: {article.url}")
    print(f"Stock codes: {article.stock_codes}")
    print("-" * 80)
```

### Example 2: Collect KAP Disclosures Only

```python
from src.data.collectors.news_collector import KAPCollector
from datetime import datetime, timedelta

# Initialize KAP collector
kap = KAPCollector()

# Collect disclosures for AKBNK (Akbank)
disclosures = kap.collect_news(
    stock_code="AKBNK",
    start_date=datetime.now() - timedelta(days=30),
    limit=10
)

for disclosure in disclosures:
    print(f"KAP: {disclosure.title}")
    print(f"Date: {disclosure.published_date}")
```

### Example 3: Get Latest Market News

```python
from src.data.collectors.news_collector import NewsCollectorManager

manager = NewsCollectorManager()

# Get 50 latest news articles from all sources
latest_news = manager.get_latest_news(limit=50)

print(f"Collected {len(latest_news)} articles")
```

### Example 4: Collect from Specific Sources

```python
from src.data.collectors.news_collector import (
    NewsCollectorManager,
    NewsSource
)

manager = NewsCollectorManager()

# Collect only from KAP and Bloomberg HT
articles = manager.collect_from_all_sources(
    limit_per_source=20,
    sources=[NewsSource.KAP, NewsSource.BLOOMBERG_HT]
)
```

## API Reference

### NewsArticle

Data class representing a news article.

**Attributes:**
- `source` (str): News source identifier
- `title` (str): Article title
- `url` (str): Article URL
- `published_date` (datetime): Publication date
- `stock_codes` (List[str]): List of BIST stock codes mentioned
- `content` (Optional[str]): Full article content
- `summary` (Optional[str]): Article summary
- `sentiment_score` (Optional[float]): Sentiment score (-1 to 1)
- `category` (Optional[str]): Article category
- `author` (Optional[str]): Article author
- `tags` (Optional[List[str]]): Article tags

**Methods:**
- `to_dict()`: Convert article to dictionary

### BaseNewsCollector

Abstract base class for all news collectors.

**Methods:**
- `collect_news(stock_code, start_date, end_date, limit)`: Collect news articles

### KAPCollector

Collector for KAP (Public Disclosure Platform).

```python
from src.data.collectors.news_collector import KAPCollector

collector = KAPCollector(
    cache_enabled=True,  # Enable caching
    rate_limit=1.0       # Seconds between requests
)

articles = collector.collect_news(
    stock_code="THYAO",              # Optional: filter by stock
    start_date=datetime(2024, 1, 1), # Optional: start date
    end_date=datetime(2024, 1, 31),  # Optional: end date
    limit=100                         # Optional: max articles
)
```

### BloombergHTCollector

Collector for Bloomberg HT news.

```python
from src.data.collectors.news_collector import BloombergHTCollector

collector = BloombergHTCollector()
articles = collector.collect_news(limit=50)
```

### NewsCollectorManager

Manager to coordinate multiple collectors.

**Methods:**

#### `collect_from_all_sources()`
```python
articles = manager.collect_from_all_sources(
    stock_code="THYAO",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    limit_per_source=50,
    sources=[NewsSource.KAP, NewsSource.BLOOMBERG_HT]
)
```

#### `collect_for_stock()`
```python
articles = manager.collect_for_stock(
    stock_code="THYAO",
    days_back=7,
    limit_per_source=20
)
```

#### `get_latest_news()`
```python
articles = manager.get_latest_news(
    limit=100,
    sources=[NewsSource.KAP]
)
```

## Advanced Usage

### Adding a Custom Collector

```python
from src.data.collectors.news_collector import BaseNewsCollector, NewsArticle

class MyCustomCollector(BaseNewsCollector):
    """Custom news collector"""

    BASE_URL = "https://example.com"

    def collect_news(self, stock_code=None, start_date=None,
                    end_date=None, limit=None):
        articles = []

        # Your scraping logic here
        response = self._fetch_page(self.BASE_URL)

        if response:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Parse articles...

        return articles

# Use the custom collector
custom = MyCustomCollector()
articles = custom.collect_news(limit=10)

# Or add to manager
from src.data.collectors.news_collector import NewsSource
manager.add_collector(NewsSource.MYNET_FINANS, custom)
```

### Using Scrapy for Large-Scale Scraping

```python
from src.data.collectors.news_collector import KAPSpider
from scrapy.crawler import CrawlerProcess

# Configure Scrapy
process = CrawlerProcess(settings={
    'USER_AGENT': 'Mozilla/5.0',
    'ROBOTSTXT_OBEY': True,
    'CONCURRENT_REQUESTS': 8,
    'DOWNLOAD_DELAY': 1,
})

# Run the spider
process.crawl(KAPSpider)
process.start()
```

### Export to Database

```python
from sqlalchemy import create_engine
import pandas as pd

# Collect articles
manager = NewsCollectorManager()
articles = manager.get_latest_news(limit=100)

# Convert to DataFrame
data = [article.to_dict() for article in articles]
df = pd.DataFrame(data)

# Save to database
engine = create_engine('sqlite:///bist_news.db')
df.to_sql('news_articles', engine, if_exists='append', index=False)
```

### Integration with NLP Module

```python
from src.data.collectors.news_collector import NewsCollectorManager
# from src.models.nlp.turkish_sentiment import TurkishSentimentAnalyzer  # Future

manager = NewsCollectorManager()
articles = manager.collect_for_stock("THYAO", days_back=7)

# Analyze sentiment (placeholder for future implementation)
# analyzer = TurkishSentimentAnalyzer()
# for article in articles:
#     article.sentiment_score = analyzer.analyze(article.content)
```

## Configuration

### Rate Limiting

Control request frequency to respect websites:

```python
collector = KAPCollector(rate_limit=2.0)  # 2 seconds between requests
```

### Caching

Enable caching to avoid redundant requests:

```python
collector = KAPCollector(cache_enabled=True)
```

### Timeouts

Customize request timeout:

```python
response = collector._fetch_page(url, timeout=60)
```

## Best Practices

1. **Respect Rate Limits**: Always use rate limiting to avoid overloading servers
2. **Cache Results**: Enable caching for frequently accessed data
3. **Error Handling**: Wrap collection calls in try-except blocks
4. **Filter Early**: Use stock_code parameter to reduce unnecessary requests
5. **Regular Updates**: Run collectors periodically (e.g., every 30 minutes)
6. **Store Results**: Save collected articles to database for historical analysis

## Troubleshooting

### Common Issues

**Issue**: "Connection timeout"
```python
# Solution: Increase timeout
collector = KAPCollector()
response = collector._fetch_page(url, timeout=60)
```

**Issue**: "Too many requests / Rate limited"
```python
# Solution: Increase rate limit
collector = KAPCollector(rate_limit=3.0)  # 3 seconds between requests
```

**Issue**: "No articles found"
```python
# Solution: Check date range and stock code
articles = collector.collect_news(
    stock_code="THYAO",  # Verify stock code is correct
    start_date=datetime.now() - timedelta(days=30),  # Expand date range
    limit=100
)
```

**Issue**: "Scrapy not available"
```python
# Solution: Install Scrapy
pip install scrapy
```

## Performance Tips

1. **Parallel Collection**: Use threading for multiple stocks
2. **Batch Processing**: Collect in batches instead of one-by-one
3. **Database Indexing**: Index by stock_code and published_date
4. **Cache Frequently**: Cache popular stocks and recent articles

## Future Enhancements

- [ ] Sentiment analysis integration
- [ ] Real-time streaming from news sources
- [ ] More Turkish financial news sources
- [ ] Twitter/X integration for real-time updates
- [ ] Image and chart extraction from articles
- [ ] Translation to English for international sources
- [ ] Topic modeling and categorization
- [ ] Alert system for important news

## Related Modules

- **NLP Module** (`src/models/nlp/`): Sentiment analysis and text processing
- **Data Storage** (`src/data/storage/`): Database operations for news articles
- **Signal Generation** (`src/signals/`): Incorporate news sentiment into trading signals

## Examples

See `examples/news_collector_example.py` for complete working examples.

## Tests

Run tests with:

```bash
python -m pytest tests/test_data/test_news_collector.py -v
```

## License

Part of the BIST AI Trading System project.

## Contributing

When adding new news sources:

1. Create a new collector class inheriting from `BaseNewsCollector`
2. Implement the `collect_news()` method
3. Add appropriate error handling
4. Include rate limiting
5. Add to `NewsSource` enum
6. Update documentation
7. Add tests

## Contact

For issues or questions, please refer to the main project documentation.
