"""
Unit tests for the News Collector module

Tests the functionality of news collectors for Turkish financial news
and KAP disclosures.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.collectors.news_collector import (
    NewsArticle,
    BaseNewsCollector,
    KAPCollector,
    BloombergHTCollector,
    InvestingComCollector,
    NewsCollectorManager,
    NewsSource
)


class TestNewsArticle(unittest.TestCase):
    """Test the NewsArticle dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        self.article = NewsArticle(
            source="test_source",
            title="Test Article",
            url="https://example.com/article",
            published_date=datetime(2024, 1, 15, 10, 30),
            stock_codes=["THYAO", "AKBNK"],
            content="Test content",
            category="test"
        )

    def test_article_creation(self):
        """Test creating a NewsArticle"""
        self.assertEqual(self.article.source, "test_source")
        self.assertEqual(self.article.title, "Test Article")
        self.assertEqual(len(self.article.stock_codes), 2)
        self.assertIn("THYAO", self.article.stock_codes)

    def test_to_dict(self):
        """Test converting article to dictionary"""
        article_dict = self.article.to_dict()

        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict['source'], "test_source")
        self.assertEqual(article_dict['title'], "Test Article")
        self.assertIn('published_date', article_dict)
        self.assertEqual(len(article_dict['stock_codes']), 2)

    def test_to_dict_with_optional_fields(self):
        """Test to_dict with optional fields"""
        article = NewsArticle(
            source="test",
            title="Test",
            url="https://example.com",
            published_date=datetime.now(),
            stock_codes=[],
            sentiment_score=0.75,
            tags=["tag1", "tag2"]
        )

        article_dict = article.to_dict()
        self.assertEqual(article_dict['sentiment_score'], 0.75)
        self.assertEqual(len(article_dict['tags']), 2)


class TestBaseNewsCollector(unittest.TestCase):
    """Test the BaseNewsCollector abstract class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a concrete implementation for testing
        class TestCollector(BaseNewsCollector):
            def collect_news(self, stock_code=None, start_date=None,
                           end_date=None, limit=None):
                return []

        self.collector = TestCollector()

    def test_session_creation(self):
        """Test that session is created with proper headers"""
        self.assertIsNotNone(self.collector.session)
        self.assertIn('User-Agent', self.collector.session.headers)

    def test_extract_stock_codes(self):
        """Test stock code extraction from text"""
        text = "THYAO ve AKBNK hisseleri yükseldi. GARAN da pozitif."

        codes = self.collector._extract_stock_codes(text)

        self.assertIn("THYAO", codes)
        self.assertIn("AKBNK", codes)
        self.assertIn("GARAN", codes)

    def test_extract_stock_codes_filters_stop_words(self):
        """Test that stop words are filtered out"""
        text = "Hisse fiyatları VE TL bazında yükseldi."

        codes = self.collector._extract_stock_codes(text)

        self.assertNotIn("VE", codes)
        self.assertNotIn("TL", codes)

    def test_rate_limiting(self):
        """Test rate limiting between requests"""
        self.collector.rate_limit = 0.1  # 100ms
        start_time = datetime.now()

        self.collector._rate_limit_wait()
        self.collector._rate_limit_wait()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.assertGreaterEqual(elapsed, 0.1)

    @patch('requests.Session.get')
    def test_fetch_page_success(self, mock_get):
        """Test successful page fetch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Test</html>"
        mock_get.return_value = mock_response

        response = self.collector._fetch_page("https://example.com")

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    @patch('requests.Session.get')
    def test_fetch_page_error(self, mock_get):
        """Test page fetch with error"""
        mock_get.side_effect = Exception("Connection error")

        response = self.collector._fetch_page("https://example.com")

        self.assertIsNone(response)


class TestKAPCollector(unittest.TestCase):
    """Test the KAP collector"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = KAPCollector(rate_limit=0)

    def test_initialization(self):
        """Test KAP collector initialization"""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.BASE_URL, "https://www.kap.org.tr")

    def test_disclosure_types(self):
        """Test disclosure type mapping"""
        self.assertIn('FR', self.collector.disclosure_types)
        self.assertIn('OI', self.collector.disclosure_types)

    @patch.object(KAPCollector, '_fetch_page')
    def test_collect_news_with_mock(self, mock_fetch):
        """Test collecting news with mocked response"""
        # Mock HTML response
        mock_response = Mock()
        mock_response.content = b"""
        <html>
            <div class="w-clearfix w-row export-row">
                <div class="column-type">Test Disclosure</div>
                <div class="column-date">15.01.2024 10:30:00</div>
                <a class="disclosureLink" href="/tr/bildirim/123" data-id="123"></a>
            </div>
        </html>
        """
        mock_fetch.return_value = mock_response

        articles = self.collector.collect_news(
            stock_code="THYAO",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            limit=10
        )

        # Should attempt to fetch
        self.assertIsInstance(articles, list)


class TestBloombergHTCollector(unittest.TestCase):
    """Test the Bloomberg HT collector"""

    def setUp(self):
        """Set up test fixtures"""
        self.collector = BloombergHTCollector(rate_limit=0)

    def test_initialization(self):
        """Test Bloomberg HT collector initialization"""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.BASE_URL, "https://www.bloomberght.com")

    @patch.object(BloombergHTCollector, '_fetch_page')
    def test_collect_news(self, mock_fetch):
        """Test collecting news with mocked response"""
        mock_response = Mock()
        mock_response.content = b"""
        <html>
            <article class="news-item">
                <h3>THYAO hisseleri yükseldi</h3>
                <a href="/hisse/thyao-123"></a>
                <time datetime="2024-01-15T10:30:00">15 Ocak 2024</time>
            </article>
        </html>
        """
        mock_fetch.return_value = mock_response

        articles = self.collector.collect_news(limit=10)

        self.assertIsInstance(articles, list)


class TestNewsCollectorManager(unittest.TestCase):
    """Test the NewsCollectorManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = NewsCollectorManager()

    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertIn(NewsSource.KAP, self.manager.collectors)
        self.assertIn(NewsSource.BLOOMBERG_HT, self.manager.collectors)

    def test_add_collector(self):
        """Test adding a custom collector"""
        class CustomCollector(BaseNewsCollector):
            def collect_news(self, stock_code=None, start_date=None,
                           end_date=None, limit=None):
                return []

        custom = CustomCollector()
        custom_source = NewsSource.MYNET_FINANS

        self.manager.add_collector(custom_source, custom)

        self.assertIn(custom_source, self.manager.collectors)
        self.assertEqual(self.manager.collectors[custom_source], custom)

    @patch.object(KAPCollector, 'collect_news')
    @patch.object(BloombergHTCollector, 'collect_news')
    def test_collect_from_all_sources(self, mock_bloomberg, mock_kap):
        """Test collecting from all sources"""
        # Mock return values
        kap_article = NewsArticle(
            source=NewsSource.KAP.value,
            title="KAP Test",
            url="https://kap.org.tr/test",
            published_date=datetime.now(),
            stock_codes=["THYAO"]
        )

        bloomberg_article = NewsArticle(
            source=NewsSource.BLOOMBERG_HT.value,
            title="Bloomberg Test",
            url="https://bloomberght.com/test",
            published_date=datetime.now(),
            stock_codes=["AKBNK"]
        )

        mock_kap.return_value = [kap_article]
        mock_bloomberg.return_value = [bloomberg_article]

        articles = self.manager.collect_from_all_sources(limit_per_source=10)

        self.assertEqual(len(articles), 2)
        sources = [a.source for a in articles]
        self.assertIn(NewsSource.KAP.value, sources)
        self.assertIn(NewsSource.BLOOMBERG_HT.value, sources)

    @patch.object(KAPCollector, 'collect_news')
    def test_collect_for_stock(self, mock_collect):
        """Test collecting news for specific stock"""
        mock_article = NewsArticle(
            source=NewsSource.KAP.value,
            title="THYAO Test",
            url="https://kap.org.tr/test",
            published_date=datetime.now(),
            stock_codes=["THYAO"]
        )

        mock_collect.return_value = [mock_article]

        articles = self.manager.collect_for_stock("THYAO", days_back=7)

        # Should call collect_news with stock code
        mock_collect.assert_called()
        self.assertIsInstance(articles, list)

    @patch.object(KAPCollector, 'collect_news')
    def test_get_latest_news(self, mock_collect):
        """Test getting latest news"""
        mock_articles = [
            NewsArticle(
                source=NewsSource.KAP.value,
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                published_date=datetime.now() - timedelta(hours=i),
                stock_codes=[]
            )
            for i in range(5)
        ]

        mock_collect.return_value = mock_articles

        articles = self.manager.get_latest_news(limit=10)

        self.assertIsInstance(articles, list)
        self.assertLessEqual(len(articles), 10)


class TestIntegration(unittest.TestCase):
    """Integration tests (require network access)"""

    @unittest.skip("Requires network access")
    def test_real_kap_collection(self):
        """Test real KAP collection (skipped by default)"""
        collector = KAPCollector()

        articles = collector.collect_news(
            start_date=datetime.now() - timedelta(days=7),
            limit=5
        )

        self.assertIsInstance(articles, list)
        if articles:
            self.assertIsInstance(articles[0], NewsArticle)

    @unittest.skip("Requires network access")
    def test_real_manager_collection(self):
        """Test real collection from manager (skipped by default)"""
        manager = NewsCollectorManager()

        articles = manager.get_latest_news(limit=10)

        self.assertIsInstance(articles, list)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNewsArticle))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBaseNewsCollector))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKAPCollector))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBloombergHTCollector))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNewsCollectorManager))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
