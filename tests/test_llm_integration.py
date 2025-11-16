#!/usr/bin/env python3
"""
Unit Tests for LLM Integration Module

Tests the LLM news analysis functionality including:
- Analyzer initialization
- News impact analysis
- Batch processing
- Caching mechanisms
- Prompt generation
- Result parsing

Author: BIST AI Trading System
Date: 2025-11-16
"""

import unittest
import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.nlp.llm_integration import (
    LLMNewsAnalyzer,
    create_analyzer,
    NewsImpactScore,
    SentimentImpact,
    LLMProvider,
    TurkishFinancialPrompts
)


class TestTurkishFinancialPrompts(unittest.TestCase):
    """Test prompt generation"""

    def test_get_system_prompt_turkish(self):
        """Test Turkish system prompt"""
        prompt = TurkishFinancialPrompts.get_system_prompt("turkish")
        self.assertIsInstance(prompt, str)
        self.assertIn("BIST", prompt)
        self.assertIn("finansal", prompt.lower())

    def test_get_system_prompt_english(self):
        """Test English system prompt"""
        prompt = TurkishFinancialPrompts.get_system_prompt("english")
        self.assertIsInstance(prompt, str)
        self.assertIn("BIST", prompt)
        self.assertIn("financial", prompt.lower())

    def test_get_analysis_prompt(self):
        """Test analysis prompt generation"""
        news = "Test news content"
        prompt = TurkishFinancialPrompts.get_analysis_prompt(
            stock_code="AKBNK",
            news_summaries=news,
            language="turkish"
        )
        self.assertIn("AKBNK", prompt)
        self.assertIn(news, prompt)


class TestNewsImpactScore(unittest.TestCase):
    """Test NewsImpactScore data class"""

    def test_create_impact_score(self):
        """Test creating impact score"""
        score = NewsImpactScore(
            stock_code="THYAO",
            impact_category=SentimentImpact.POSITIVE,
            impact_score=0.65,
            confidence=0.85,
            reasoning="Strong earnings",
            key_factors=["earnings", "growth"],
            analyzed_articles=3,
            analysis_timestamp=datetime.now(),
            short_term_outlook="Bullish",
            risk_level="low"
        )

        self.assertEqual(score.stock_code, "THYAO")
        self.assertEqual(score.impact_score, 0.65)
        self.assertEqual(score.confidence, 0.85)
        self.assertEqual(len(score.key_factors), 2)

    def test_to_dict(self):
        """Test converting to dictionary"""
        score = NewsImpactScore(
            stock_code="AKBNK",
            impact_category=SentimentImpact.NEUTRAL,
            impact_score=0.0,
            confidence=0.5,
            reasoning="Mixed signals",
            key_factors=["factor1"],
            analyzed_articles=2,
            analysis_timestamp=datetime.now(),
            short_term_outlook="Neutral"
        )

        result_dict = score.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['stock_code'], "AKBNK")
        self.assertEqual(result_dict['impact_category'], "neutral")
        self.assertIsInstance(result_dict['analysis_timestamp'], str)


class TestLLMNewsAnalyzer(unittest.TestCase):
    """Test LLM News Analyzer functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_news = [
            {
                'title': 'Test news 1',
                'content': 'Positive earnings report',
                'source': 'Test Source',
                'published_date': '2024-11-15'
            },
            {
                'title': 'Test news 2',
                'content': 'Market expansion',
                'source': 'Test Source',
                'published_date': '2024-11-14'
            }
        ]

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', False)
    def test_initialization_without_openai(self):
        """Test initialization fails without OpenAI when using OpenAI provider"""
        with self.assertRaises(ImportError):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)

    def test_create_analyzer_function(self):
        """Test create_analyzer convenience function"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True):
                with patch('src.models.nlp.llm_integration.OpenAI'):
                    analyzer = create_analyzer(provider="openai", model_name="gpt-3.5-turbo")
                    self.assertIsInstance(analyzer, LLMNewsAnalyzer)
                    self.assertEqual(analyzer.provider, LLMProvider.OPENAI)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_format_news_for_prompt(self, mock_openai):
        """Test news formatting for prompts"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)
            formatted = analyzer._format_news_for_prompt(self.sample_news, True)

            self.assertIsInstance(formatted, str)
            self.assertIn("Test news 1", formatted)
            self.assertIn("Positive earnings report", formatted)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_cache_key_generation(self, mock_openai):
        """Test cache key generation"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)
            key1 = analyzer._generate_cache_key("AKBNK", self.sample_news)
            key2 = analyzer._generate_cache_key("AKBNK", self.sample_news)
            key3 = analyzer._generate_cache_key("THYAO", self.sample_news)

            # Same input should generate same key
            self.assertEqual(key1, key2)
            # Different stock should generate different key
            self.assertNotEqual(key1, key3)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_parse_impact_category(self, mock_openai):
        """Test parsing impact categories"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)

            self.assertEqual(
                analyzer._parse_impact_category("positive"),
                SentimentImpact.POSITIVE
            )
            self.assertEqual(
                analyzer._parse_impact_category("very negative"),
                SentimentImpact.VERY_NEGATIVE
            )
            self.assertEqual(
                analyzer._parse_impact_category("unknown"),
                SentimentImpact.NEUTRAL
            )

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_create_neutral_score(self, mock_openai):
        """Test creating neutral fallback score"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)
            score = analyzer._create_neutral_score("EREGL")

            self.assertEqual(score.stock_code, "EREGL")
            self.assertEqual(score.impact_category, SentimentImpact.NEUTRAL)
            self.assertEqual(score.impact_score, 0.0)
            self.assertEqual(score.confidence, 0.0)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_parse_llm_response(self, mock_openai):
        """Test parsing LLM JSON response"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)

            # Valid JSON response
            response = '''
            {
                "impact_score": 0.75,
                "confidence": 0.85,
                "impact_category": "positive",
                "reasoning": "Strong earnings",
                "key_factors": ["earnings", "growth"],
                "short_term_outlook": "Bullish",
                "risk_level": "low"
            }
            '''

            result = analyzer._parse_llm_response("AKBNK", response, 3)

            self.assertEqual(result.stock_code, "AKBNK")
            self.assertEqual(result.impact_score, 0.75)
            self.assertEqual(result.confidence, 0.85)
            self.assertEqual(result.impact_category, SentimentImpact.POSITIVE)
            self.assertEqual(result.analyzed_articles, 3)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_clear_cache(self, mock_openai):
        """Test cache clearing"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI, cache_enabled=True)

            # Add something to cache
            test_score = analyzer._create_neutral_score("TEST")
            analyzer._cache['test_key'] = (test_score, 123456)

            self.assertEqual(len(analyzer._cache), 1)

            # Clear cache
            analyzer.clear_cache()
            self.assertEqual(len(analyzer._cache), 0)

    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    @patch('src.models.nlp.llm_integration.OpenAI')
    def test_analyze_empty_news(self, mock_openai):
        """Test analyzing with empty news list"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            analyzer = LLMNewsAnalyzer(provider=LLMProvider.OPENAI)
            result = analyzer.analyze_news_impact("THYAO", [])

            self.assertEqual(result.stock_code, "THYAO")
            self.assertEqual(result.impact_category, SentimentImpact.NEUTRAL)
            self.assertEqual(result.analyzed_articles, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests (require API key)"""

    @unittest.skipUnless(os.getenv('OPENAI_API_KEY'), "OpenAI API key not set")
    @patch('src.models.nlp.llm_integration.OPENAI_AVAILABLE', True)
    def test_real_openai_analysis(self):
        """Test with real OpenAI API (if key is available)"""
        analyzer = create_analyzer(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )

        news = [
            {
                'title': 'Garanti BBVA güçlü kar açıkladı',
                'content': 'Banka beklentileri aşan kar rakamları açıkladı',
                'source': 'Test',
                'published_date': '2024-11-15'
            }
        ]

        result = analyzer.analyze_news_impact("GARAN", news)

        # Verify result structure
        self.assertEqual(result.stock_code, "GARAN")
        self.assertIsInstance(result.impact_score, float)
        self.assertGreaterEqual(result.impact_score, -1.0)
        self.assertLessEqual(result.impact_score, 1.0)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.reasoning, str)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestTurkishFinancialPrompts))
    suite.addTests(loader.loadTestsFromTestCase(TestNewsImpactScore))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMNewsAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
