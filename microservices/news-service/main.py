"""
News Collection and Analysis Microservice
Handles Turkish financial news collection, sentiment analysis, and LLM integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/app')

from src.data.collectors.news_collector import NewsCollectorManager
from src.models.nlp.turkish_sentiment import TurkishFinancialSentimentAnalyzer
from src.models.nlp.news_analyzer import NewsAnalyzer
from src.models.nlp.llm_integration import LLMNewsAnalyzer
from src.utils.logger import get_logger

app = FastAPI(
    title="BIST News Service",
    description="Microservice for news collection and sentiment analysis",
    version="1.0.0"
)

logger = get_logger("news-service")

# Initialize components
news_collector = NewsCollectorManager()
sentiment_analyzer = TurkishFinancialSentimentAnalyzer()
news_analyzer = NewsAnalyzer()
llm_analyzer = LLMNewsAnalyzer(provider="openai")

# Models
class NewsCollectionRequest(BaseModel):
    stock_codes: Optional[List[str]] = None
    days_back: int = 7
    limit_per_source: int = 50

class SentimentAnalysisRequest(BaseModel):
    texts: List[str]

class NewsImpactRequest(BaseModel):
    stock_code: str
    articles: List[Dict]

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "news-service",
        "timestamp": datetime.now().isoformat()
    }

# News Collection
@app.post("/api/v1/news/collect")
async def collect_news(request: NewsCollectionRequest):
    """Collect Turkish financial news"""
    try:
        logger.info(f"Collecting news for {request.stock_codes}")

        articles = []

        if request.stock_codes:
            for stock_code in request.stock_codes:
                stock_articles = news_collector.collect_for_stock(
                    stock_code=stock_code,
                    days_back=request.days_back,
                    limit_per_source=request.limit_per_source
                )
                articles.extend([a.to_dict() for a in stock_articles])
        else:
            # Collect general market news
            all_articles = news_collector.collect_latest_news(
                days_back=request.days_back,
                limit_per_source=request.limit_per_source
            )
            articles = [a.to_dict() for a in all_articles]

        return {
            "status": "success",
            "articles_count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Error collecting news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/news/stock/{stock_code}")
async def get_stock_news(
    stock_code: str,
    days_back: int = 7,
    limit: int = 50
):
    """Get news for a specific stock"""
    try:
        articles = news_collector.collect_for_stock(
            stock_code=stock_code,
            days_back=days_back,
            limit_per_source=limit
        )

        return {
            "status": "success",
            "stock_code": stock_code,
            "articles_count": len(articles),
            "articles": [a.to_dict() for a in articles]
        }

    except Exception as e:
        logger.error(f"Error getting news for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Sentiment Analysis
@app.post("/api/v1/news/sentiment")
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of Turkish texts"""
    try:
        logger.info(f"Analyzing sentiment for {len(request.texts)} texts")

        results = []
        for text in request.texts:
            prediction = sentiment_analyzer.predict(text)
            results.append({
                "text": text[:100] + "...",  # Truncate for response
                "sentiment": prediction.label.name,
                "score": prediction.sentiment_score,
                "confidence": prediction.confidence
            })

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# News Analysis
@app.post("/api/v1/news/analyze")
async def analyze_news_article(
    title: str,
    content: str,
    published_date: Optional[str] = None
):
    """Comprehensive news analysis"""
    try:
        pub_date = datetime.fromisoformat(published_date) if published_date else datetime.now()

        analysis = news_analyzer.analyze_article(
            title=title,
            content=content,
            published_date=pub_date
        )

        return {
            "status": "success",
            "stock_codes": analysis.stock_codes,
            "sentiment": {
                "score": analysis.sentiment.score,
                "label": analysis.sentiment.label.name,
                "confidence": analysis.sentiment.confidence
            },
            "events": [
                {
                    "type": e.event_type.name,
                    "impact": e.impact_score,
                    "description": e.description
                }
                for e in analysis.events
            ],
            "keywords": analysis.keywords,
            "relevance": analysis.relevance_score
        }

    except Exception as e:
        logger.error(f"Error analyzing news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM Impact Analysis
@app.post("/api/v1/news/impact")
async def analyze_news_impact(request: NewsImpactRequest):
    """LLM-based news impact analysis"""
    try:
        logger.info(f"Analyzing news impact for {request.stock_code}")

        impact = llm_analyzer.analyze_news_impact(
            stock_code=request.stock_code,
            news_articles=request.articles
        )

        return {
            "status": "success",
            "stock_code": request.stock_code,
            "impact_score": impact.impact_score,
            "impact_category": impact.impact_category.value,
            "confidence": impact.confidence,
            "reasoning": impact.reasoning,
            "key_factors": impact.key_factors,
            "short_term_outlook": impact.short_term_outlook,
            "medium_term_outlook": impact.medium_term_outlook
        }

    except Exception as e:
        logger.error(f"Error analyzing news impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background Tasks
@app.post("/api/v1/news/collect-and-analyze/{stock_code}")
async def collect_and_analyze(
    background_tasks: BackgroundTasks,
    stock_code: str,
    days_back: int = 3
):
    """Collect news and analyze in background"""

    def process_task():
        # Collect news
        articles = news_collector.collect_for_stock(stock_code, days_back)

        # Analyze each article
        for article in articles:
            analysis = news_analyzer.analyze_article(
                title=article.title,
                content=article.content,
                published_date=article.published_date
            )
            # Store analysis (would save to database in production)
            logger.info(f"Analyzed article: {article.title}")

    background_tasks.add_task(process_task)

    return {
        "status": "started",
        "message": "News collection and analysis started in background",
        "stock_code": stock_code
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
