"""
Data Collection Microservice
Handles BIST OHLCV, fundamental, macro, and whale data collection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import os
import sys

# Add src to path
sys.path.insert(0, '/app')

from src.data.collectors.bist_collector import BISTCollector
from src.data.collectors.fundamental_collector import FundamentalCollector
from src.data.collectors.macro_collector import MacroCollector
from src.data.collectors.whale_collector import WhaleCollector
from src.data.processors.cleaner import OHLCVCleaner
from src.data.processors.validator import DataValidator
from src.utils.logger import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="BIST Data Service",
    description="Microservice for collecting and processing BIST market data",
    version="1.0.0"
)

logger = get_logger("data-service")

# Initialize collectors
bist_collector = BISTCollector()
fundamental_collector = FundamentalCollector()
macro_collector = MacroCollector()
whale_collector = WhaleCollector()

# Request/Response Models
class OHLCVRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    period: str = "1mo"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FundamentalRequest(BaseModel):
    symbol: str
    period: str = "quarterly"

class MacroRequest(BaseModel):
    indicators: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class WhaleRequest(BaseModel):
    symbol: str
    days_back: int = 30

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-service",
        "timestamp": datetime.now().isoformat()
    }

# OHLCV Data Endpoints
@app.post("/api/v1/data/ohlcv")
async def get_ohlcv_data(request: OHLCVRequest):
    """Collect OHLCV data for specified symbols"""
    try:
        logger.info(f"Collecting OHLCV data for {request.symbols}, timeframe: {request.timeframe}")

        result = {}
        for symbol in request.symbols:
            df = bist_collector.get_historical_data(
                symbol=symbol,
                timeframe=request.timeframe,
                period=request.period,
                start_date=request.start_date,
                end_date=request.end_date
            )

            if df is not None and not df.empty:
                # Clean data
                cleaner = OHLCVCleaner(df)
                cleaned = cleaner.clean_pipeline()

                # Convert to dict
                result[symbol] = {
                    "data": cleaned.to_dict(orient='records'),
                    "records": len(cleaned)
                }

        return {
            "status": "success",
            "symbols": request.symbols,
            "timeframe": request.timeframe,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error collecting OHLCV data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/ohlcv/{symbol}")
async def get_single_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    period: str = "1mo"
):
    """Get OHLCV data for a single symbol"""
    try:
        df = bist_collector.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            period=period
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "data": df.to_dict(orient='records'),
            "records": len(df)
        }

    except Exception as e:
        logger.error(f"Error getting OHLCV for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fundamental Data Endpoints
@app.post("/api/v1/data/fundamental")
async def get_fundamental_data(request: FundamentalRequest):
    """Collect fundamental data for a symbol"""
    try:
        logger.info(f"Collecting fundamental data for {request.symbol}")

        metrics = fundamental_collector.get_comprehensive_metrics(
            request.symbol,
            request.period
        )

        return {
            "status": "success",
            "symbol": request.symbol,
            "period": request.period,
            "data": metrics.__dict__ if metrics else None
        }

    except Exception as e:
        logger.error(f"Error collecting fundamental data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Macro Data Endpoints
@app.post("/api/v1/data/macro")
async def get_macro_data(request: MacroRequest):
    """Collect macroeconomic indicators"""
    try:
        logger.info(f"Collecting macro indicators: {request.indicators}")

        result = macro_collector.get_multiple_indicators(
            request.indicators,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return {
            "status": "success",
            "indicators": request.indicators,
            "data": {k: v.to_dict(orient='records') for k, v in result.items()}
        }

    except Exception as e:
        logger.error(f"Error collecting macro data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Whale/Flow Data Endpoints
@app.post("/api/v1/data/whale")
async def get_whale_data(request: WhaleRequest):
    """Collect whale/brokerage distribution data"""
    try:
        logger.info(f"Collecting whale data for {request.symbol}")

        # This would integrate with actual whale data source
        # For now, return placeholder
        return {
            "status": "success",
            "symbol": request.symbol,
            "days_back": request.days_back,
            "message": "Whale data collection requires additional configuration"
        }

    except Exception as e:
        logger.error(f"Error collecting whale data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Operations
@app.post("/api/v1/data/collect-all")
async def collect_all_data(
    background_tasks: BackgroundTasks,
    symbols: List[str],
    timeframe: str = "1d"
):
    """Trigger collection of all data types for symbols"""

    def collect_task():
        logger.info(f"Starting batch collection for {symbols}")
        # Implement batch collection logic
        pass

    background_tasks.add_task(collect_task)

    return {
        "status": "started",
        "message": "Data collection started in background",
        "symbols": symbols,
        "timeframe": timeframe
    }

# Utility Endpoints
@app.get("/api/v1/data/symbols")
async def get_available_symbols():
    """Get list of available BIST symbols"""
    # Return BIST 100 symbols
    symbols = [
        "THYAO", "GARAN", "AKBNK", "ISCTR", "EREGL", "SISE", "TUPRS",
        "KCHOL", "SAHOL", "VAKBN", "YKBNK", "TCELL", "HALKB", "ASELS",
        "BIMAS", "PETKM", "ENKAI", "KOZAL", "MGROS", "ARCLK"
    ]

    return {
        "status": "success",
        "symbols": symbols,
        "count": len(symbols)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
