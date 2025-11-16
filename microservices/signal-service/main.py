"""
Signal Generation Microservice
Generates trading signals by aggregating ML predictions, sentiment, and whale data
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import requests
import sys

sys.path.insert(0, '/app')

from src.signals.generator import SignalGenerator, create_model_output
from src.signals.prioritizer import SignalPrioritizer, create_signal_input, PrioritizationStrategy
from src.signals.confidence import ConfidenceCalculator
from src.utils.logger import get_logger

app = FastAPI(
    title="BIST Signal Service",
    description="Trading signal generation microservice",
    version="1.0.0"
)

logger = get_logger("signal-service")

# Initialize components
signal_generator = SignalGenerator()
signal_prioritizer = SignalPrioritizer(strategy=PrioritizationStrategy.BALANCED)
confidence_calculator = ConfidenceCalculator()

# Service URLs
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000")
NEWS_SERVICE_URL = os.getenv("NEWS_SERVICE_URL", "http://news-service:8000")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data-service:8000")

# Models
class SignalRequest(BaseModel):
    symbols: List[str]
    strategy: Optional[str] = "BALANCED"

class SignalResponse(BaseModel):
    symbol: str
    signal: str
    confidence: float
    target_price: float
    stop_loss: float
    take_profit: float

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "signal-service",
        "timestamp": datetime.now().isoformat()
    }

# Generate Signals
@app.post("/api/v1/signals/generate")
async def generate_signals(request: SignalRequest):
    """Generate trading signals for symbols"""
    try:
        logger.info(f"Generating signals for {request.symbols}")

        all_signals = []

        for symbol in request.symbols:
            try:
                # Get data from data service
                data_response = requests.get(
                    f"{DATA_SERVICE_URL}/api/v1/data/ohlcv/{symbol}",
                    params={"timeframe": "1d", "period": "3mo"}
                )
                data_response.raise_for_status()
                ohlcv_data = data_response.json()

                # Get features (you would calculate these from OHLCV)
                features = [[1.0] * 50]  # Placeholder

                # Get ML predictions
                predictions = {}
                for model_name in ["xgboost", "lightgbm", "lstm"]:
                    try:
                        ml_response = requests.post(
                            f"{ML_SERVICE_URL}/api/v1/predict",
                            json={
                                "model_name": model_name,
                                "features": features
                            }
                        )
                        if ml_response.status_code == 200:
                            predictions[model_name] = ml_response.json()["predictions"][0]
                    except:
                        logger.warning(f"Could not get prediction from {model_name}")

                # Get news sentiment
                sentiment_score = 0.0
                try:
                    news_response = requests.get(
                        f"{NEWS_SERVICE_URL}/api/v1/news/stock/{symbol}",
                        params={"days_back": 3, "limit": 10}
                    )
                    if news_response.status_code == 200:
                        news_data = news_response.json()
                        # Calculate average sentiment from articles
                        # This is simplified - you'd use the actual sentiment analyzer
                        sentiment_score = 0.5
                except:
                    logger.warning(f"Could not get news for {symbol}")

                # Create model outputs
                model_outputs = []
                if "xgboost" in predictions:
                    model_outputs.append(create_model_output(
                        "xgboost", "regression", predictions["xgboost"], 0.75
                    ))
                if "lightgbm" in predictions:
                    model_outputs.append(create_model_output(
                        "lightgbm", "regression", predictions["lightgbm"], 0.70
                    ))

                # Add sentiment
                model_outputs.append(create_model_output(
                    "sentiment", "nlp", sentiment_score, 0.60
                ))

                # Generate signal
                current_price = 100.0  # Get from data
                signal = signal_generator.generate_signal(
                    stock_code=symbol,
                    model_outputs=model_outputs,
                    current_price=current_price,
                    historical_prices=None  # Would pass actual prices
                )

                all_signals.append({
                    "symbol": symbol,
                    "signal": signal.signal.name,
                    "confidence": signal.confidence_score,
                    "target_price": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "position_size": signal.position_size,
                    "generated_at": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue

        return {
            "status": "success",
            "signals": all_signals,
            "count": len(all_signals)
        }

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Prioritize Signals
@app.post("/api/v1/signals/prioritize")
async def prioritize_signals(signals: List[Dict]):
    """Prioritize trading signals"""
    try:
        logger.info(f"Prioritizing {len(signals)} signals")

        signal_inputs = []
        for sig in signals:
            signal_input = create_signal_input(
                symbol=sig["symbol"],
                signal_direction=sig["signal"],
                confidence_score=sig.get("confidence", 50.0),
                wai_score=sig.get("wai_score", 50.0),
                news_sentiment=sig.get("sentiment", 0.0),
                current_price=sig.get("current_price", 100.0),
                target_price=sig.get("target_price", 105.0)
            )
            signal_inputs.append(signal_input)

        # Prioritize
        prioritized = signal_prioritizer.prioritize_signals(signal_inputs)

        return {
            "status": "success",
            "signals": [
                {
                    "symbol": s.symbol,
                    "signal": s.signal_direction,
                    "priority_score": s.priority_score,
                    "risk_adjusted_score": s.risk_adjusted_score,
                    "rank": s.rank
                }
                for s in prioritized
            ]
        }

    except Exception as e:
        logger.error(f"Error prioritizing signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=8000)
