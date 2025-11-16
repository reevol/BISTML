"""
Machine Learning Models Microservice
Handles model training, prediction, and serving
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import sys
import os

sys.path.insert(0, '/app')

from src.models.forecasting.lstm_model import LSTMPriceForecaster
from src.models.forecasting.xgboost_model import XGBoostPricePredictor
from src.models.forecasting.lightgbm_model import LightGBMForecaster
from src.models.classification.random_forest import TradingSignalClassifier
from src.models.classification.ensemble import EnsembleClassifier
from src.utils.logger import get_logger

app = FastAPI(
    title="BIST ML Service",
    description="Microservice for ML model training and predictions",
    version="1.0.0"
)

logger = get_logger("ml-service")

# Model registry
loaded_models = {}
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")

# Models
class PredictionRequest(BaseModel):
    model_name: str
    features: List[List[float]]
    feature_names: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    model_type: str  # lstm, xgboost, lightgbm, random_forest, ensemble
    X_train: List[List[float]]
    y_train: List[float]
    model_params: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    version: str
    trained_at: str

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-service",
        "models_loaded": len(loaded_models),
        "timestamp": datetime.now().isoformat()
    }

# Model Management
@app.get("/api/v1/models")
async def list_models():
    """List all available models"""
    models = []

    # Check loaded models
    for name, model_info in loaded_models.items():
        models.append({
            "name": name,
            "type": model_info.get("type"),
            "loaded": True,
            "loaded_at": model_info.get("loaded_at")
        })

    # Check saved models
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith(('.pkl', '.h5', '.pt')):
                model_name = filename.rsplit('.', 1)[0]
                if model_name not in loaded_models:
                    models.append({
                        "name": model_name,
                        "type": "unknown",
                        "loaded": False
                    })

    return {
        "status": "success",
        "models": models,
        "count": len(models)
    }

@app.post("/api/v1/models/load/{model_name}")
async def load_model(model_name: str):
    """Load a saved model"""
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = joblib.load(model_path)

        loaded_models[model_name] = {
            "model": model,
            "type": type(model).__name__,
            "loaded_at": datetime.now().isoformat()
        }

        logger.info(f"Loaded model: {model_name}")

        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully"
        }

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Predictions
@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """Make predictions using a loaded model"""
    try:
        if request.model_name not in loaded_models:
            # Try to load it
            await load_model(request.model_name)

        model_info = loaded_models[request.model_name]
        model = model_info["model"]

        # Convert features to numpy array
        X = np.array(request.features)

        # Make prediction
        predictions = model.predict(X)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X).tolist()

        return {
            "status": "success",
            "model": request.model_name,
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "probabilities": probabilities
        }

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Training
@app.post("/api/v1/train")
async def train_model(request: TrainingRequest):
    """Train a new model"""
    try:
        logger.info(f"Training {request.model_type} model")

        X = np.array(request.X_train)
        y = np.array(request.y_train)

        model = None
        model_name = None

        if request.model_type == "xgboost":
            model = XGBoostPricePredictor()
            model.train(X, y)
            model_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        elif request.model_type == "lightgbm":
            model = LightGBMForecaster()
            model.fit(X, y)
            model_name = f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        elif request.model_type == "random_forest":
            model = TradingSignalClassifier()
            model.fit(X, y)
            model_name = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        elif request.model_type == "ensemble":
            model = EnsembleClassifier()
            model.train_base_models(X, y)
            model.train_ensemble(X, y, ensemble_type='stacking')
            model_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        # Load into memory
        loaded_models[model_name] = {
            "model": model,
            "type": request.model_type,
            "loaded_at": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_name": model_name,
            "model_type": request.model_type
        }

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Predictions
@app.post("/api/v1/predict/batch")
async def batch_predict(
    model_name: str,
    features: List[List[float]]
):
    """Batch predictions for multiple samples"""
    try:
        if model_name not in loaded_models:
            await load_model(model_name)

        model = loaded_models[model_name]["model"]
        X = np.array(features)

        predictions = model.predict(X)

        return {
            "status": "success",
            "model": model_name,
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "count": len(predictions)
        }

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
