"""
Confidence Score Calculator for Trading Signals

This module calculates confidence scores (0-100%) for trading predictions based on:
1. Model predictions - consistency, probability scores, ensemble agreement
2. Feature quality - completeness, validity, statistical properties
3. Historical accuracy - past performance, recent track record
4. Prediction variance - uncertainty quantification, ensemble disagreement

The confidence score helps traders assess the reliability of trading signals
and make informed decisions about position sizing and risk management.

Key Features:
- Multi-factor confidence scoring
- Historical performance tracking
- Feature quality assessment
- Ensemble prediction analysis
- Adaptive weighting based on recent performance
- Detailed confidence breakdown

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
import logging
import warnings
from datetime import datetime, timedelta
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class ConfidenceWeights:
    """
    Weights for different components of confidence score.

    All weights should sum to 1.0 for normalized scoring.
    """
    model_prediction: float = 0.35      # Model output quality
    feature_quality: float = 0.25       # Input data quality
    historical_accuracy: float = 0.30   # Past performance
    prediction_variance: float = 0.10   # Uncertainty measure

    def __post_init__(self):
        """Validate that weights sum to 1.0"""
        total = (self.model_prediction + self.feature_quality +
                self.historical_accuracy + self.prediction_variance)
        if not np.isclose(total, 1.0):
            logger.warning(f"Confidence weights sum to {total:.3f}, not 1.0. Normalizing...")
            # Normalize weights
            self.model_prediction /= total
            self.feature_quality /= total
            self.historical_accuracy /= total
            self.prediction_variance /= total


@dataclass
class ConfidenceScore:
    """
    Container for confidence score and its components.

    Attributes:
        overall_score: Overall confidence score (0-100%)
        model_score: Score from model predictions (0-100)
        feature_score: Score from feature quality (0-100)
        historical_score: Score from historical accuracy (0-100)
        variance_score: Score from prediction variance (0-100)
        metadata: Additional information about the calculation
    """
    overall_score: float
    model_score: float
    feature_score: float
    historical_score: float
    variance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'overall_score': self.overall_score,
            'model_score': self.model_score,
            'feature_score': self.feature_score,
            'historical_score': self.historical_score,
            'variance_score': self.variance_score,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """String representation"""
        return (f"Confidence Score: {self.overall_score:.1f}%\n"
                f"  Model: {self.model_score:.1f}%\n"
                f"  Features: {self.feature_score:.1f}%\n"
                f"  Historical: {self.historical_score:.1f}%\n"
                f"  Variance: {self.variance_score:.1f}%")


class HistoricalPerformanceTracker:
    """
    Tracks historical prediction performance for adaptive confidence scoring.

    Maintains a rolling window of recent predictions and their outcomes
    to calculate accuracy metrics and adjust confidence scores.
    """

    def __init__(
        self,
        max_history: int = 1000,
        recent_window: int = 100,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize the performance tracker.

        Parameters:
        -----------
        max_history : int, default=1000
            Maximum number of historical records to keep
        recent_window : int, default=100
            Window size for recent performance calculation
        persistence_path : str, optional
            Path to save/load historical data
        """
        self.max_history = max_history
        self.recent_window = recent_window
        self.persistence_path = persistence_path

        # Historical records
        self.predictions = deque(maxlen=max_history)
        self.actuals = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.accuracies = deque(maxlen=max_history)

        # Load persisted data if available
        if persistence_path and Path(persistence_path).exists():
            self.load()

    def add_prediction(
        self,
        prediction: Union[float, int],
        actual: Optional[Union[float, int]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a prediction to history.

        Parameters:
        -----------
        prediction : float or int
            Predicted value
        actual : float or int, optional
            Actual observed value (if available)
        timestamp : datetime, optional
            Timestamp of prediction
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)

        # Calculate accuracy if actual is provided
        if actual is not None:
            # For classification: exact match
            # For regression: within threshold
            if isinstance(prediction, (int, np.integer)):
                accuracy = 1.0 if prediction == actual else 0.0
            else:
                # For continuous values, use relative error
                rel_error = abs(prediction - actual) / (abs(actual) + 1e-8)
                accuracy = max(0.0, 1.0 - rel_error)

            self.accuracies.append(accuracy)

    def get_overall_accuracy(self) -> float:
        """
        Calculate overall accuracy from all historical predictions.

        Returns:
        --------
        float
            Overall accuracy score (0-1)
        """
        if not self.accuracies:
            return 0.5  # Neutral score when no history

        return np.mean(list(self.accuracies))

    def get_recent_accuracy(self, window: Optional[int] = None) -> float:
        """
        Calculate accuracy from recent predictions.

        Parameters:
        -----------
        window : int, optional
            Number of recent predictions to consider

        Returns:
        --------
        float
            Recent accuracy score (0-1)
        """
        if not self.accuracies:
            return 0.5  # Neutral score when no history

        if window is None:
            window = self.recent_window

        recent_accuracies = list(self.accuracies)[-window:]
        return np.mean(recent_accuracies) if recent_accuracies else 0.5

    def get_accuracy_trend(self, window: int = 50) -> float:
        """
        Calculate trend in accuracy (improving or declining).

        Parameters:
        -----------
        window : int, default=50
            Window size for trend calculation

        Returns:
        --------
        float
            Trend coefficient (-1 to 1, positive means improving)
        """
        if len(self.accuracies) < window:
            return 0.0  # Neutral when insufficient data

        recent_accuracies = list(self.accuracies)[-window:]
        x = np.arange(len(recent_accuracies))

        # Calculate linear regression slope
        slope = np.polyfit(x, recent_accuracies, 1)[0]

        # Normalize to -1 to 1 range
        normalized_slope = np.clip(slope * 100, -1.0, 1.0)

        return normalized_slope

    def get_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive performance statistics.

        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        stats = {
            'total_predictions': len(self.predictions),
            'overall_accuracy': self.get_overall_accuracy(),
            'recent_accuracy': self.get_recent_accuracy(),
            'accuracy_trend': self.get_accuracy_trend(),
            'predictions_with_actuals': len([a for a in self.actuals if a is not None])
        }

        if self.accuracies:
            accuracies_list = list(self.accuracies)
            stats.update({
                'accuracy_std': np.std(accuracies_list),
                'accuracy_min': np.min(accuracies_list),
                'accuracy_max': np.max(accuracies_list)
            })

        return stats

    def save(self):
        """Save historical data to disk"""
        if not self.persistence_path:
            logger.warning("No persistence path set. Cannot save.")
            return

        data = {
            'predictions': list(self.predictions),
            'actuals': list(self.actuals),
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'accuracies': list(self.accuracies),
            'max_history': self.max_history,
            'recent_window': self.recent_window
        }

        Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Historical data saved to {self.persistence_path}")

    def load(self):
        """Load historical data from disk"""
        if not self.persistence_path or not Path(self.persistence_path).exists():
            logger.warning("Cannot load: file does not exist")
            return

        with open(self.persistence_path, 'r') as f:
            data = json.load(f)

        self.predictions = deque(data['predictions'], maxlen=self.max_history)
        self.actuals = deque(data['actuals'], maxlen=self.max_history)
        self.timestamps = deque(
            [datetime.fromisoformat(ts) for ts in data['timestamps']],
            maxlen=self.max_history
        )
        self.accuracies = deque(data['accuracies'], maxlen=self.max_history)

        logger.info(f"Historical data loaded from {self.persistence_path}")


class ConfidenceCalculator:
    """
    Main confidence score calculator combining multiple factors.

    This class orchestrates confidence scoring by evaluating:
    - Model predictions (probability scores, consistency)
    - Feature quality (completeness, validity, statistics)
    - Historical accuracy (past performance, trends)
    - Prediction variance (uncertainty, ensemble disagreement)
    """

    def __init__(
        self,
        weights: Optional[ConfidenceWeights] = None,
        enable_history_tracking: bool = True,
        history_persistence_path: Optional[str] = None,
        feature_quality_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the confidence calculator.

        Parameters:
        -----------
        weights : ConfidenceWeights, optional
            Custom weights for confidence components
        enable_history_tracking : bool, default=True
            Whether to track historical performance
        history_persistence_path : str, optional
            Path to save/load historical performance data
        feature_quality_thresholds : Dict[str, float], optional
            Custom thresholds for feature quality assessment
        """
        self.weights = weights or ConfidenceWeights()
        self.enable_history_tracking = enable_history_tracking

        # Historical performance tracker
        self.performance_tracker = None
        if enable_history_tracking:
            self.performance_tracker = HistoricalPerformanceTracker(
                persistence_path=history_persistence_path
            )

        # Feature quality thresholds
        self.feature_thresholds = feature_quality_thresholds or {
            'missing_threshold': 0.05,      # 5% missing values acceptable
            'outlier_threshold': 3.0,       # 3 standard deviations
            'min_variance': 1e-6,           # Minimum variance for features
            'correlation_threshold': 0.99   # Max correlation between features
        }

        logger.info("ConfidenceCalculator initialized")

    def calculate_model_score(
        self,
        predictions: Union[np.ndarray, List[float]],
        probabilities: Optional[np.ndarray] = None,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
        model_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate confidence score from model predictions.

        Parameters:
        -----------
        predictions : array-like
            Model predictions
        probabilities : np.ndarray, optional
            Prediction probabilities (for classification)
        ensemble_predictions : List[np.ndarray], optional
            Predictions from multiple models in ensemble
        model_metrics : Dict[str, float], optional
            Model evaluation metrics (R2, accuracy, etc.)

        Returns:
        --------
        float
            Model confidence score (0-100)
        """
        scores = []

        # 1. Probability score (for classification)
        if probabilities is not None:
            # Higher max probability = higher confidence
            max_probs = np.max(probabilities, axis=-1) if probabilities.ndim > 1 else probabilities
            prob_score = np.mean(max_probs) * 100
            scores.append(prob_score)

        # 2. Ensemble agreement (if multiple predictions available)
        if ensemble_predictions is not None and len(ensemble_predictions) > 1:
            # Calculate variance/disagreement across ensemble
            ensemble_array = np.array(ensemble_predictions)

            if ensemble_array.ndim == 1:
                # Single value predictions
                agreement = 1.0 - (np.std(ensemble_array) / (np.mean(np.abs(ensemble_array)) + 1e-8))
            else:
                # Multi-class predictions - measure agreement
                prediction_mode = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=ensemble_array
                )
                agreement = np.mean(ensemble_array == prediction_mode[None, :])

            agreement_score = np.clip(agreement * 100, 0, 100)
            scores.append(agreement_score)

        # 3. Model evaluation metrics
        if model_metrics is not None:
            # Use metrics like R2, accuracy, F1, etc.
            metric_scores = []

            if 'r2' in model_metrics:
                # R2 can be negative, so clip and scale
                r2_score = np.clip(model_metrics['r2'], 0, 1) * 100
                metric_scores.append(r2_score)

            if 'accuracy' in model_metrics:
                metric_scores.append(model_metrics['accuracy'] * 100)

            if 'f1' in model_metrics:
                metric_scores.append(model_metrics['f1'] * 100)

            if 'roc_auc' in model_metrics:
                metric_scores.append(model_metrics['roc_auc'] * 100)

            if 'directional_accuracy' in model_metrics:
                metric_scores.append(model_metrics['directional_accuracy'] * 100)

            if metric_scores:
                scores.append(np.mean(metric_scores))

        # 4. Prediction consistency (low variance = high confidence)
        if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 1:
            pred_array = np.array(predictions)
            # Normalize variance to 0-100 scale
            variance = np.var(pred_array)
            mean_val = np.mean(np.abs(pred_array))
            if mean_val > 0:
                cv = np.sqrt(variance) / mean_val  # Coefficient of variation
                consistency_score = np.clip((1 - cv) * 100, 0, 100)
                scores.append(consistency_score)

        # Average all component scores
        if scores:
            return np.mean(scores)
        else:
            return 50.0  # Neutral score if no information

    def calculate_feature_quality_score(
        self,
        features: pd.DataFrame,
        feature_stats: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate confidence score from feature quality.

        Parameters:
        -----------
        features : pd.DataFrame
            Input features used for prediction
        feature_stats : Dict[str, Any], optional
            Pre-computed feature statistics

        Returns:
        --------
        float
            Feature quality score (0-100)
        """
        scores = []

        # 1. Completeness score (missing values)
        missing_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        completeness_score = np.clip(
            (1 - missing_ratio / self.feature_thresholds['missing_threshold']) * 100,
            0, 100
        )
        scores.append(completeness_score)

        # 2. Validity score (outliers and ranges)
        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            # Calculate z-scores for outlier detection
            z_scores = np.abs((numeric_features - numeric_features.mean()) /
                             (numeric_features.std() + 1e-8))
            outlier_ratio = (z_scores > self.feature_thresholds['outlier_threshold']).sum().sum() / \
                           (z_scores.shape[0] * z_scores.shape[1])

            validity_score = np.clip((1 - outlier_ratio) * 100, 0, 100)
            scores.append(validity_score)

        # 3. Feature variance (features should have meaningful variation)
        if not numeric_features.empty:
            variances = numeric_features.var()
            low_variance_ratio = (variances < self.feature_thresholds['min_variance']).sum() / len(variances)
            variance_score = np.clip((1 - low_variance_ratio) * 100, 0, 100)
            scores.append(variance_score)

        # 4. Feature correlation (avoid highly correlated features)
        if not numeric_features.empty and numeric_features.shape[1] > 1:
            corr_matrix = numeric_features.corr().abs()
            # Exclude diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            high_corr_ratio = (corr_matrix > self.feature_thresholds['correlation_threshold']).sum().sum() / \
                             (corr_matrix.shape[0] * corr_matrix.shape[1])
            correlation_score = np.clip((1 - high_corr_ratio) * 100, 0, 100)
            scores.append(correlation_score)

        # 5. Data freshness (if timestamps available)
        if 'timestamp' in features.columns or isinstance(features.index, pd.DatetimeIndex):
            try:
                if isinstance(features.index, pd.DatetimeIndex):
                    latest_time = features.index.max()
                else:
                    latest_time = pd.to_datetime(features['timestamp']).max()

                time_diff = (datetime.now() - latest_time).total_seconds() / 3600  # Hours
                # Score decreases as data gets older (exponential decay)
                freshness_score = np.clip(100 * np.exp(-time_diff / 24), 0, 100)
                scores.append(freshness_score)
            except:
                pass  # Skip if timestamp parsing fails

        # Average all component scores
        if scores:
            return np.mean(scores)
        else:
            return 50.0  # Neutral score if no features to evaluate

    def calculate_historical_accuracy_score(
        self,
        recent_window: int = 100
    ) -> float:
        """
        Calculate confidence score from historical accuracy.

        Parameters:
        -----------
        recent_window : int, default=100
            Number of recent predictions to consider

        Returns:
        --------
        float
            Historical accuracy score (0-100)
        """
        if not self.enable_history_tracking or self.performance_tracker is None:
            return 50.0  # Neutral score when history tracking disabled

        scores = []

        # 1. Overall historical accuracy
        overall_accuracy = self.performance_tracker.get_overall_accuracy()
        scores.append(overall_accuracy * 100)

        # 2. Recent accuracy (weighted more heavily)
        recent_accuracy = self.performance_tracker.get_recent_accuracy(recent_window)
        scores.append(recent_accuracy * 100)
        scores.append(recent_accuracy * 100)  # Add twice for more weight

        # 3. Accuracy trend (improving = bonus, declining = penalty)
        trend = self.performance_tracker.get_accuracy_trend()
        trend_bonus = trend * 10  # +/- 10 points based on trend

        # Average scores and apply trend bonus
        base_score = np.mean(scores)
        final_score = np.clip(base_score + trend_bonus, 0, 100)

        return final_score

    def calculate_variance_score(
        self,
        predictions: Union[np.ndarray, List[float]],
        prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        ensemble_predictions: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Calculate confidence score from prediction variance/uncertainty.

        Parameters:
        -----------
        predictions : array-like
            Model predictions
        prediction_intervals : Tuple[np.ndarray, np.ndarray], optional
            Lower and upper bounds of prediction intervals
        ensemble_predictions : List[np.ndarray], optional
            Predictions from multiple models

        Returns:
        --------
        float
            Variance score (0-100), higher = lower uncertainty
        """
        scores = []

        # 1. Prediction interval width (narrower = higher confidence)
        if prediction_intervals is not None:
            lower, upper = prediction_intervals
            interval_width = np.mean(upper - lower)
            mean_prediction = np.mean(predictions)

            if abs(mean_prediction) > 0:
                relative_width = interval_width / abs(mean_prediction)
                # Narrower intervals get higher scores
                interval_score = np.clip((1 - relative_width) * 100, 0, 100)
                scores.append(interval_score)

        # 2. Ensemble prediction variance
        if ensemble_predictions is not None and len(ensemble_predictions) > 1:
            ensemble_array = np.array(ensemble_predictions)
            ensemble_std = np.std(ensemble_array, axis=0)
            ensemble_mean = np.mean(np.abs(ensemble_array), axis=0)

            # Calculate coefficient of variation
            cv = np.mean(ensemble_std / (ensemble_mean + 1e-8))
            variance_score = np.clip((1 - cv) * 100, 0, 100)
            scores.append(variance_score)

        # 3. Prediction stability (if multiple predictions available)
        if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 1:
            pred_array = np.array(predictions)
            std = np.std(pred_array)
            mean = np.mean(np.abs(pred_array))

            if mean > 0:
                stability = 1 - (std / mean)
                stability_score = np.clip(stability * 100, 0, 100)
                scores.append(stability_score)

        # Average all component scores
        if scores:
            return np.mean(scores)
        else:
            return 75.0  # Slightly optimistic default if no variance info

    def calculate_confidence(
        self,
        predictions: Union[np.ndarray, List[float]],
        features: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
        prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        actual_values: Optional[Union[np.ndarray, List[float]]] = None
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score combining all factors.

        Parameters:
        -----------
        predictions : array-like
            Model predictions
        features : pd.DataFrame
            Input features used for prediction
        probabilities : np.ndarray, optional
            Prediction probabilities (for classification)
        ensemble_predictions : List[np.ndarray], optional
            Predictions from multiple models
        prediction_intervals : Tuple[np.ndarray, np.ndarray], optional
            Prediction intervals (lower, upper bounds)
        model_metrics : Dict[str, float], optional
            Model evaluation metrics
        actual_values : array-like, optional
            Actual observed values (for updating history)

        Returns:
        --------
        ConfidenceScore
            Comprehensive confidence score with breakdown
        """
        # Calculate component scores
        model_score = self.calculate_model_score(
            predictions=predictions,
            probabilities=probabilities,
            ensemble_predictions=ensemble_predictions,
            model_metrics=model_metrics
        )

        feature_score = self.calculate_feature_quality_score(features=features)

        historical_score = self.calculate_historical_accuracy_score()

        variance_score = self.calculate_variance_score(
            predictions=predictions,
            prediction_intervals=prediction_intervals,
            ensemble_predictions=ensemble_predictions
        )

        # Calculate weighted overall score
        overall_score = (
            self.weights.model_prediction * model_score +
            self.weights.feature_quality * feature_score +
            self.weights.historical_accuracy * historical_score +
            self.weights.prediction_variance * variance_score
        )

        # Update historical tracking
        if self.enable_history_tracking and self.performance_tracker is not None:
            # Add predictions to history
            pred_array = np.array(predictions)
            actual_array = np.array(actual_values) if actual_values is not None else None

            for i, pred in enumerate(pred_array):
                actual = actual_array[i] if actual_array is not None and i < len(actual_array) else None
                self.performance_tracker.add_prediction(pred, actual)

        # Collect metadata
        metadata = {
            'weights': {
                'model_prediction': self.weights.model_prediction,
                'feature_quality': self.weights.feature_quality,
                'historical_accuracy': self.weights.historical_accuracy,
                'prediction_variance': self.weights.prediction_variance
            },
            'timestamp': datetime.now().isoformat(),
            'n_predictions': len(predictions),
            'n_features': len(features.columns)
        }

        if self.enable_history_tracking and self.performance_tracker is not None:
            metadata['historical_stats'] = self.performance_tracker.get_statistics()

        return ConfidenceScore(
            overall_score=overall_score,
            model_score=model_score,
            feature_score=feature_score,
            historical_score=historical_score,
            variance_score=variance_score,
            metadata=metadata
        )

    def get_confidence_level(self, score: float) -> str:
        """
        Convert numeric confidence score to descriptive level.

        Parameters:
        -----------
        score : float
            Confidence score (0-100)

        Returns:
        --------
        str
            Confidence level description
        """
        if score >= 90:
            return "Very High"
        elif score >= 75:
            return "High"
        elif score >= 60:
            return "Moderate"
        elif score >= 40:
            return "Low"
        else:
            return "Very Low"

    def save_performance_history(self):
        """Save performance tracking history to disk"""
        if self.enable_history_tracking and self.performance_tracker is not None:
            self.performance_tracker.save()

    def load_performance_history(self):
        """Load performance tracking history from disk"""
        if self.enable_history_tracking and self.performance_tracker is not None:
            self.performance_tracker.load()


# Utility functions

def create_confidence_calculator(
    model_weight: float = 0.35,
    feature_weight: float = 0.25,
    historical_weight: float = 0.30,
    variance_weight: float = 0.10,
    enable_history: bool = True,
    history_path: Optional[str] = None
) -> ConfidenceCalculator:
    """
    Create a confidence calculator with custom weights.

    Parameters:
    -----------
    model_weight : float, default=0.35
        Weight for model prediction component
    feature_weight : float, default=0.25
        Weight for feature quality component
    historical_weight : float, default=0.30
        Weight for historical accuracy component
    variance_weight : float, default=0.10
        Weight for prediction variance component
    enable_history : bool, default=True
        Whether to enable historical tracking
    history_path : str, optional
        Path to save/load history

    Returns:
    --------
    ConfidenceCalculator
        Configured confidence calculator
    """
    weights = ConfidenceWeights(
        model_prediction=model_weight,
        feature_quality=feature_weight,
        historical_accuracy=historical_weight,
        prediction_variance=variance_weight
    )

    return ConfidenceCalculator(
        weights=weights,
        enable_history_tracking=enable_history,
        history_persistence_path=history_path
    )


def quick_confidence_check(
    predictions: Union[np.ndarray, List[float]],
    features: pd.DataFrame,
    probabilities: Optional[np.ndarray] = None
) -> float:
    """
    Quick confidence check without full tracking.

    Parameters:
    -----------
    predictions : array-like
        Model predictions
    features : pd.DataFrame
        Input features
    probabilities : np.ndarray, optional
        Prediction probabilities

    Returns:
    --------
    float
        Confidence score (0-100)
    """
    calculator = ConfidenceCalculator(enable_history_tracking=False)
    confidence = calculator.calculate_confidence(
        predictions=predictions,
        features=features,
        probabilities=probabilities
    )
    return confidence.overall_score


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Confidence Score Calculator - BIST AI Trading System")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)

    # Sample predictions
    predictions = np.array([105.2, 106.1, 105.8])
    probabilities = np.array([[0.1, 0.7, 0.2], [0.15, 0.75, 0.1], [0.12, 0.73, 0.15]])

    # Ensemble predictions from multiple models
    ensemble_preds = [
        np.array([105.0, 106.0, 105.5]),
        np.array([105.5, 106.2, 106.0]),
        np.array([105.3, 106.1, 105.9])
    ]

    # Sample features
    features = pd.DataFrame({
        'price': [100, 101, 102],
        'volume': [1000000, 1100000, 1050000],
        'rsi': [65, 70, 68],
        'macd': [0.5, 0.6, 0.55],
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='D')
    })

    # Model metrics
    model_metrics = {
        'r2': 0.85,
        'rmse': 2.3,
        'mae': 1.8,
        'directional_accuracy': 0.78
    }

    # Initialize calculator
    print("\n1. Creating confidence calculator...")
    calculator = create_confidence_calculator(
        enable_history=True,
        history_path='data/confidence_history.json'
    )

    # Calculate confidence
    print("\n2. Calculating confidence score...")
    confidence = calculator.calculate_confidence(
        predictions=predictions,
        features=features,
        probabilities=probabilities,
        ensemble_predictions=ensemble_preds,
        model_metrics=model_metrics
    )

    # Display results
    print("\n3. Confidence Score Results:")
    print("=" * 80)
    print(confidence)
    print("=" * 80)

    print(f"\nConfidence Level: {calculator.get_confidence_level(confidence.overall_score)}")

    # Show metadata
    print("\n4. Metadata:")
    print(json.dumps(confidence.metadata, indent=2))

    # Simulate some predictions with actuals for history tracking
    print("\n5. Simulating historical predictions...")
    for i in range(10):
        pred = 100 + np.random.randn() * 5
        actual = pred + np.random.randn() * 2
        calculator.performance_tracker.add_prediction(pred, actual)

    # Recalculate with history
    print("\n6. Recalculating with historical data...")
    confidence_with_history = calculator.calculate_confidence(
        predictions=predictions,
        features=features,
        probabilities=probabilities,
        ensemble_predictions=ensemble_preds,
        model_metrics=model_metrics
    )

    print("\nUpdated Confidence Score:")
    print(confidence_with_history)

    # Show performance statistics
    print("\n7. Historical Performance Statistics:")
    stats = calculator.performance_tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Test quick confidence check
    print("\n8. Quick Confidence Check:")
    quick_score = quick_confidence_check(predictions, features, probabilities)
    print(f"Quick Score: {quick_score:.1f}%")

    print("\n" + "=" * 80)
    print("Confidence calculator example completed successfully!")
    print("=" * 80)
