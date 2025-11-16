"""
Trading Signal Generator for BIST AI Trading System

This module generates final trading signals by combining outputs from multiple
models (regression, classification, NLP) using sophisticated aggregation logic,
ensemble methods, and dynamic threshold determination.

The signal generator implements:
- Multi-model ensemble aggregation
- Weighted voting with confidence scoring
- Dynamic threshold adjustment based on market conditions
- Risk-adjusted signal generation
- Position sizing recommendations
- Signal quality assessment
- Multi-timeframe analysis integration

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration for trading signal types"""
    STRONG_BUY = 4
    BUY = 3
    HOLD = 2
    SELL = 1
    STRONG_SELL = 0


class SignalConfidence(Enum):
    """Enumeration for signal confidence levels"""
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1


@dataclass
class ModelOutput:
    """
    Container for individual model outputs.

    Attributes:
    -----------
    model_name : str
        Name of the model (e.g., 'lstm', 'random_forest', 'ann', 'nlp_sentiment')
    model_type : str
        Type of model ('regression', 'classification', 'nlp')
    prediction : Union[float, int, str]
        Raw prediction from the model
    confidence : Optional[float]
        Confidence score (0-1) if available
    probabilities : Optional[np.ndarray]
        Class probabilities for classification models
    metadata : Dict[str, Any]
        Additional metadata from the model
    """
    model_name: str
    model_type: str
    prediction: Union[float, int, str]
    confidence: Optional[float] = None
    probabilities: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """
    Final trading signal with comprehensive information.

    Attributes:
    -----------
    signal : SignalType
        Final trading signal (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
    confidence : SignalConfidence
        Confidence level of the signal
    confidence_score : float
        Numerical confidence score (0-1)
    timestamp : datetime
        When the signal was generated
    stock_code : str
        Stock symbol (e.g., 'THYAO')
    current_price : Optional[float]
        Current stock price
    target_price : Optional[float]
        Target price from regression models
    expected_return : Optional[float]
        Expected return percentage
    position_size : Optional[float]
        Recommended position size (0-1, fraction of portfolio)
    risk_score : Optional[float]
        Risk score (0-1, higher = more risky)
    stop_loss : Optional[float]
        Suggested stop loss price
    take_profit : Optional[float]
        Suggested take profit price
    model_contributions : Dict[str, float]
        Contribution of each model to the final signal
    rationale : str
        Explanation of the signal
    metadata : Dict[str, Any]
        Additional information
    """
    signal: SignalType
    confidence: SignalConfidence
    confidence_score: float
    timestamp: datetime
    stock_code: str
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    expected_return: Optional[float] = None
    position_size: Optional[float] = None
    risk_score: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    model_contributions: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'signal': self.signal.name,
            'signal_value': self.signal.value,
            'confidence': self.confidence.name,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'stock_code': self.stock_code,
            'current_price': self.current_price,
            'target_price': self.target_price,
            'expected_return': self.expected_return,
            'position_size': self.position_size,
            'risk_score': self.risk_score,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'model_contributions': self.model_contributions,
            'rationale': self.rationale,
            'metadata': self.metadata
        }


class SignalGenerator:
    """
    Main signal generator that combines outputs from multiple models.

    This class implements ensemble methods to aggregate predictions from:
    - Regression models (LSTM, GRU, XGBoost, LightGBM) for price forecasting
    - Classification models (Random Forest, ANN) for signal classification
    - NLP models for sentiment analysis
    - Technical and fundamental indicators

    The generator uses weighted voting, confidence scoring, and dynamic
    thresholds to produce high-quality trading signals.
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        signal_thresholds: Optional[Dict[str, float]] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        enable_dynamic_thresholds: bool = True,
        risk_adjustment: bool = True,
        min_confidence: float = 0.3,
        volatility_window: int = 20,
        sentiment_weight: float = 0.15,
        regression_weight: float = 0.35,
        classification_weight: float = 0.35,
        technical_weight: float = 0.15
    ):
        """
        Initialize the Signal Generator.

        Parameters:
        -----------
        model_weights : Dict[str, float], optional
            Custom weights for each model (must sum to 1.0)
        signal_thresholds : Dict[str, float], optional
            Thresholds for signal classification
        confidence_thresholds : Dict[str, float], optional
            Thresholds for confidence levels
        enable_dynamic_thresholds : bool, default=True
            Whether to adjust thresholds based on market conditions
        risk_adjustment : bool, default=True
            Whether to adjust signals based on risk metrics
        min_confidence : float, default=0.3
            Minimum confidence score to generate non-HOLD signal
        volatility_window : int, default=20
            Window for volatility calculation
        sentiment_weight : float, default=0.15
            Weight for sentiment analysis
        regression_weight : float, default=0.35
            Weight for regression models
        classification_weight : float, default=0.35
            Weight for classification models
        technical_weight : float, default=0.15
            Weight for technical indicators
        """
        # Model weights
        self.model_weights = model_weights or {}
        self.sentiment_weight = sentiment_weight
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.technical_weight = technical_weight

        # Signal thresholds (for converting regression to signals)
        self.signal_thresholds = signal_thresholds or {
            'strong_buy': 0.03,      # 3% expected return
            'buy': 0.01,              # 1% expected return
            'sell': -0.01,            # -1% expected return
            'strong_sell': -0.03      # -3% expected return
        }

        # Confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'very_high': 0.8,
            'high': 0.65,
            'medium': 0.45,
            'low': 0.3,
            'very_low': 0.0
        }

        # Configuration
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        self.risk_adjustment = risk_adjustment
        self.min_confidence = min_confidence
        self.volatility_window = volatility_window

        # Market condition tracking
        self.market_volatility = None
        self.market_trend = None

        logger.info("SignalGenerator initialized with dynamic_thresholds={}, risk_adjustment={}".format(
            enable_dynamic_thresholds, risk_adjustment
        ))

    def generate_signal(
        self,
        stock_code: str,
        model_outputs: List[ModelOutput],
        current_price: Optional[float] = None,
        historical_prices: Optional[pd.Series] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """
        Generate a trading signal by aggregating outputs from multiple models.

        Parameters:
        -----------
        stock_code : str
            Stock symbol (e.g., 'THYAO')
        model_outputs : List[ModelOutput]
            List of outputs from different models
        current_price : float, optional
            Current stock price
        historical_prices : pd.Series, optional
            Historical price series for volatility calculation
        market_data : Dict[str, Any], optional
            Additional market data (volume, market cap, etc.)

        Returns:
        --------
        TradingSignal
            Final aggregated trading signal
        """
        if not model_outputs:
            raise ValueError("No model outputs provided")

        # Calculate market conditions if historical data is available
        if historical_prices is not None:
            self._update_market_conditions(historical_prices)

        # Adjust thresholds based on market conditions
        if self.enable_dynamic_thresholds:
            adjusted_thresholds = self._adjust_thresholds()
        else:
            adjusted_thresholds = self.signal_thresholds.copy()

        # Separate model outputs by type
        regression_outputs = [m for m in model_outputs if m.model_type == 'regression']
        classification_outputs = [m for m in model_outputs if m.model_type == 'classification']
        nlp_outputs = [m for m in model_outputs if m.model_type == 'nlp']

        # Aggregate signals from each model type
        regression_signal, regression_confidence = self._aggregate_regression_models(
            regression_outputs, current_price, adjusted_thresholds
        )

        classification_signal, classification_confidence = self._aggregate_classification_models(
            classification_outputs
        )

        sentiment_signal, sentiment_confidence = self._aggregate_sentiment(nlp_outputs)

        # Combine all signals using weighted voting
        final_signal_value, combined_confidence, contributions = self._weighted_voting(
            regression_signal=regression_signal,
            regression_confidence=regression_confidence,
            classification_signal=classification_signal,
            classification_confidence=classification_confidence,
            sentiment_signal=sentiment_signal,
            sentiment_confidence=sentiment_confidence
        )

        # Apply risk adjustment
        if self.risk_adjustment and historical_prices is not None:
            final_signal_value, combined_confidence, risk_score = self._apply_risk_adjustment(
                final_signal_value, combined_confidence, historical_prices
            )
        else:
            risk_score = None

        # Apply minimum confidence filter
        if combined_confidence < self.min_confidence:
            final_signal_value = SignalType.HOLD.value
            combined_confidence = max(combined_confidence, 0.1)

        # Convert to signal type
        final_signal = self._value_to_signal_type(final_signal_value)
        confidence_level = self._score_to_confidence_level(combined_confidence)

        # Calculate target price and expected return
        target_price = self._calculate_target_price(
            regression_outputs, current_price
        )

        expected_return = None
        if current_price and target_price:
            expected_return = (target_price - current_price) / current_price

        # Calculate position sizing
        position_size = self._calculate_position_size(
            final_signal, combined_confidence, risk_score
        )

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(
            current_price, final_signal, historical_prices
        )

        # Generate rationale
        rationale = self._generate_rationale(
            final_signal, contributions, regression_outputs,
            classification_outputs, nlp_outputs
        )

        # Create trading signal
        signal = TradingSignal(
            signal=final_signal,
            confidence=confidence_level,
            confidence_score=combined_confidence,
            timestamp=datetime.now(),
            stock_code=stock_code,
            current_price=current_price,
            target_price=target_price,
            expected_return=expected_return,
            position_size=position_size,
            risk_score=risk_score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_contributions=contributions,
            rationale=rationale,
            metadata={
                'market_volatility': self.market_volatility,
                'market_trend': self.market_trend,
                'adjusted_thresholds': adjusted_thresholds,
                'num_models': len(model_outputs)
            }
        )

        return signal

    def _aggregate_regression_models(
        self,
        outputs: List[ModelOutput],
        current_price: Optional[float],
        thresholds: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Aggregate regression model outputs to signal.

        Returns:
        --------
        Tuple[float, float]
            (signal_value, confidence)
        """
        if not outputs:
            return SignalType.HOLD.value, 0.0

        # Calculate expected returns from price predictions
        returns = []
        confidences = []

        for output in outputs:
            if current_price and isinstance(output.prediction, (int, float)):
                expected_return = (output.prediction - current_price) / current_price
                returns.append(expected_return)

                # Use model confidence if available, otherwise use default
                model_confidence = output.confidence if output.confidence is not None else 0.5
                confidences.append(model_confidence)

        if not returns:
            return SignalType.HOLD.value, 0.0

        # Weighted average of returns
        if confidences:
            avg_return = np.average(returns, weights=confidences)
            avg_confidence = np.mean(confidences)
        else:
            avg_return = np.mean(returns)
            avg_confidence = 0.5

        # Convert return to signal
        if avg_return >= thresholds['strong_buy']:
            signal = SignalType.STRONG_BUY.value
        elif avg_return >= thresholds['buy']:
            signal = SignalType.BUY.value
        elif avg_return <= thresholds['strong_sell']:
            signal = SignalType.STRONG_SELL.value
        elif avg_return <= thresholds['sell']:
            signal = SignalType.SELL.value
        else:
            signal = SignalType.HOLD.value

        # Adjust confidence based on agreement
        returns_std = np.std(returns) if len(returns) > 1 else 0
        agreement_factor = 1.0 - min(returns_std / 0.1, 1.0)  # Normalize by 10% std
        final_confidence = avg_confidence * agreement_factor

        return signal, final_confidence

    def _aggregate_classification_models(
        self,
        outputs: List[ModelOutput]
    ) -> Tuple[float, float]:
        """
        Aggregate classification model outputs.

        Returns:
        --------
        Tuple[float, float]
            (signal_value, confidence)
        """
        if not outputs:
            return SignalType.HOLD.value, 0.0

        signals = []
        confidences = []

        for output in outputs:
            # Convert prediction to signal value
            if isinstance(output.prediction, (int, np.integer)):
                signal_value = int(output.prediction)
            elif isinstance(output.prediction, str):
                signal_value = self._signal_name_to_value(output.prediction)
            else:
                continue

            signals.append(signal_value)

            # Extract confidence from probabilities or use provided confidence
            if output.probabilities is not None:
                # Confidence is the max probability
                confidence = np.max(output.probabilities)
            elif output.confidence is not None:
                confidence = output.confidence
            else:
                confidence = 0.5

            confidences.append(confidence)

        if not signals:
            return SignalType.HOLD.value, 0.0

        # Weighted voting
        if confidences:
            signal_value = np.average(signals, weights=confidences)
            avg_confidence = np.mean(confidences)
        else:
            signal_value = np.mean(signals)
            avg_confidence = 0.5

        # Agreement factor
        signals_std = np.std(signals) if len(signals) > 1 else 0
        agreement_factor = 1.0 - min(signals_std / 2.0, 1.0)  # Normalize by signal range
        final_confidence = avg_confidence * agreement_factor

        return signal_value, final_confidence

    def _aggregate_sentiment(
        self,
        outputs: List[ModelOutput]
    ) -> Tuple[float, float]:
        """
        Aggregate sentiment analysis outputs.

        Returns:
        --------
        Tuple[float, float]
            (signal_value, confidence)
        """
        if not outputs:
            return SignalType.HOLD.value, 0.0

        sentiments = []
        confidences = []

        for output in outputs:
            # Sentiment should be a float between -1 (negative) and 1 (positive)
            if isinstance(output.prediction, (int, float)):
                sentiment = float(output.prediction)
                sentiments.append(sentiment)

                confidence = output.confidence if output.confidence is not None else 0.5
                confidences.append(confidence)

        if not sentiments:
            return SignalType.HOLD.value, 0.0

        # Weighted average sentiment
        avg_sentiment = np.average(sentiments, weights=confidences) if confidences else np.mean(sentiments)
        avg_confidence = np.mean(confidences) if confidences else 0.5

        # Convert sentiment to signal (sentiment is -1 to 1, signal is 0 to 4)
        # Map: -1 -> 0 (STRONG_SELL), 0 -> 2 (HOLD), 1 -> 4 (STRONG_BUY)
        signal_value = (avg_sentiment + 1) * 2  # Maps -1,1 to 0,4

        return signal_value, avg_confidence

    def _weighted_voting(
        self,
        regression_signal: float,
        regression_confidence: float,
        classification_signal: float,
        classification_confidence: float,
        sentiment_signal: float,
        sentiment_confidence: float
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Combine signals using weighted voting.

        Returns:
        --------
        Tuple[float, float, Dict[str, float]]
            (final_signal, combined_confidence, contributions)
        """
        # Calculate weighted signals
        weighted_signals = []
        weights = []
        contributions = {}

        if regression_confidence > 0:
            weight = self.regression_weight * regression_confidence
            weighted_signals.append(regression_signal * weight)
            weights.append(weight)
            contributions['regression'] = weight

        if classification_confidence > 0:
            weight = self.classification_weight * classification_confidence
            weighted_signals.append(classification_signal * weight)
            weights.append(weight)
            contributions['classification'] = weight

        if sentiment_confidence > 0:
            weight = self.sentiment_weight * sentiment_confidence
            weighted_signals.append(sentiment_signal * weight)
            weights.append(weight)
            contributions['sentiment'] = weight

        if not weighted_signals:
            return SignalType.HOLD.value, 0.0, {}

        # Normalize contributions
        total_weight = sum(contributions.values())
        if total_weight > 0:
            contributions = {k: v / total_weight for k, v in contributions.items()}

        # Calculate final signal
        final_signal = sum(weighted_signals) / sum(weights)

        # Combined confidence (weighted average of confidences)
        combined_confidence = (
            regression_confidence * self.regression_weight +
            classification_confidence * self.classification_weight +
            sentiment_confidence * self.sentiment_weight
        ) / (self.regression_weight + self.classification_weight + self.sentiment_weight)

        return final_signal, combined_confidence, contributions

    def _apply_risk_adjustment(
        self,
        signal_value: float,
        confidence: float,
        historical_prices: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Apply risk-based adjustment to signal and confidence.

        Returns:
        --------
        Tuple[float, float, float]
            (adjusted_signal, adjusted_confidence, risk_score)
        """
        # Calculate volatility
        returns = historical_prices.pct_change().dropna()
        volatility = returns.std()

        # Calculate risk score (0-1, higher = riskier)
        # Normalize volatility to risk score
        risk_score = min(volatility / 0.05, 1.0)  # 5% daily vol = max risk

        # Reduce confidence in high volatility
        volatility_factor = 1.0 - (risk_score * 0.3)  # Max 30% reduction
        adjusted_confidence = confidence * volatility_factor

        # Moderate extreme signals in high volatility
        if risk_score > 0.7:  # High risk
            # Pull signal toward HOLD (value 2)
            signal_distance = signal_value - SignalType.HOLD.value
            adjusted_signal = signal_value - (signal_distance * 0.3)
        else:
            adjusted_signal = signal_value

        return adjusted_signal, adjusted_confidence, risk_score

    def _adjust_thresholds(self) -> Dict[str, float]:
        """
        Adjust signal thresholds based on market conditions.

        Returns:
        --------
        Dict[str, float]
            Adjusted thresholds
        """
        adjusted = self.signal_thresholds.copy()

        # Adjust based on volatility
        if self.market_volatility is not None:
            # In high volatility, require higher returns for buy signals
            vol_factor = 1.0 + (self.market_volatility / 0.02)  # Scale by 2% volatility

            adjusted['strong_buy'] *= vol_factor
            adjusted['buy'] *= vol_factor
            adjusted['sell'] *= vol_factor
            adjusted['strong_sell'] *= vol_factor

        # Adjust based on market trend
        if self.market_trend is not None:
            # In downtrend, be more conservative
            if self.market_trend < 0:
                trend_factor = 1.0 + abs(self.market_trend)
                adjusted['strong_buy'] *= trend_factor
                adjusted['buy'] *= trend_factor

        return adjusted

    def _update_market_conditions(self, historical_prices: pd.Series):
        """Update market volatility and trend from historical data."""
        if len(historical_prices) < self.volatility_window:
            return

        # Calculate volatility
        returns = historical_prices.pct_change().dropna()
        self.market_volatility = returns.tail(self.volatility_window).std()

        # Calculate trend (slope of linear regression)
        recent_prices = historical_prices.tail(self.volatility_window).values
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        self.market_trend = slope / recent_prices[-1]  # Normalize by current price

    def _calculate_target_price(
        self,
        regression_outputs: List[ModelOutput],
        current_price: Optional[float]
    ) -> Optional[float]:
        """Calculate target price from regression models."""
        if not regression_outputs:
            return None

        predictions = []
        confidences = []

        for output in regression_outputs:
            if isinstance(output.prediction, (int, float)):
                predictions.append(float(output.prediction))
                confidences.append(output.confidence if output.confidence else 0.5)

        if not predictions:
            return None

        # Weighted average
        target = np.average(predictions, weights=confidences) if confidences else np.mean(predictions)
        return float(target)

    def _calculate_position_size(
        self,
        signal: SignalType,
        confidence: float,
        risk_score: Optional[float]
    ) -> float:
        """
        Calculate recommended position size.

        Returns:
        --------
        float
            Position size as fraction of portfolio (0-1)
        """
        # Base position size on signal strength and confidence
        if signal == SignalType.STRONG_BUY:
            base_size = 0.15 * confidence
        elif signal == SignalType.BUY:
            base_size = 0.10 * confidence
        elif signal == SignalType.STRONG_SELL or signal == SignalType.SELL:
            base_size = 0.0  # Don't recommend short positions
        else:  # HOLD
            base_size = 0.0

        # Adjust for risk
        if risk_score is not None:
            # Reduce position size in high risk
            risk_factor = 1.0 - (risk_score * 0.5)
            base_size *= risk_factor

        return min(max(base_size, 0.0), 0.2)  # Cap at 20% of portfolio

    def _calculate_risk_levels(
        self,
        current_price: Optional[float],
        signal: SignalType,
        historical_prices: Optional[pd.Series]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels.

        Returns:
        --------
        Tuple[Optional[float], Optional[float]]
            (stop_loss, take_profit)
        """
        if not current_price:
            return None, None

        # Calculate ATR-based levels if historical data available
        if historical_prices is not None and len(historical_prices) > 14:
            returns = historical_prices.pct_change().dropna()
            atr = returns.tail(14).std() * current_price

            # Stop loss: 2 ATR below (for buys)
            # Take profit: 3 ATR above (for buys)
            if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
            elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
            else:
                stop_loss = None
                take_profit = None
        else:
            # Fixed percentage levels
            if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * 0.95  # 5% stop loss
                take_profit = current_price * 1.10  # 10% take profit
            elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = current_price * 1.05
                take_profit = current_price * 0.90
            else:
                stop_loss = None
                take_profit = None

        return stop_loss, take_profit

    def _generate_rationale(
        self,
        signal: SignalType,
        contributions: Dict[str, float],
        regression_outputs: List[ModelOutput],
        classification_outputs: List[ModelOutput],
        nlp_outputs: List[ModelOutput]
    ) -> str:
        """Generate human-readable rationale for the signal."""
        rationale_parts = [f"{signal.name} signal generated."]

        # Model contributions
        if contributions:
            contrib_str = ", ".join([
                f"{k}: {v*100:.1f}%" for k, v in sorted(
                    contributions.items(), key=lambda x: x[1], reverse=True
                )
            ])
            rationale_parts.append(f"Model contributions: {contrib_str}.")

        # Model counts
        total_models = len(regression_outputs) + len(classification_outputs) + len(nlp_outputs)
        rationale_parts.append(f"Based on {total_models} models.")

        # Market conditions
        if self.market_volatility is not None:
            vol_pct = self.market_volatility * 100
            rationale_parts.append(f"Market volatility: {vol_pct:.2f}%.")

        return " ".join(rationale_parts)

    def _value_to_signal_type(self, value: float) -> SignalType:
        """Convert numerical signal value to SignalType."""
        rounded = round(value)
        rounded = max(0, min(4, rounded))  # Clamp to 0-4
        return SignalType(rounded)

    def _signal_name_to_value(self, name: str) -> int:
        """Convert signal name to numerical value."""
        name = name.upper().replace(' ', '_')
        mapping = {
            'STRONG_SELL': 0,
            'SELL': 1,
            'HOLD': 2,
            'BUY': 3,
            'STRONG_BUY': 4
        }
        return mapping.get(name, 2)  # Default to HOLD

    def _score_to_confidence_level(self, score: float) -> SignalConfidence:
        """Convert confidence score to confidence level."""
        if score >= self.confidence_thresholds['very_high']:
            return SignalConfidence.VERY_HIGH
        elif score >= self.confidence_thresholds['high']:
            return SignalConfidence.HIGH
        elif score >= self.confidence_thresholds['medium']:
            return SignalConfidence.MEDIUM
        elif score >= self.confidence_thresholds['low']:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW

    def generate_batch_signals(
        self,
        stocks_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, TradingSignal]:
        """
        Generate signals for multiple stocks.

        Parameters:
        -----------
        stocks_data : Dict[str, Dict[str, Any]]
            Dictionary mapping stock codes to their data
            Each stock's data should contain:
            - 'model_outputs': List[ModelOutput]
            - 'current_price': float (optional)
            - 'historical_prices': pd.Series (optional)
            - 'market_data': Dict[str, Any] (optional)

        Returns:
        --------
        Dict[str, TradingSignal]
            Dictionary mapping stock codes to their trading signals
        """
        signals = {}

        for stock_code, data in stocks_data.items():
            try:
                signal = self.generate_signal(
                    stock_code=stock_code,
                    model_outputs=data['model_outputs'],
                    current_price=data.get('current_price'),
                    historical_prices=data.get('historical_prices'),
                    market_data=data.get('market_data')
                )
                signals[stock_code] = signal

            except Exception as e:
                logger.error(f"Error generating signal for {stock_code}: {str(e)}")
                # Generate a HOLD signal with low confidence as fallback
                signals[stock_code] = TradingSignal(
                    signal=SignalType.HOLD,
                    confidence=SignalConfidence.VERY_LOW,
                    confidence_score=0.1,
                    timestamp=datetime.now(),
                    stock_code=stock_code,
                    rationale=f"Error in signal generation: {str(e)}"
                )

        return signals

    def update_model_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights.

        Parameters:
        -----------
        new_weights : Dict[str, float]
            New weights for models (should sum to ~1.0)
        """
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            warnings.warn(f"Model weights sum to {total}, not 1.0. Normalizing...")
            self.model_weights = {k: v/total for k, v in new_weights.items()}
        else:
            self.model_weights = new_weights

        logger.info(f"Model weights updated: {self.model_weights}")

    def update_thresholds(
        self,
        signal_thresholds: Optional[Dict[str, float]] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Update signal or confidence thresholds.

        Parameters:
        -----------
        signal_thresholds : Dict[str, float], optional
            New signal thresholds
        confidence_thresholds : Dict[str, float], optional
            New confidence thresholds
        """
        if signal_thresholds:
            self.signal_thresholds.update(signal_thresholds)
            logger.info(f"Signal thresholds updated: {self.signal_thresholds}")

        if confidence_thresholds:
            self.confidence_thresholds.update(confidence_thresholds)
            logger.info(f"Confidence thresholds updated: {self.confidence_thresholds}")


# Convenience functions
def create_signal_generator(
    **kwargs
) -> SignalGenerator:
    """
    Create a signal generator with default or custom parameters.

    Parameters:
    -----------
    **kwargs : dict
        Parameters for SignalGenerator

    Returns:
    --------
    SignalGenerator
        Initialized signal generator
    """
    return SignalGenerator(**kwargs)


def create_model_output(
    model_name: str,
    model_type: str,
    prediction: Union[float, int, str],
    confidence: Optional[float] = None,
    probabilities: Optional[np.ndarray] = None,
    **metadata
) -> ModelOutput:
    """
    Create a ModelOutput object.

    Parameters:
    -----------
    model_name : str
        Name of the model
    model_type : str
        Type of model ('regression', 'classification', 'nlp')
    prediction : Union[float, int, str]
        Model prediction
    confidence : float, optional
        Confidence score
    probabilities : np.ndarray, optional
        Class probabilities
    **metadata : dict
        Additional metadata

    Returns:
    --------
    ModelOutput
        ModelOutput object
    """
    return ModelOutput(
        model_name=model_name,
        model_type=model_type,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        metadata=metadata
    )


if __name__ == "__main__":
    # Example usage
    print("Signal Generator for BIST AI Trading System")
    print("=" * 60)

    # Create signal generator
    generator = create_signal_generator(
        enable_dynamic_thresholds=True,
        risk_adjustment=True,
        min_confidence=0.3
    )

    # Example model outputs
    model_outputs = [
        # LSTM price prediction
        create_model_output(
            model_name='lstm_price_forecaster',
            model_type='regression',
            prediction=105.5,  # Predicted price
            confidence=0.75
        ),
        # Random Forest classification
        create_model_output(
            model_name='random_forest_classifier',
            model_type='classification',
            prediction=3,  # BUY signal
            confidence=0.68,
            probabilities=np.array([0.05, 0.10, 0.17, 0.68, 0.00])
        ),
        # ANN classification
        create_model_output(
            model_name='ann_classifier',
            model_type='classification',
            prediction=3,  # BUY signal
            confidence=0.72,
            probabilities=np.array([0.03, 0.08, 0.17, 0.72, 0.00])
        ),
        # Sentiment analysis
        create_model_output(
            model_name='sentiment_analyzer',
            model_type='nlp',
            prediction=0.45,  # Positive sentiment
            confidence=0.60
        )
    ]

    # Generate signal
    current_price = 100.0
    historical_prices = pd.Series([95, 96, 97, 98, 99, 100] * 5)  # Sample data

    signal = generator.generate_signal(
        stock_code='THYAO',
        model_outputs=model_outputs,
        current_price=current_price,
        historical_prices=historical_prices
    )

    # Print results
    print("\nGenerated Signal:")
    print("-" * 60)
    print(f"Stock: {signal.stock_code}")
    print(f"Signal: {signal.signal.name}")
    print(f"Confidence: {signal.confidence.name} ({signal.confidence_score:.2%})")
    print(f"Current Price: ${signal.current_price:.2f}")
    print(f"Target Price: ${signal.target_price:.2f}")
    print(f"Expected Return: {signal.expected_return:.2%}")
    print(f"Position Size: {signal.position_size:.2%}")
    print(f"Risk Score: {signal.risk_score:.2f}" if signal.risk_score else "Risk Score: N/A")
    print(f"Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "Stop Loss: N/A")
    print(f"Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "Take Profit: N/A")
    print(f"\nModel Contributions:")
    for model, contrib in signal.model_contributions.items():
        print(f"  {model}: {contrib*100:.1f}%")
    print(f"\nRationale: {signal.rationale}")
