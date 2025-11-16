"""
Signal Prioritizer - Multi-Factor Ranking Algorithm

This module implements a sophisticated signal prioritization system that ranks
trading signals based on multiple factors including model confidence, whale
activity (WAI), news sentiment, and model agreement across different forecasting
models.

The prioritizer uses a weighted scoring algorithm to combine various signal
quality metrics into a single, actionable priority score. This helps traders
focus on the highest-probability signals with the strongest institutional backing
and positive sentiment.

Key Features:
- Multi-factor signal ranking (confidence, WAI, sentiment, model agreement)
- Configurable factor weights for custom prioritization strategies
- Model consensus calculation across multiple forecasting models
- Signal strength normalization and scaling
- Risk-adjusted scoring
- Batch signal processing
- Signal filtering and sorting capabilities
- Detailed signal metadata and component scores

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction enumeration"""
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2


class PrioritizationStrategy(Enum):
    """Prioritization strategy types"""
    BALANCED = "balanced"  # Equal weight to all factors
    CONFIDENCE_FOCUSED = "confidence_focused"  # Prioritize model confidence
    WHALE_FOCUSED = "whale_focused"  # Prioritize whale activity
    SENTIMENT_FOCUSED = "sentiment_focused"  # Prioritize news sentiment
    CONSENSUS_FOCUSED = "consensus_focused"  # Prioritize model agreement
    AGGRESSIVE = "aggressive"  # High risk, high reward
    CONSERVATIVE = "conservative"  # Low risk, proven patterns


@dataclass
class SignalInput:
    """
    Input data structure for a trading signal to be prioritized.

    Attributes:
    -----------
    symbol : str
        Stock symbol (e.g., 'THYAO', 'AKBNK')
    timestamp : datetime
        Signal generation timestamp
    signal_direction : SignalDirection
        Trading signal direction
    confidence_score : float
        Model confidence score (0-100)
    wai_score : float, optional
        Whale Activity Index score (0-100)
    news_sentiment : float, optional
        News sentiment score (-1 to +1)
    model_predictions : Dict[str, float], optional
        Individual model predictions {model_name: predicted_return}
    target_price : float, optional
        Predicted target price
    current_price : float, optional
        Current market price
    additional_metadata : Dict[str, Any], optional
        Additional signal metadata
    """
    symbol: str
    timestamp: datetime
    signal_direction: SignalDirection
    confidence_score: float
    wai_score: Optional[float] = None
    news_sentiment: Optional[float] = None
    model_predictions: Optional[Dict[str, float]] = None
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    additional_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate input data"""
        # Validate confidence score
        if not 0 <= self.confidence_score <= 100:
            raise ValueError(f"confidence_score must be between 0 and 100, got {self.confidence_score}")

        # Validate WAI score if provided
        if self.wai_score is not None and not 0 <= self.wai_score <= 100:
            raise ValueError(f"wai_score must be between 0 and 100, got {self.wai_score}")

        # Validate sentiment score if provided
        if self.news_sentiment is not None and not -1 <= self.news_sentiment <= 1:
            raise ValueError(f"news_sentiment must be between -1 and 1, got {self.news_sentiment}")

        # Convert string direction to enum if needed
        if isinstance(self.signal_direction, str):
            self.signal_direction = SignalDirection[self.signal_direction.upper().replace(' ', '_')]


@dataclass
class PrioritizedSignal:
    """
    Output data structure for a prioritized signal with ranking information.

    Attributes:
    -----------
    symbol : str
        Stock symbol
    timestamp : datetime
        Signal generation timestamp
    signal_direction : SignalDirection
        Trading signal direction
    priority_score : float
        Overall priority score (0-100)
    priority_rank : int
        Rank among all signals (1 = highest priority)
    confidence_component : float
        Contribution from confidence score
    wai_component : float
        Contribution from WAI score
    sentiment_component : float
        Contribution from news sentiment
    agreement_component : float
        Contribution from model agreement
    model_agreement_pct : float
        Percentage of models in agreement
    raw_scores : Dict[str, float]
        Original input scores
    risk_adjusted_score : float
        Risk-adjusted priority score
    signal_strength : str
        Categorical signal strength (WEAK, MODERATE, STRONG, VERY_STRONG)
    """
    symbol: str
    timestamp: datetime
    signal_direction: SignalDirection
    priority_score: float
    priority_rank: int
    confidence_component: float
    wai_component: float
    sentiment_component: float
    agreement_component: float
    model_agreement_pct: float
    raw_scores: Dict[str, float]
    risk_adjusted_score: float
    signal_strength: str
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    expected_return: Optional[float] = None
    additional_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['signal_direction'] = self.signal_direction.name
        return result


class SignalPrioritizer:
    """
    Multi-factor signal prioritization engine.

    This class implements a sophisticated ranking algorithm that combines
    multiple signal quality factors into a single priority score. The algorithm
    supports configurable weighting strategies and can adapt to different
    trading styles and risk preferences.
    """

    # Default factor weights for different strategies
    STRATEGY_WEIGHTS = {
        PrioritizationStrategy.BALANCED: {
            'confidence': 0.30,
            'wai': 0.25,
            'sentiment': 0.20,
            'agreement': 0.25
        },
        PrioritizationStrategy.CONFIDENCE_FOCUSED: {
            'confidence': 0.50,
            'wai': 0.20,
            'sentiment': 0.10,
            'agreement': 0.20
        },
        PrioritizationStrategy.WHALE_FOCUSED: {
            'confidence': 0.20,
            'wai': 0.50,
            'sentiment': 0.10,
            'agreement': 0.20
        },
        PrioritizationStrategy.SENTIMENT_FOCUSED: {
            'confidence': 0.25,
            'wai': 0.15,
            'sentiment': 0.45,
            'agreement': 0.15
        },
        PrioritizationStrategy.CONSENSUS_FOCUSED: {
            'confidence': 0.20,
            'wai': 0.20,
            'sentiment': 0.15,
            'agreement': 0.45
        },
        PrioritizationStrategy.AGGRESSIVE: {
            'confidence': 0.40,
            'wai': 0.30,
            'sentiment': 0.15,
            'agreement': 0.15
        },
        PrioritizationStrategy.CONSERVATIVE: {
            'confidence': 0.25,
            'wai': 0.20,
            'sentiment': 0.20,
            'agreement': 0.35
        }
    }

    def __init__(
        self,
        strategy: Union[PrioritizationStrategy, str] = PrioritizationStrategy.BALANCED,
        custom_weights: Optional[Dict[str, float]] = None,
        min_confidence_threshold: float = 50.0,
        min_wai_threshold: float = 0.0,
        min_agreement_threshold: float = 0.5,
        enable_risk_adjustment: bool = True,
        sentiment_multiplier: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize the Signal Prioritizer.

        Parameters:
        -----------
        strategy : PrioritizationStrategy or str, default='balanced'
            Prioritization strategy to use
        custom_weights : Dict[str, float], optional
            Custom factor weights (must sum to 1.0)
        min_confidence_threshold : float, default=50.0
            Minimum confidence score to consider (0-100)
        min_wai_threshold : float, default=0.0
            Minimum WAI score to consider (0-100)
        min_agreement_threshold : float, default=0.5
            Minimum model agreement percentage (0-1)
        enable_risk_adjustment : bool, default=True
            Whether to apply risk-adjusted scoring
        sentiment_multiplier : float, default=1.0
            Multiplier for sentiment impact (higher = more weight)
        verbose : bool, default=True
            Whether to log prioritization details
        """
        # Convert strategy string to enum if needed
        if isinstance(strategy, str):
            strategy = PrioritizationStrategy(strategy.lower())

        self.strategy = strategy
        self.min_confidence_threshold = min_confidence_threshold
        self.min_wai_threshold = min_wai_threshold
        self.min_agreement_threshold = min_agreement_threshold
        self.enable_risk_adjustment = enable_risk_adjustment
        self.sentiment_multiplier = sentiment_multiplier
        self.verbose = verbose

        # Set factor weights
        if custom_weights is not None:
            # Validate custom weights
            if not np.isclose(sum(custom_weights.values()), 1.0):
                raise ValueError("Custom weights must sum to 1.0")
            self.weights = custom_weights
        else:
            self.weights = self.STRATEGY_WEIGHTS[strategy].copy()

        if self.verbose:
            logger.info(f"SignalPrioritizer initialized with strategy: {strategy.value}")
            logger.info(f"Factor weights: {self.weights}")

    def prioritize_signal(self, signal: SignalInput) -> PrioritizedSignal:
        """
        Prioritize a single signal.

        Parameters:
        -----------
        signal : SignalInput
            Input signal to prioritize

        Returns:
        --------
        PrioritizedSignal
            Prioritized signal with ranking information
        """
        # Calculate individual components
        confidence_component = self._calculate_confidence_component(signal.confidence_score)
        wai_component = self._calculate_wai_component(signal.wai_score)
        sentiment_component = self._calculate_sentiment_component(
            signal.news_sentiment,
            signal.signal_direction
        )
        agreement_component, agreement_pct = self._calculate_agreement_component(
            signal.model_predictions,
            signal.signal_direction
        )

        # Calculate weighted priority score
        priority_score = (
            confidence_component * self.weights['confidence'] +
            wai_component * self.weights['wai'] +
            sentiment_component * self.weights['sentiment'] +
            agreement_component * self.weights['agreement']
        )

        # Apply risk adjustment if enabled
        risk_adjusted_score = self._calculate_risk_adjusted_score(
            priority_score,
            signal.confidence_score,
            agreement_pct
        ) if self.enable_risk_adjustment else priority_score

        # Determine signal strength
        signal_strength = self._classify_signal_strength(risk_adjusted_score)

        # Calculate expected return if prices available
        expected_return = None
        if signal.target_price is not None and signal.current_price is not None:
            expected_return = ((signal.target_price - signal.current_price) /
                             signal.current_price * 100)

        # Store raw scores
        raw_scores = {
            'confidence': signal.confidence_score,
            'wai': signal.wai_score if signal.wai_score is not None else 0.0,
            'sentiment': signal.news_sentiment if signal.news_sentiment is not None else 0.0,
            'agreement': agreement_pct
        }

        # Create prioritized signal
        prioritized = PrioritizedSignal(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            signal_direction=signal.signal_direction,
            priority_score=priority_score,
            priority_rank=0,  # Will be set during batch prioritization
            confidence_component=confidence_component,
            wai_component=wai_component,
            sentiment_component=sentiment_component,
            agreement_component=agreement_component,
            model_agreement_pct=agreement_pct,
            raw_scores=raw_scores,
            risk_adjusted_score=risk_adjusted_score,
            signal_strength=signal_strength,
            target_price=signal.target_price,
            current_price=signal.current_price,
            expected_return=expected_return,
            additional_metadata=signal.additional_metadata
        )

        return prioritized

    def prioritize_signals(
        self,
        signals: List[SignalInput],
        filter_by_threshold: bool = True,
        return_all: bool = False
    ) -> List[PrioritizedSignal]:
        """
        Prioritize multiple signals and rank them.

        Parameters:
        -----------
        signals : List[SignalInput]
            List of signals to prioritize
        filter_by_threshold : bool, default=True
            Whether to filter signals below minimum thresholds
        return_all : bool, default=False
            Whether to return all signals or only actionable ones (BUY/SELL)

        Returns:
        --------
        List[PrioritizedSignal]
            Ranked list of prioritized signals
        """
        if not signals:
            logger.warning("No signals provided for prioritization")
            return []

        # Prioritize each signal
        prioritized_signals = []
        for signal in signals:
            try:
                prioritized = self.prioritize_signal(signal)

                # Apply threshold filtering
                if filter_by_threshold:
                    if not self._meets_thresholds(signal, prioritized):
                        continue

                # Filter HOLD signals if not returning all
                if not return_all and signal.signal_direction == SignalDirection.HOLD:
                    continue

                prioritized_signals.append(prioritized)

            except Exception as e:
                logger.error(f"Error prioritizing signal for {signal.symbol}: {e}")
                continue

        # Sort by priority score (descending)
        prioritized_signals.sort(key=lambda x: x.risk_adjusted_score, reverse=True)

        # Assign ranks
        for rank, signal in enumerate(prioritized_signals, start=1):
            signal.priority_rank = rank

        if self.verbose:
            logger.info(f"Prioritized {len(prioritized_signals)} signals from {len(signals)} inputs")

        return prioritized_signals

    def _calculate_confidence_component(self, confidence_score: float) -> float:
        """
        Calculate the confidence component of the priority score.

        Parameters:
        -----------
        confidence_score : float
            Model confidence score (0-100)

        Returns:
        --------
        float
            Normalized confidence component (0-100)
        """
        # Confidence is already normalized to 0-100
        return confidence_score

    def _calculate_wai_component(self, wai_score: Optional[float]) -> float:
        """
        Calculate the WAI (Whale Activity Index) component.

        Parameters:
        -----------
        wai_score : float or None
            Whale Activity Index score (0-100)

        Returns:
        --------
        float
            Normalized WAI component (0-100)
        """
        if wai_score is None:
            # If no WAI data, return neutral score
            return 50.0

        # WAI is already normalized to 0-100
        # Apply non-linear scaling to emphasize high WAI scores
        # Use sigmoid-like transformation
        normalized_wai = wai_score / 100.0
        enhanced_wai = 100 * (1 / (1 + np.exp(-5 * (normalized_wai - 0.5))))

        return enhanced_wai

    def _calculate_sentiment_component(
        self,
        sentiment_score: Optional[float],
        signal_direction: SignalDirection
    ) -> float:
        """
        Calculate the news sentiment component.

        Parameters:
        -----------
        sentiment_score : float or None
            News sentiment score (-1 to +1)
        signal_direction : SignalDirection
            Trading signal direction

        Returns:
        --------
        float
            Normalized sentiment component (0-100)
        """
        if sentiment_score is None:
            # If no sentiment data, return neutral score
            return 50.0

        # Convert sentiment (-1 to +1) to 0-100 scale
        base_sentiment = (sentiment_score + 1) * 50

        # Check if sentiment aligns with signal direction
        sentiment_aligned = False
        if signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            sentiment_aligned = sentiment_score > 0
        elif signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
            sentiment_aligned = sentiment_score < 0
        else:  # HOLD
            sentiment_aligned = abs(sentiment_score) < 0.2

        # Boost score if sentiment aligns with signal
        if sentiment_aligned:
            alignment_boost = abs(sentiment_score) * 20 * self.sentiment_multiplier
            base_sentiment = min(100, base_sentiment + alignment_boost)
        else:
            # Penalize if sentiment conflicts
            alignment_penalty = abs(sentiment_score) * 15 * self.sentiment_multiplier
            base_sentiment = max(0, base_sentiment - alignment_penalty)

        return base_sentiment

    def _calculate_agreement_component(
        self,
        model_predictions: Optional[Dict[str, float]],
        signal_direction: SignalDirection
    ) -> Tuple[float, float]:
        """
        Calculate the model agreement component.

        Parameters:
        -----------
        model_predictions : Dict[str, float] or None
            Individual model predictions {model_name: predicted_return}
        signal_direction : SignalDirection
            Trading signal direction

        Returns:
        --------
        Tuple[float, float]
            (agreement_component (0-100), agreement_percentage (0-1))
        """
        if not model_predictions or len(model_predictions) == 0:
            # If no model predictions, return neutral
            return 50.0, 0.5

        # Determine expected direction based on signal
        if signal_direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
            expected_positive = True
        elif signal_direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
            expected_positive = False
        else:  # HOLD
            # For HOLD, expect predictions close to zero
            neutral_predictions = sum(1 for pred in model_predictions.values()
                                     if abs(pred) < 0.005)
            agreement_pct = neutral_predictions / len(model_predictions)
            return agreement_pct * 100, agreement_pct

        # Count models agreeing with the signal direction
        agreeing_models = 0
        for model_name, prediction in model_predictions.items():
            if expected_positive and prediction > 0:
                agreeing_models += 1
            elif not expected_positive and prediction < 0:
                agreeing_models += 1

        # Calculate agreement percentage
        agreement_pct = agreeing_models / len(model_predictions)

        # Calculate component score with non-linear scaling
        # Higher agreement gets disproportionately higher scores
        if agreement_pct >= 0.8:
            # Very high agreement: 80-100
            component_score = 80 + (agreement_pct - 0.8) * 100
        elif agreement_pct >= 0.6:
            # Good agreement: 60-80
            component_score = 60 + (agreement_pct - 0.6) * 100
        elif agreement_pct >= 0.4:
            # Moderate agreement: 40-60
            component_score = 40 + (agreement_pct - 0.4) * 100
        else:
            # Low agreement: 0-40
            component_score = agreement_pct * 100

        # Also consider prediction magnitude consistency
        if len(model_predictions) > 1:
            predictions_array = np.array(list(model_predictions.values()))
            # Calculate coefficient of variation (lower is better)
            cv = np.std(predictions_array) / (np.abs(np.mean(predictions_array)) + 1e-10)
            # Adjust component score based on consistency
            consistency_factor = 1 / (1 + cv)
            component_score = component_score * (0.7 + 0.3 * consistency_factor)

        return component_score, agreement_pct

    def _calculate_risk_adjusted_score(
        self,
        priority_score: float,
        confidence_score: float,
        agreement_pct: float
    ) -> float:
        """
        Calculate risk-adjusted priority score.

        Parameters:
        -----------
        priority_score : float
            Base priority score
        confidence_score : float
            Model confidence score (0-100)
        agreement_pct : float
            Model agreement percentage (0-1)

        Returns:
        --------
        float
            Risk-adjusted priority score
        """
        # Calculate risk factor based on confidence and agreement
        # Higher confidence and agreement = lower risk
        confidence_factor = confidence_score / 100.0
        agreement_factor = agreement_pct

        # Risk factor: 0 (high risk) to 1 (low risk)
        risk_factor = (confidence_factor * 0.6 + agreement_factor * 0.4)

        # Apply risk adjustment
        # High risk signals get penalized more
        risk_adjusted = priority_score * (0.5 + 0.5 * risk_factor)

        return risk_adjusted

    def _classify_signal_strength(self, risk_adjusted_score: float) -> str:
        """
        Classify signal strength based on risk-adjusted score.

        Parameters:
        -----------
        risk_adjusted_score : float
            Risk-adjusted priority score

        Returns:
        --------
        str
            Signal strength category
        """
        if risk_adjusted_score >= 80:
            return 'VERY_STRONG'
        elif risk_adjusted_score >= 65:
            return 'STRONG'
        elif risk_adjusted_score >= 50:
            return 'MODERATE'
        else:
            return 'WEAK'

    def _meets_thresholds(
        self,
        signal: SignalInput,
        prioritized: PrioritizedSignal
    ) -> bool:
        """
        Check if signal meets minimum thresholds.

        Parameters:
        -----------
        signal : SignalInput
            Original signal input
        prioritized : PrioritizedSignal
            Prioritized signal

        Returns:
        --------
        bool
            True if signal meets all thresholds
        """
        # Check confidence threshold
        if signal.confidence_score < self.min_confidence_threshold:
            return False

        # Check WAI threshold if available
        if signal.wai_score is not None:
            if signal.wai_score < self.min_wai_threshold:
                return False

        # Check agreement threshold
        if prioritized.model_agreement_pct < self.min_agreement_threshold:
            return False

        return True

    def to_dataframe(
        self,
        prioritized_signals: List[PrioritizedSignal],
        include_components: bool = True
    ) -> pd.DataFrame:
        """
        Convert prioritized signals to a pandas DataFrame.

        Parameters:
        -----------
        prioritized_signals : List[PrioritizedSignal]
            List of prioritized signals
        include_components : bool, default=True
            Whether to include component scores

        Returns:
        --------
        pd.DataFrame
            DataFrame with prioritized signals
        """
        if not prioritized_signals:
            return pd.DataFrame()

        # Convert to list of dictionaries
        data = []
        for signal in prioritized_signals:
            row = {
                'rank': signal.priority_rank,
                'symbol': signal.symbol,
                'timestamp': signal.timestamp,
                'signal': signal.signal_direction.name,
                'priority_score': round(signal.priority_score, 2),
                'risk_adjusted_score': round(signal.risk_adjusted_score, 2),
                'signal_strength': signal.signal_strength,
                'model_agreement_pct': round(signal.model_agreement_pct * 100, 1),
            }

            if include_components:
                row.update({
                    'confidence_component': round(signal.confidence_component, 2),
                    'wai_component': round(signal.wai_component, 2),
                    'sentiment_component': round(signal.sentiment_component, 2),
                    'agreement_component': round(signal.agreement_component, 2),
                })

            # Add price and return info if available
            if signal.current_price is not None:
                row['current_price'] = round(signal.current_price, 2)
            if signal.target_price is not None:
                row['target_price'] = round(signal.target_price, 2)
            if signal.expected_return is not None:
                row['expected_return_pct'] = round(signal.expected_return, 2)

            data.append(row)

        return pd.DataFrame(data)

    def get_top_signals(
        self,
        prioritized_signals: List[PrioritizedSignal],
        top_n: int = 10,
        signal_filter: Optional[SignalDirection] = None
    ) -> List[PrioritizedSignal]:
        """
        Get top N prioritized signals.

        Parameters:
        -----------
        prioritized_signals : List[PrioritizedSignal]
            List of all prioritized signals
        top_n : int, default=10
            Number of top signals to return
        signal_filter : SignalDirection, optional
            Filter by specific signal direction

        Returns:
        --------
        List[PrioritizedSignal]
            Top N signals
        """
        filtered_signals = prioritized_signals

        if signal_filter is not None:
            filtered_signals = [s for s in prioritized_signals
                               if s.signal_direction == signal_filter]

        return filtered_signals[:top_n]


# Convenience functions

def prioritize_signals(
    signals: List[SignalInput],
    strategy: str = 'balanced',
    min_confidence: float = 50.0,
    top_n: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Quick function to prioritize signals and return as DataFrame.

    Parameters:
    -----------
    signals : List[SignalInput]
        Signals to prioritize
    strategy : str, default='balanced'
        Prioritization strategy
    min_confidence : float, default=50.0
        Minimum confidence threshold
    top_n : int, optional
        Return only top N signals
    **kwargs
        Additional arguments for SignalPrioritizer

    Returns:
    --------
    pd.DataFrame
        Prioritized signals as DataFrame
    """
    prioritizer = SignalPrioritizer(
        strategy=strategy,
        min_confidence_threshold=min_confidence,
        **kwargs
    )

    prioritized = prioritizer.prioritize_signals(signals)

    if top_n is not None:
        prioritized = prioritized[:top_n]

    return prioritizer.to_dataframe(prioritized)


def create_signal_input(
    symbol: str,
    signal_direction: str,
    confidence_score: float,
    wai_score: Optional[float] = None,
    news_sentiment: Optional[float] = None,
    model_predictions: Optional[Dict[str, float]] = None,
    **kwargs
) -> SignalInput:
    """
    Helper function to create a SignalInput object.

    Parameters:
    -----------
    symbol : str
        Stock symbol
    signal_direction : str
        Signal direction (e.g., 'BUY', 'SELL', 'STRONG_BUY')
    confidence_score : float
        Model confidence (0-100)
    wai_score : float, optional
        Whale Activity Index (0-100)
    news_sentiment : float, optional
        News sentiment (-1 to +1)
    model_predictions : Dict[str, float], optional
        Individual model predictions
    **kwargs
        Additional parameters for SignalInput

    Returns:
    --------
    SignalInput
        Created signal input object
    """
    return SignalInput(
        symbol=symbol,
        timestamp=kwargs.get('timestamp', datetime.now()),
        signal_direction=SignalDirection[signal_direction.upper().replace(' ', '_')],
        confidence_score=confidence_score,
        wai_score=wai_score,
        news_sentiment=news_sentiment,
        model_predictions=model_predictions,
        target_price=kwargs.get('target_price'),
        current_price=kwargs.get('current_price'),
        additional_metadata=kwargs.get('additional_metadata', {})
    )


if __name__ == "__main__":
    # Example usage and demonstration
    print("Signal Prioritizer - Multi-Factor Ranking Algorithm")
    print("=" * 70)

    # Create example signals
    example_signals = [
        create_signal_input(
            symbol='THYAO',
            signal_direction='STRONG_BUY',
            confidence_score=85.0,
            wai_score=78.0,
            news_sentiment=0.6,
            model_predictions={
                'LSTM': 0.035,
                'GRU': 0.028,
                'XGBoost': 0.032,
                'LightGBM': 0.030
            },
            current_price=100.0,
            target_price=103.5
        ),
        create_signal_input(
            symbol='AKBNK',
            signal_direction='BUY',
            confidence_score=72.0,
            wai_score=45.0,
            news_sentiment=0.2,
            model_predictions={
                'LSTM': 0.015,
                'GRU': 0.018,
                'XGBoost': -0.002,
                'LightGBM': 0.012
            },
            current_price=50.0,
            target_price=51.0
        ),
        create_signal_input(
            symbol='GARAN',
            signal_direction='SELL',
            confidence_score=68.0,
            wai_score=62.0,
            news_sentiment=-0.4,
            model_predictions={
                'LSTM': -0.025,
                'GRU': -0.022,
                'XGBoost': -0.028,
                'LightGBM': -0.024
            },
            current_price=95.0,
            target_price=92.5
        )
    ]

    # Prioritize with balanced strategy
    print("\n1. Balanced Strategy:")
    print("-" * 70)
    df_balanced = prioritize_signals(example_signals, strategy='balanced')
    print(df_balanced.to_string(index=False))

    # Prioritize with whale-focused strategy
    print("\n2. Whale-Focused Strategy:")
    print("-" * 70)
    df_whale = prioritize_signals(example_signals, strategy='whale_focused')
    print(df_whale[['rank', 'symbol', 'signal', 'risk_adjusted_score',
                    'wai_component']].to_string(index=False))

    # Prioritize with consensus-focused strategy
    print("\n3. Consensus-Focused Strategy:")
    print("-" * 70)
    df_consensus = prioritize_signals(example_signals, strategy='consensus_focused')
    print(df_consensus[['rank', 'symbol', 'signal', 'risk_adjusted_score',
                       'model_agreement_pct']].to_string(index=False))

    print("\n" + "=" * 70)
    print("Signal Prioritizer ready for use!")
