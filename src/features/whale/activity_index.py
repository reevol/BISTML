"""
Whale Activity Index (WAI) - Institutional Flow Analysis for BIST

This module implements the Whale Activity Index (WAI) system for detecting and
quantifying institutional trading activity on the Borsa Istanbul (BIST). It tracks
large broker movements, identifies accumulation/distribution patterns, and detects
unusual institutional activity that may indicate manipulation or informed trading.

Key Features:
- Whale Activity Index (WAI) calculation
- Top N broker flow tracking relative to average daily volume
- Unusual activity detection using statistical methods
- Price-flow discrepancy identification
- Accumulation/distribution pattern recognition
- Institutional pressure metrics
- Multi-timeframe analysis support

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict
import warnings
from scipy import stats
from scipy.signal import find_peaks
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhaleActivityIndex:
    """
    A comprehensive class for calculating and analyzing whale (institutional)
    trading activity indices.

    The Whale Activity Index (WAI) quantifies the net flow impact of top N brokers
    relative to the stock's average daily volume, helping identify unusual
    institutional activity, accumulation/distribution patterns, and potential
    price manipulation.
    """

    def __init__(self,
                 brokerage_data: pd.DataFrame,
                 price_data: Optional[pd.DataFrame] = None,
                 volume_data: Optional[pd.DataFrame] = None,
                 top_n_brokers: int = 10,
                 lookback_period: int = 20):
        """
        Initialize the WhaleActivityIndex class.

        Parameters:
        -----------
        brokerage_data : pd.DataFrame
            DataFrame with brokerage distribution data. Required columns:
            'date', 'symbol', 'broker_code', 'buy_volume', 'sell_volume',
            'net_volume', 'buy_value', 'sell_value'
        price_data : pd.DataFrame, optional
            DataFrame with price data. Columns: 'date', 'symbol', 'close', 'open', 'high', 'low'
        volume_data : pd.DataFrame, optional
            DataFrame with volume data. Columns: 'date', 'symbol', 'volume'
        top_n_brokers : int, default=10
            Number of top brokers to track for whale analysis
        lookback_period : int, default=20
            Default lookback period for rolling calculations
        """
        self.brokerage_data = brokerage_data.copy()
        self.price_data = price_data.copy() if price_data is not None else None
        self.volume_data = volume_data.copy() if volume_data is not None else None
        self.top_n_brokers = top_n_brokers
        self.lookback_period = lookback_period

        # Standardize column names
        self._standardize_columns()

        # Validate data
        self._validate_data()

        logger.info(f"WhaleActivityIndex initialized with {len(self.brokerage_data)} records")

    def _standardize_columns(self):
        """Standardize column names to lowercase for consistency."""
        if not self.brokerage_data.empty:
            self.brokerage_data.columns = [col.lower() for col in self.brokerage_data.columns]

        if self.price_data is not None and not self.price_data.empty:
            self.price_data.columns = [col.lower() for col in self.price_data.columns]

        if self.volume_data is not None and not self.volume_data.empty:
            self.volume_data.columns = [col.lower() for col in self.volume_data.columns]

    def _validate_data(self):
        """Validate that required columns exist in the data."""
        required_broker_cols = ['date', 'symbol', 'broker_code', 'buy_volume',
                               'sell_volume', 'net_volume']

        missing_cols = [col for col in required_broker_cols
                       if col not in self.brokerage_data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns in brokerage_data: {missing_cols}")

        # Convert date columns to datetime
        self.brokerage_data['date'] = pd.to_datetime(self.brokerage_data['date'])

        if self.price_data is not None:
            self.price_data['date'] = pd.to_datetime(self.price_data['date'])

        if self.volume_data is not None:
            self.volume_data['date'] = pd.to_datetime(self.volume_data['date'])

    def identify_top_brokers(self,
                            symbol: str,
                            by: str = 'net_volume',
                            absolute: bool = True) -> List[str]:
        """
        Identify top N brokers for a given symbol.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        by : str, default='net_volume'
            Metric to rank brokers by ('net_volume', 'buy_volume', 'sell_volume',
            'total_volume', 'buy_value', 'sell_value')
        absolute : bool, default=True
            If True, use absolute values for ranking (treats large sells same as large buys)

        Returns:
        --------
        List[str]
            List of top N broker codes
        """
        symbol_data = self.brokerage_data[
            self.brokerage_data['symbol'] == symbol
        ].copy()

        if symbol_data.empty:
            logger.warning(f"No brokerage data found for {symbol}")
            return []

        # Aggregate by broker
        if by == 'total_volume':
            broker_agg = symbol_data.groupby('broker_code').agg({
                'buy_volume': 'sum',
                'sell_volume': 'sum'
            })
            broker_agg['total_volume'] = (
                broker_agg['buy_volume'] + broker_agg['sell_volume']
            )
            rank_col = 'total_volume'
        else:
            broker_agg = symbol_data.groupby('broker_code')[by].sum()
            rank_col = by

        # Apply absolute value if requested
        if absolute:
            broker_agg_sorted = broker_agg[rank_col].abs().sort_values(ascending=False)
        else:
            broker_agg_sorted = broker_agg[rank_col].sort_values(ascending=False)

        top_brokers = broker_agg_sorted.head(self.top_n_brokers).index.tolist()

        logger.info(f"Identified {len(top_brokers)} top brokers for {symbol}")
        return top_brokers

    def calculate_wai(self,
                     symbol: str,
                     custom_brokers: Optional[List[str]] = None,
                     normalize: bool = True,
                     include_components: bool = True) -> pd.DataFrame:
        """
        Calculate the Whale Activity Index (WAI) for a symbol.

        The WAI quantifies institutional trading pressure by measuring the net flow
        of top brokers relative to average daily volume. Higher values indicate
        stronger institutional interest.

        WAI = (Whale Net Flow / Average Daily Volume) * Whale Participation %

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        custom_brokers : List[str], optional
            Specific broker codes to track. If None, uses top N brokers
        normalize : bool, default=True
            If True, normalize WAI scores to 0-100 scale
        include_components : bool, default=True
            If True, include component metrics in output

        Returns:
        --------
        pd.DataFrame
            DataFrame with WAI scores and components by date
        """
        symbol_data = self.brokerage_data[
            self.brokerage_data['symbol'] == symbol
        ].copy()

        if symbol_data.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()

        # Identify top brokers if not provided
        if custom_brokers is None:
            top_brokers = self.identify_top_brokers(symbol, by='total_volume')
        else:
            top_brokers = custom_brokers

        if not top_brokers:
            return pd.DataFrame()

        # Filter for whale brokers
        whale_data = symbol_data[
            symbol_data['broker_code'].isin(top_brokers)
        ].copy()

        # Aggregate whale activity by date
        daily_whale = whale_data.groupby('date').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'net_volume': 'sum',
            'buy_value': 'sum',
            'sell_value': 'sum'
        }).reset_index()

        daily_whale['whale_total_volume'] = (
            daily_whale['buy_volume'] + daily_whale['sell_volume']
        )

        # Calculate total market activity
        daily_total = symbol_data.groupby('date').agg({
            'buy_volume': 'sum',
            'sell_volume': 'sum'
        }).reset_index()

        daily_total['total_volume'] = (
            daily_total['buy_volume'] + daily_total['sell_volume']
        )

        # Merge whale and total data
        wai_data = daily_whale.merge(
            daily_total[['date', 'total_volume']],
            on='date',
            how='left'
        )

        # Calculate average daily volume (rolling)
        wai_data['avg_daily_volume'] = (
            wai_data['total_volume'].rolling(
                window=self.lookback_period,
                min_periods=5
            ).mean()
        )

        # Calculate WAI components
        # 1. Whale Participation Rate
        wai_data['whale_participation_pct'] = (
            wai_data['whale_total_volume'] / wai_data['total_volume'] * 100
        )

        # 2. Net Flow Ratio (relative to average volume)
        wai_data['net_flow_ratio'] = (
            wai_data['net_volume'] / wai_data['avg_daily_volume']
        )

        # 3. Net Flow Intensity (absolute magnitude)
        wai_data['net_flow_intensity'] = (
            abs(wai_data['net_volume']) / wai_data['avg_daily_volume']
        )

        # 4. Directional Strength (consistency of direction)
        wai_data['net_flow_sign'] = np.sign(wai_data['net_volume'])
        wai_data['directional_consistency'] = (
            wai_data['net_flow_sign'].rolling(
                window=5,
                min_periods=3
            ).apply(lambda x: abs(x.sum()) / len(x), raw=True)
        )

        # Calculate WAI Score
        # Combines participation rate and net flow magnitude
        wai_data['wai_raw'] = (
            wai_data['whale_participation_pct'] * 0.4 +
            wai_data['net_flow_intensity'] * 100 * 0.6
        )

        # Directional WAI (positive for accumulation, negative for distribution)
        wai_data['wai_directional'] = (
            wai_data['wai_raw'] * np.sign(wai_data['net_volume'])
        )

        # Momentum-adjusted WAI (considers consistency)
        wai_data['wai_momentum'] = (
            wai_data['wai_directional'] * wai_data['directional_consistency']
        )

        # Normalize if requested
        if normalize:
            # Use rolling percentile for dynamic normalization
            wai_data['wai_normalized'] = (
                wai_data['wai_raw'].rolling(
                    window=60,
                    min_periods=20
                ).apply(
                    lambda x: stats.percentileofscore(x, x.iloc[-1]),
                    raw=False
                )
            )
            wai_data['wai_score'] = wai_data['wai_normalized']
        else:
            wai_data['wai_score'] = wai_data['wai_raw']

        # Add metadata
        wai_data['symbol'] = symbol
        wai_data['top_broker_count'] = len(top_brokers)

        # Select output columns
        output_cols = ['date', 'symbol', 'wai_score', 'wai_directional', 'wai_momentum']

        if include_components:
            output_cols.extend([
                'whale_participation_pct',
                'net_flow_ratio',
                'net_flow_intensity',
                'directional_consistency',
                'net_volume',
                'whale_total_volume',
                'total_volume',
                'top_broker_count'
            ])

        return wai_data[output_cols].sort_values('date')

    def detect_unusual_activity(self,
                               symbol: str,
                               z_threshold: float = 2.5,
                               min_participation: float = 20.0) -> pd.DataFrame:
        """
        Detect unusual whale activity using statistical methods.

        Identifies days where whale activity significantly deviates from normal
        patterns, which may indicate informed trading, manipulation, or major
        institutional positioning.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        z_threshold : float, default=2.5
            Z-score threshold for flagging unusual activity
        min_participation : float, default=20.0
            Minimum whale participation percentage to consider

        Returns:
        --------
        pd.DataFrame
            DataFrame with unusual activity flags and statistics
        """
        # Calculate WAI
        wai_data = self.calculate_wai(symbol, include_components=True)

        if wai_data.empty:
            return pd.DataFrame()

        # Calculate rolling statistics
        window = self.lookback_period

        # Z-score for net flow ratio
        wai_data['net_flow_mean'] = (
            wai_data['net_flow_ratio'].rolling(window=window, min_periods=10).mean()
        )
        wai_data['net_flow_std'] = (
            wai_data['net_flow_ratio'].rolling(window=window, min_periods=10).std()
        )
        wai_data['net_flow_zscore'] = (
            (wai_data['net_flow_ratio'] - wai_data['net_flow_mean']) /
            wai_data['net_flow_std'].replace(0, 1)
        )

        # Z-score for whale participation
        wai_data['participation_mean'] = (
            wai_data['whale_participation_pct'].rolling(
                window=window, min_periods=10
            ).mean()
        )
        wai_data['participation_std'] = (
            wai_data['whale_participation_pct'].rolling(
                window=window, min_periods=10
            ).std()
        )
        wai_data['participation_zscore'] = (
            (wai_data['whale_participation_pct'] - wai_data['participation_mean']) /
            wai_data['participation_std'].replace(0, 1)
        )

        # Detect unusual activity
        wai_data['unusual_accumulation'] = (
            (wai_data['net_flow_zscore'] > z_threshold) &
            (wai_data['whale_participation_pct'] >= min_participation)
        )

        wai_data['unusual_distribution'] = (
            (wai_data['net_flow_zscore'] < -z_threshold) &
            (wai_data['whale_participation_pct'] >= min_participation)
        )

        wai_data['unusual_participation'] = (
            abs(wai_data['participation_zscore']) > z_threshold
        )

        # Combined unusual activity flag
        wai_data['unusual_activity'] = (
            wai_data['unusual_accumulation'] |
            wai_data['unusual_distribution'] |
            wai_data['unusual_participation']
        )

        # Calculate unusual activity strength
        wai_data['unusual_strength'] = (
            abs(wai_data['net_flow_zscore']) +
            abs(wai_data['participation_zscore'])
        ) / 2

        # Activity classification
        wai_data['activity_type'] = 'NORMAL'
        wai_data.loc[wai_data['unusual_accumulation'], 'activity_type'] = 'UNUSUAL_ACCUMULATION'
        wai_data.loc[wai_data['unusual_distribution'], 'activity_type'] = 'UNUSUAL_DISTRIBUTION'
        wai_data.loc[wai_data['unusual_participation'], 'activity_type'] = 'UNUSUAL_PARTICIPATION'

        return wai_data

    def detect_price_flow_discrepancy(self,
                                     symbol: str,
                                     discrepancy_threshold: float = 1.5,
                                     window: int = 5) -> pd.DataFrame:
        """
        Detect discrepancies between price movement and whale flow.

        Identifies situations where price action doesn't match institutional flow,
        which may indicate price manipulation, accumulation under selling pressure,
        or distribution during price pumps.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        discrepancy_threshold : float, default=1.5
            Threshold for flagging significant discrepancies (in standard deviations)
        window : int, default=5
            Rolling window for price and flow smoothing

        Returns:
        --------
        pd.DataFrame
            DataFrame with discrepancy analysis
        """
        if self.price_data is None:
            raise ValueError("Price data is required for price-flow discrepancy analysis")

        # Calculate WAI
        wai_data = self.calculate_wai(symbol, include_components=True)

        if wai_data.empty:
            return pd.DataFrame()

        # Get price data for symbol
        price_symbol = self.price_data[
            self.price_data['symbol'] == symbol
        ].copy()

        if price_symbol.empty:
            logger.warning(f"No price data found for {symbol}")
            return pd.DataFrame()

        # Merge with price data
        merged = wai_data.merge(
            price_symbol[['date', 'close', 'open', 'high', 'low']],
            on='date',
            how='inner'
        )

        if merged.empty:
            return pd.DataFrame()

        merged = merged.sort_values('date')

        # Calculate price changes
        merged['price_change'] = merged['close'].pct_change() * 100
        merged['price_change_smooth'] = (
            merged['price_change'].rolling(window=window).mean()
        )

        # Calculate expected price movement based on flow
        # Positive flow should lead to positive price movement
        merged['flow_direction'] = np.sign(merged['net_flow_ratio'])
        merged['flow_magnitude'] = abs(merged['net_flow_ratio']) * 100
        merged['flow_smooth'] = merged['net_flow_ratio'].rolling(window=window).mean()

        # Calculate discrepancy
        # High discrepancy = strong flow in one direction, price moving opposite
        merged['flow_price_correlation'] = (
            merged['flow_smooth'] * merged['price_change_smooth']
        )

        merged['discrepancy_raw'] = (
            merged['flow_smooth'] - merged['price_change_smooth']
        )

        # Normalize discrepancy
        discrepancy_mean = merged['discrepancy_raw'].rolling(
            window=20, min_periods=10
        ).mean()
        discrepancy_std = merged['discrepancy_raw'].rolling(
            window=20, min_periods=10
        ).std()

        merged['discrepancy_zscore'] = (
            (merged['discrepancy_raw'] - discrepancy_mean) /
            discrepancy_std.replace(0, 1)
        )

        # Detect specific patterns
        # 1. Accumulation under pressure (whales buying, price falling)
        merged['accumulation_under_pressure'] = (
            (merged['net_flow_ratio'] > 0.1) &
            (merged['price_change_smooth'] < -0.5) &
            (merged['whale_participation_pct'] > 15)
        )

        # 2. Distribution during pump (whales selling, price rising)
        merged['distribution_during_pump'] = (
            (merged['net_flow_ratio'] < -0.1) &
            (merged['price_change_smooth'] > 0.5) &
            (merged['whale_participation_pct'] > 15)
        )

        # 3. Significant discrepancy
        merged['significant_discrepancy'] = (
            abs(merged['discrepancy_zscore']) > discrepancy_threshold
        )

        # 4. Potential manipulation flag
        merged['potential_manipulation'] = (
            merged['accumulation_under_pressure'] |
            merged['distribution_during_pump']
        ) & merged['significant_discrepancy']

        # Calculate manipulation score
        merged['manipulation_score'] = (
            abs(merged['discrepancy_zscore']) *
            (merged['whale_participation_pct'] / 100)
        )

        return merged

    def calculate_accumulation_distribution_score(self,
                                                  symbol: str,
                                                  short_window: int = 5,
                                                  long_window: int = 20) -> pd.DataFrame:
        """
        Calculate accumulation/distribution scores using multiple timeframes.

        Identifies whether whales are systematically accumulating or distributing
        positions over time by analyzing flow patterns across different windows.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        short_window : int, default=5
            Short-term window for recent activity
        long_window : int, default=20
            Long-term window for trend identification

        Returns:
        --------
        pd.DataFrame
            DataFrame with accumulation/distribution scores
        """
        # Calculate WAI
        wai_data = self.calculate_wai(symbol, include_components=True)

        if wai_data.empty:
            return pd.DataFrame()

        # Short-term accumulation/distribution
        wai_data['net_flow_short'] = (
            wai_data['net_volume'].rolling(window=short_window).sum()
        )
        wai_data['avg_volume_short'] = (
            wai_data['total_volume'].rolling(window=short_window).mean()
        )
        wai_data['ad_score_short'] = (
            wai_data['net_flow_short'] / wai_data['avg_volume_short']
        )

        # Long-term accumulation/distribution
        wai_data['net_flow_long'] = (
            wai_data['net_volume'].rolling(window=long_window).sum()
        )
        wai_data['avg_volume_long'] = (
            wai_data['total_volume'].rolling(window=long_window).mean()
        )
        wai_data['ad_score_long'] = (
            wai_data['net_flow_long'] / wai_data['avg_volume_long']
        )

        # Trend consistency
        wai_data['ad_trend_alignment'] = (
            np.sign(wai_data['ad_score_short']) ==
            np.sign(wai_data['ad_score_long'])
        ).astype(int)

        # Combined score
        # When short and long term align, signal is stronger
        wai_data['ad_score_combined'] = (
            (wai_data['ad_score_short'] * 0.6 + wai_data['ad_score_long'] * 0.4) *
            (1 + wai_data['ad_trend_alignment'] * 0.3)
        )

        # Classify phase
        wai_data['phase'] = 'NEUTRAL'
        wai_data.loc[wai_data['ad_score_combined'] > 0.2, 'phase'] = 'ACCUMULATION'
        wai_data.loc[wai_data['ad_score_combined'] < -0.2, 'phase'] = 'DISTRIBUTION'
        wai_data.loc[
            (wai_data['ad_score_combined'] > 0.5) &
            (wai_data['ad_trend_alignment'] == 1),
            'phase'
        ] = 'STRONG_ACCUMULATION'
        wai_data.loc[
            (wai_data['ad_score_combined'] < -0.5) &
            (wai_data['ad_trend_alignment'] == 1),
            'phase'
        ] = 'STRONG_DISTRIBUTION'

        # Calculate strength (0-100 scale)
        wai_data['phase_strength'] = (
            np.tanh(wai_data['ad_score_combined']) * 50 + 50
        )

        return wai_data

    def calculate_institutional_pressure(self,
                                        symbol: str,
                                        window: int = 10) -> pd.DataFrame:
        """
        Calculate institutional buying/selling pressure metrics.

        Measures the sustained directional pressure from institutional traders,
        which can precede significant price movements.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        window : int, default=10
            Window for pressure calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with institutional pressure metrics
        """
        # Calculate WAI
        wai_data = self.calculate_wai(symbol, include_components=True)

        if wai_data.empty:
            return pd.DataFrame()

        # Calculate cumulative net flow
        wai_data['cumulative_net_flow'] = wai_data['net_volume'].cumsum()

        # Buying pressure (percentage of days with net buying in window)
        wai_data['buy_pressure_pct'] = (
            (wai_data['net_volume'] > 0).rolling(window=window).mean() * 100
        )

        # Selling pressure
        wai_data['sell_pressure_pct'] = (
            (wai_data['net_volume'] < 0).rolling(window=window).mean() * 100
        )

        # Net pressure (buy - sell)
        wai_data['net_pressure'] = (
            wai_data['buy_pressure_pct'] - wai_data['sell_pressure_pct']
        )

        # Pressure intensity (magnitude of average flow in window)
        wai_data['pressure_intensity'] = (
            abs(wai_data['net_volume']).rolling(window=window).mean() /
            wai_data['total_volume'].rolling(window=window).mean()
        )

        # Pressure consistency (low std = consistent pressure)
        net_flow_std = wai_data['net_flow_ratio'].rolling(window=window).std()
        net_flow_mean = abs(wai_data['net_flow_ratio'].rolling(window=window).mean())

        wai_data['pressure_consistency'] = (
            1 / (1 + (net_flow_std / net_flow_mean.replace(0, 1)))
        )

        # Combined pressure score
        wai_data['institutional_pressure_score'] = (
            (wai_data['net_pressure'] / 100) *
            wai_data['pressure_intensity'] *
            wai_data['pressure_consistency'] *
            100
        )

        # Classify pressure level
        wai_data['pressure_level'] = 'NEUTRAL'
        wai_data.loc[
            wai_data['institutional_pressure_score'] > 30,
            'pressure_level'
        ] = 'STRONG_BUY_PRESSURE'
        wai_data.loc[
            wai_data['institutional_pressure_score'] > 15,
            'pressure_level'
        ] = 'MODERATE_BUY_PRESSURE'
        wai_data.loc[
            wai_data['institutional_pressure_score'] < -30,
            'pressure_level'
        ] = 'STRONG_SELL_PRESSURE'
        wai_data.loc[
            wai_data['institutional_pressure_score'] < -15,
            'pressure_level'
        ] = 'MODERATE_SELL_PRESSURE'

        return wai_data

    def generate_whale_signals(self,
                              symbol: str,
                              confidence_threshold: float = 70.0) -> pd.DataFrame:
        """
        Generate trading signals based on whale activity analysis.

        Combines multiple whale activity metrics to produce actionable trading signals
        with confidence scores.

        Parameters:
        -----------
        symbol : str
            Stock symbol to analyze
        confidence_threshold : float, default=70.0
            Minimum confidence score to generate signal

        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals and confidence scores
        """
        # Get all analysis components
        unusual_activity = self.detect_unusual_activity(symbol)

        if unusual_activity.empty:
            return pd.DataFrame()

        ad_score = self.calculate_accumulation_distribution_score(symbol)
        pressure = self.calculate_institutional_pressure(symbol)

        # Merge all analyses
        signals = unusual_activity.merge(
            ad_score[['date', 'ad_score_combined', 'phase', 'phase_strength']],
            on='date',
            how='left'
        )

        signals = signals.merge(
            pressure[['date', 'institutional_pressure_score', 'pressure_level']],
            on='date',
            how='left'
        )

        # Calculate signal components
        # 1. WAI component (0-100)
        signals['wai_component'] = signals['wai_score']

        # 2. Unusual activity component
        signals['unusual_component'] = (
            signals['unusual_activity'].astype(int) *
            signals['unusual_strength'] * 20
        ).clip(0, 100)

        # 3. Accumulation/distribution component
        signals['ad_component'] = signals['phase_strength']

        # 4. Pressure component
        signals['pressure_component'] = (
            (signals['institutional_pressure_score'] + 100) / 2
        ).clip(0, 100)

        # Calculate combined confidence score
        signals['confidence_score'] = (
            signals['wai_component'] * 0.3 +
            signals['unusual_component'] * 0.2 +
            signals['ad_component'] * 0.25 +
            signals['pressure_component'] * 0.25
        )

        # Generate signal direction
        signals['signal_direction'] = 0
        signals.loc[
            (signals['net_flow_ratio'] > 0) &
            (signals['institutional_pressure_score'] > 10),
            'signal_direction'
        ] = 1  # BUY
        signals.loc[
            (signals['net_flow_ratio'] < 0) &
            (signals['institutional_pressure_score'] < -10),
            'signal_direction'
        ] = -1  # SELL

        # Generate final signal with confidence threshold
        signals['signal'] = 'HOLD'
        signals.loc[
            (signals['signal_direction'] == 1) &
            (signals['confidence_score'] >= confidence_threshold),
            'signal'
        ] = 'BUY'
        signals.loc[
            (signals['signal_direction'] == 1) &
            (signals['confidence_score'] >= confidence_threshold + 15),
            'signal'
        ] = 'STRONG_BUY'
        signals.loc[
            (signals['signal_direction'] == -1) &
            (signals['confidence_score'] >= confidence_threshold),
            'signal'
        ] = 'SELL'
        signals.loc[
            (signals['signal_direction'] == -1) &
            (signals['confidence_score'] >= confidence_threshold + 15),
            'signal'
        ] = 'STRONG_SELL'

        # Add metadata
        signals['signal_source'] = 'WHALE_ACTIVITY'

        return signals


# Convenience functions for quick calculations

def calculate_wai(brokerage_data: pd.DataFrame,
                 symbol: str,
                 top_n_brokers: int = 10,
                 **kwargs) -> pd.DataFrame:
    """
    Quick function to calculate Whale Activity Index.

    Parameters:
    -----------
    brokerage_data : pd.DataFrame
        Brokerage distribution data
    symbol : str
        Stock symbol
    top_n_brokers : int, default=10
        Number of top brokers to track
    **kwargs : dict
        Additional parameters for WhaleActivityIndex

    Returns:
    --------
    pd.DataFrame
        WAI scores by date
    """
    wai = WhaleActivityIndex(brokerage_data, top_n_brokers=top_n_brokers)
    return wai.calculate_wai(symbol, **kwargs)


def detect_unusual_activity(brokerage_data: pd.DataFrame,
                           symbol: str,
                           z_threshold: float = 2.5,
                           **kwargs) -> pd.DataFrame:
    """
    Quick function to detect unusual whale activity.

    Parameters:
    -----------
    brokerage_data : pd.DataFrame
        Brokerage distribution data
    symbol : str
        Stock symbol
    z_threshold : float, default=2.5
        Z-score threshold for unusual activity
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    pd.DataFrame
        Unusual activity detection results
    """
    wai = WhaleActivityIndex(brokerage_data, **kwargs)
    return wai.detect_unusual_activity(symbol, z_threshold=z_threshold)


def generate_whale_signals(brokerage_data: pd.DataFrame,
                          symbol: str,
                          price_data: Optional[pd.DataFrame] = None,
                          confidence_threshold: float = 70.0,
                          **kwargs) -> pd.DataFrame:
    """
    Quick function to generate whale-based trading signals.

    Parameters:
    -----------
    brokerage_data : pd.DataFrame
        Brokerage distribution data
    symbol : str
        Stock symbol
    price_data : pd.DataFrame, optional
        Price data for enhanced analysis
    confidence_threshold : float, default=70.0
        Minimum confidence for signals
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    pd.DataFrame
        Trading signals with confidence scores
    """
    wai = WhaleActivityIndex(brokerage_data, price_data=price_data, **kwargs)
    return wai.generate_whale_signals(symbol, confidence_threshold=confidence_threshold)


if __name__ == "__main__":
    # Example usage and documentation
    print("Whale Activity Index (WAI) Module")
    print("=" * 70)
    print("\nThis module provides comprehensive whale (institutional) activity analysis")
    print("for BIST stocks, including:")
    print("\n1. Whale Activity Index (WAI) Calculation")
    print("   - Tracks top N brokers' net flow relative to average daily volume")
    print("   - Provides normalized and directional scores")
    print("\n2. Unusual Activity Detection")
    print("   - Statistical anomaly detection (z-score based)")
    print("   - Identifies unusual accumulation/distribution")
    print("\n3. Price-Flow Discrepancy Analysis")
    print("   - Detects manipulation patterns")
    print("   - Identifies accumulation under pressure")
    print("   - Flags distribution during price pumps")
    print("\n4. Accumulation/Distribution Scoring")
    print("   - Multi-timeframe analysis")
    print("   - Phase classification (accumulation/distribution/neutral)")
    print("\n5. Institutional Pressure Metrics")
    print("   - Buying/selling pressure calculation")
    print("   - Pressure consistency and intensity")
    print("\n6. Signal Generation")
    print("   - Combines all metrics into actionable signals")
    print("   - Provides confidence scores")
    print("\n" + "=" * 70)
    print("\nFor usage examples, see the data collectors and backtesting modules.")
