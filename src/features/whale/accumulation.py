"""
Whale Accumulation/Distribution Detection

This module detects long-term accumulation and distribution patterns by institutional
investors (whales). It identifies quiet accumulation despite price suppression and
other smart money footprints in the market.

Key Features:
- Accumulation/Distribution detection with volume analysis
- Price suppression identification during accumulation phases
- Institutional footprint analysis
- Smart money divergence detection
- Volume profile analysis for whale activity
- Long-term accumulation pattern recognition
- Distribution phase detection

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.stats import linregress
from scipy.signal import find_peaks

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using pandas implementations.")


class AccumulationDistributionError(Exception):
    """Base exception for Accumulation/Distribution errors."""
    pass


class InsufficientDataError(AccumulationDistributionError):
    """Raised when there is insufficient data for calculation."""
    pass


class WhaleAccumulationDetector:
    """
    Detects institutional accumulation and distribution patterns.

    This class provides comprehensive analysis of whale activity through
    volume patterns, price action, and market microstructure analysis.

    The detector identifies:
    1. Quiet accumulation during price suppression
    2. Distribution during price pumps
    3. Smart money divergences
    4. Volume profile anomalies
    5. Institutional footprints
    """

    def __init__(self, data: pd.DataFrame, use_talib: bool = True):
        """
        Initialize the WhaleAccumulationDetector.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing OHLCV data with columns: 'open', 'high', 'low', 'close', 'volume'
        use_talib : bool, default=True
            Whether to use TA-Lib if available
        """
        self.data = data.copy()
        self.use_talib = use_talib and TALIB_AVAILABLE
        self._standardize_columns()

        if len(self.data) < 20:
            raise InsufficientDataError("Need at least 20 data points for analysis")

    def _standardize_columns(self):
        """Standardize column names to lowercase for consistency."""
        column_mapping = {}
        for col in self.data.columns:
            lower_col = col.lower()
            if lower_col in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = lower_col

        if column_mapping:
            self.data.rename(columns=column_mapping, inplace=True)

    def calculate_ad_line(self) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        The A/D Line is a volume-based indicator that measures cumulative
        buying and selling pressure. Rising A/D with falling prices suggests
        accumulation.

        Returns:
        --------
        pd.Series
            Accumulation/Distribution Line values
        """
        # Money Flow Multiplier
        clv = ((self.data['close'] - self.data['low']) -
               (self.data['high'] - self.data['close'])) / \
              (self.data['high'] - self.data['low'])

        # Handle division by zero (when high == low)
        clv = clv.fillna(0)

        # Money Flow Volume
        mfv = clv * self.data['volume']

        # Accumulation/Distribution Line (cumulative)
        ad_line = mfv.cumsum()

        return pd.Series(ad_line, index=self.data.index, name='AD_Line')

    def calculate_obv(self) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a momentum indicator that uses volume flow to predict
        changes in stock price. Divergences between OBV and price can
        signal accumulation or distribution.

        Returns:
        --------
        pd.Series
            On-Balance Volume values
        """
        obv = pd.Series(0.0, index=self.data.index)
        close_diff = self.data['close'].diff()

        # Add volume on up days, subtract on down days
        obv[close_diff > 0] = self.data['volume'][close_diff > 0]
        obv[close_diff < 0] = -self.data['volume'][close_diff < 0]
        obv[close_diff == 0] = 0

        obv = obv.cumsum()

        return pd.Series(obv, index=self.data.index, name='OBV')

    def calculate_vwap(self, window: Optional[int] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP represents the average price weighted by volume. Institutional
        traders often use VWAP as a benchmark. Price consistently below VWAP
        with rising volume suggests accumulation.

        Parameters:
        -----------
        window : Optional[int]
            Rolling window period (None for cumulative)

        Returns:
        --------
        pd.Series
            VWAP values
        """
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        volume = self.data['volume']

        if window is None:
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
        else:
            vwap = ((typical_price * volume).rolling(window=window).sum() /
                    volume.rolling(window=window).sum())

        return pd.Series(vwap, index=self.data.index, name='VWAP')

    def detect_price_suppression(self, window: int = 20,
                                  threshold: float = 0.5) -> pd.Series:
        """
        Detect price suppression patterns.

        Identifies periods where price is being kept artificially low despite
        increasing volume or buying pressure. This often indicates accumulation.

        Parameters:
        -----------
        window : int, default=20
            Lookback window for analysis
        threshold : float, default=0.5
            Sensitivity threshold (0-1, higher = more sensitive)

        Returns:
        --------
        pd.Series
            Boolean series indicating suppression periods
        """
        # Calculate price volatility
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=window).std()

        # Calculate volume trend
        volume_ma = self.data['volume'].rolling(window=window).mean()
        volume_ratio = self.data['volume'] / volume_ma

        # Calculate A/D Line slope
        ad_line = self.calculate_ad_line()
        ad_slope = pd.Series(index=self.data.index, dtype=float)

        for i in range(window, len(self.data)):
            y = ad_line.iloc[i-window:i].values
            x = np.arange(window)
            if len(y) == window:
                slope, _, _, _, _ = linregress(x, y)
                ad_slope.iloc[i] = slope

        # Calculate price slope
        price_slope = pd.Series(index=self.data.index, dtype=float)

        for i in range(window, len(self.data)):
            y = self.data['close'].iloc[i-window:i].values
            x = np.arange(window)
            if len(y) == window:
                slope, _, _, _, _ = linregress(x, y)
                price_slope.iloc[i] = slope

        # Suppression: high volume, rising A/D, but flat/falling price
        suppression = (
            (volume_ratio > 1 + threshold) &  # Higher than average volume
            (ad_slope > 0) &  # Rising accumulation
            (price_slope <= 0)  # Flat or falling price
        )

        return pd.Series(suppression, index=self.data.index, name='Price_Suppression')

    def detect_quiet_accumulation(self, window: int = 30,
                                   price_threshold: float = 0.02,
                                   volume_threshold: float = 1.2) -> pd.Series:
        """
        Detect quiet accumulation patterns.

        Identifies periods of accumulation characterized by:
        - Sideways or slightly declining price action
        - Gradually increasing volume
        - Rising A/D Line and OBV

        Parameters:
        -----------
        window : int, default=30
            Analysis window
        price_threshold : float, default=0.02
            Maximum price change threshold (as decimal)
        volume_threshold : float, default=1.2
            Minimum volume increase multiplier

        Returns:
        --------
        pd.Series
            Score indicating accumulation strength (0-100)
        """
        # Price stability (lower is better for accumulation)
        price_change = abs(self.data['close'].pct_change(window))
        price_score = (1 - price_change.clip(0, price_threshold) / price_threshold) * 100

        # Volume increase
        volume_ma_short = self.data['volume'].rolling(window=window//2).mean()
        volume_ma_long = self.data['volume'].rolling(window=window).mean()
        volume_increase = volume_ma_short / volume_ma_long
        volume_score = ((volume_increase - 1) / (volume_threshold - 1)).clip(0, 1) * 100

        # A/D Line trend
        ad_line = self.calculate_ad_line()
        ad_change = ad_line.pct_change(window)
        ad_score = (ad_change / ad_change.abs().max()).clip(0, 1) * 100

        # OBV trend
        obv = self.calculate_obv()
        obv_change = obv.pct_change(window)
        obv_score = (obv_change / obv_change.abs().max()).clip(0, 1) * 100

        # Combined accumulation score
        accumulation_score = (
            price_score * 0.2 +
            volume_score * 0.3 +
            ad_score * 0.25 +
            obv_score * 0.25
        )

        return pd.Series(accumulation_score, index=self.data.index,
                        name='Accumulation_Score')

    def detect_distribution(self, window: int = 30,
                           price_threshold: float = 0.05,
                           volume_threshold: float = 1.5) -> pd.Series:
        """
        Detect distribution patterns.

        Identifies periods where institutions are distributing (selling) positions:
        - Rising or stable prices
        - Very high volume
        - Declining A/D Line and OBV

        Parameters:
        -----------
        window : int, default=30
            Analysis window
        price_threshold : float, default=0.05
            Minimum price increase for distribution
        volume_threshold : float, default=1.5
            Minimum volume spike multiplier

        Returns:
        --------
        pd.Series
            Score indicating distribution strength (0-100)
        """
        # Price increase (higher is suspicious with declining volume indicators)
        price_change = self.data['close'].pct_change(window)
        price_score = (price_change.clip(0, price_threshold * 2) /
                       (price_threshold * 2)) * 100

        # Volume spike
        volume_ma = self.data['volume'].rolling(window=window).mean()
        volume_ratio = self.data['volume'] / volume_ma
        volume_score = ((volume_ratio - 1) / (volume_threshold - 1)).clip(0, 1) * 100

        # A/D Line decline
        ad_line = self.calculate_ad_line()
        ad_change = ad_line.pct_change(window)
        ad_score = (-ad_change / ad_change.abs().max()).clip(0, 1) * 100

        # OBV decline
        obv = self.calculate_obv()
        obv_change = obv.pct_change(window)
        obv_score = (-obv_change / obv_change.abs().max()).clip(0, 1) * 100

        # Combined distribution score
        distribution_score = (
            price_score * 0.25 +
            volume_score * 0.25 +
            ad_score * 0.25 +
            obv_score * 0.25
        )

        return pd.Series(distribution_score, index=self.data.index,
                        name='Distribution_Score')

    def detect_smart_money_divergence(self, window: int = 20) -> Dict[str, pd.Series]:
        """
        Detect divergences between price and volume indicators.

        Smart money divergences occur when price moves in one direction
        while volume indicators move in another, suggesting institutional
        positioning contrary to the crowd.

        Parameters:
        -----------
        window : int, default=20
            Window for trend calculation

        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with bullish and bearish divergence signals
        """
        # Calculate trends
        price = self.data['close']
        ad_line = self.calculate_ad_line()
        obv = self.calculate_obv()

        # Price trend
        price_trend = pd.Series(index=self.data.index, dtype=float)
        for i in range(window, len(self.data)):
            x = np.arange(window)
            y = price.iloc[i-window:i].values
            slope, _, _, _, _ = linregress(x, y)
            price_trend.iloc[i] = slope

        # A/D Line trend
        ad_trend = pd.Series(index=self.data.index, dtype=float)
        for i in range(window, len(self.data)):
            x = np.arange(window)
            y = ad_line.iloc[i-window:i].values
            slope, _, _, _, _ = linregress(x, y)
            ad_trend.iloc[i] = slope

        # OBV trend
        obv_trend = pd.Series(index=self.data.index, dtype=float)
        for i in range(window, len(self.data)):
            x = np.arange(window)
            y = obv.iloc[i-window:i].values
            slope, _, _, _, _ = linregress(x, y)
            obv_trend.iloc[i] = slope

        # Bullish divergence: price down, volume indicators up
        bullish_divergence = (
            (price_trend < 0) &
            ((ad_trend > 0) | (obv_trend > 0))
        )

        # Bearish divergence: price up, volume indicators down
        bearish_divergence = (
            (price_trend > 0) &
            ((ad_trend < 0) | (obv_trend < 0))
        )

        return {
            'bullish_divergence': pd.Series(bullish_divergence,
                                           index=self.data.index,
                                           name='Bullish_Divergence'),
            'bearish_divergence': pd.Series(bearish_divergence,
                                           index=self.data.index,
                                           name='Bearish_Divergence')
        }

    def calculate_volume_profile(self, bins: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculate volume profile to identify key accumulation/distribution zones.

        Volume profile shows the amount of volume traded at different price levels,
        helping identify where institutions have accumulated or distributed.

        Parameters:
        -----------
        bins : int, default=50
            Number of price bins

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with price levels and corresponding volumes
        """
        price_min = self.data['low'].min()
        price_max = self.data['high'].max()

        # Create price bins
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins)

        # Calculate typical price for each candle
        typical_price = (self.data['high'] + self.data['low'] +
                        self.data['close']) / 3

        # Distribute volume across bins
        for i in range(len(self.data)):
            price = typical_price.iloc[i]
            volume = self.data['volume'].iloc[i]

            # Find which bin this price falls into
            bin_idx = np.digitize(price, price_bins) - 1
            bin_idx = max(0, min(bins - 1, bin_idx))  # Ensure valid index

            volume_profile[bin_idx] += volume

        # Calculate bin centers
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

        return {
            'price_levels': bin_centers,
            'volume': volume_profile,
            'poc': bin_centers[np.argmax(volume_profile)],  # Point of Control
            'value_area': self._calculate_value_area(bin_centers, volume_profile)
        }

    def _calculate_value_area(self, prices: np.ndarray,
                              volumes: np.ndarray,
                              percentage: float = 0.7) -> Tuple[float, float]:
        """
        Calculate the value area (price range containing X% of volume).

        Parameters:
        -----------
        prices : np.ndarray
            Price levels
        volumes : np.ndarray
            Volume at each price level
        percentage : float, default=0.7
            Percentage of total volume (typically 70%)

        Returns:
        --------
        Tuple[float, float]
            Value area high and value area low
        """
        total_volume = volumes.sum()
        target_volume = total_volume * percentage

        # Sort by volume (descending)
        sorted_indices = np.argsort(volumes)[::-1]

        cumulative_volume = 0
        value_area_indices = []

        for idx in sorted_indices:
            cumulative_volume += volumes[idx]
            value_area_indices.append(idx)

            if cumulative_volume >= target_volume:
                break

        # Get price range
        va_prices = prices[value_area_indices]
        return (va_prices.max(), va_prices.min())

    def detect_whale_footprints(self, volume_threshold: float = 2.0,
                               spread_threshold: float = 0.02) -> pd.Series:
        """
        Detect institutional footprints in the data.

        Identifies specific candles or patterns that suggest large institutional
        orders, such as:
        - Very high volume with tight price range (absorption)
        - Large volume spikes at support/resistance

        Parameters:
        -----------
        volume_threshold : float, default=2.0
            Volume spike threshold (multiple of average)
        spread_threshold : float, default=0.02
            Maximum spread for absorption detection

        Returns:
        --------
        pd.Series
            Score indicating whale footprint strength
        """
        # Calculate average volume
        volume_ma = self.data['volume'].rolling(window=20).mean()
        volume_ratio = self.data['volume'] / volume_ma

        # Calculate candle spread (relative to close)
        spread = (self.data['high'] - self.data['low']) / self.data['close']

        # Absorption: high volume with tight spread
        absorption_score = np.where(
            (volume_ratio > volume_threshold) & (spread < spread_threshold),
            volume_ratio * 50,  # Scale up the score
            0
        )

        # Large volume at price extremes
        price_extremes = self._detect_price_extremes()
        extreme_volume = np.where(
            price_extremes & (volume_ratio > volume_threshold),
            volume_ratio * 30,
            0
        )

        # Combined footprint score
        footprint_score = absorption_score + extreme_volume

        return pd.Series(footprint_score, index=self.data.index,
                        name='Whale_Footprint')

    def _detect_price_extremes(self, window: int = 20) -> np.ndarray:
        """
        Detect price extremes (local highs and lows).

        Parameters:
        -----------
        window : int, default=20
            Window for extreme detection

        Returns:
        --------
        np.ndarray
            Boolean array indicating extremes
        """
        # Find local maxima
        high_peaks, _ = find_peaks(self.data['high'].values, distance=window//2)

        # Find local minima (invert for finding lows)
        low_peaks, _ = find_peaks(-self.data['low'].values, distance=window//2)

        # Create boolean array
        extremes = np.zeros(len(self.data), dtype=bool)
        extremes[high_peaks] = True
        extremes[low_peaks] = True

        return extremes

    def analyze_accumulation_phase(self, window: int = 60) -> pd.DataFrame:
        """
        Comprehensive accumulation phase analysis.

        Combines multiple indicators to provide a complete picture of
        accumulation activity over time.

        Parameters:
        -----------
        window : int, default=60
            Analysis window

        Returns:
        --------
        pd.DataFrame
            DataFrame with all accumulation metrics
        """
        result = self.data.copy()

        # Core indicators
        result['AD_Line'] = self.calculate_ad_line()
        result['OBV'] = self.calculate_obv()
        result['VWAP'] = self.calculate_vwap(window=window)

        # Detection scores
        result['Accumulation_Score'] = self.detect_quiet_accumulation(window=window)
        result['Distribution_Score'] = self.detect_distribution(window=window)
        result['Price_Suppression'] = self.detect_price_suppression(window=window//2)
        result['Whale_Footprint'] = self.detect_whale_footprints()

        # Divergences
        divergences = self.detect_smart_money_divergence(window=window//3)
        result['Bullish_Divergence'] = divergences['bullish_divergence']
        result['Bearish_Divergence'] = divergences['bearish_divergence']

        # Net accumulation signal (-100 to +100)
        # Positive = accumulation, Negative = distribution
        result['Net_Accumulation'] = (
            result['Accumulation_Score'] - result['Distribution_Score']
        )

        # Strong accumulation zones
        result['Strong_Accumulation'] = (
            (result['Accumulation_Score'] > 60) &
            (result['Price_Suppression'] == True) |
            (result['Bullish_Divergence'] == True)
        )

        # Strong distribution zones
        result['Strong_Distribution'] = (
            (result['Distribution_Score'] > 60) &
            (result['Bearish_Divergence'] == True)
        )

        return result

    def get_accumulation_summary(self) -> Dict[str, any]:
        """
        Get a summary of current accumulation/distribution state.

        Returns:
        --------
        Dict[str, any]
            Summary dictionary with key metrics
        """
        analysis = self.analyze_accumulation_phase()

        # Get most recent values (last 5 days average to smooth noise)
        recent = analysis.tail(5)

        summary = {
            'current_phase': None,
            'accumulation_score': recent['Accumulation_Score'].mean(),
            'distribution_score': recent['Distribution_Score'].mean(),
            'net_accumulation': recent['Net_Accumulation'].mean(),
            'price_suppression_active': recent['Price_Suppression'].any(),
            'bullish_divergence_active': recent['Bullish_Divergence'].any(),
            'bearish_divergence_active': recent['Bearish_Divergence'].any(),
            'whale_activity': recent['Whale_Footprint'].max(),
            'ad_line_trend': 'rising' if analysis['AD_Line'].iloc[-1] > analysis['AD_Line'].iloc[-10] else 'falling',
            'obv_trend': 'rising' if analysis['OBV'].iloc[-1] > analysis['OBV'].iloc[-10] else 'falling',
        }

        # Determine current phase
        if summary['net_accumulation'] > 30:
            summary['current_phase'] = 'STRONG_ACCUMULATION'
        elif summary['net_accumulation'] > 10:
            summary['current_phase'] = 'ACCUMULATION'
        elif summary['net_accumulation'] < -30:
            summary['current_phase'] = 'STRONG_DISTRIBUTION'
        elif summary['net_accumulation'] < -10:
            summary['current_phase'] = 'DISTRIBUTION'
        else:
            summary['current_phase'] = 'NEUTRAL'

        return summary


# Convenience functions
def detect_accumulation(data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Quick function to analyze accumulation patterns.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    window : int, default=60
        Analysis window

    Returns:
    --------
    pd.DataFrame
        Data with accumulation analysis
    """
    detector = WhaleAccumulationDetector(data)
    return detector.analyze_accumulation_phase(window=window)


def get_accumulation_summary(data: pd.DataFrame) -> Dict[str, any]:
    """
    Quick function to get accumulation summary.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data

    Returns:
    --------
    Dict[str, any]
        Accumulation summary
    """
    detector = WhaleAccumulationDetector(data)
    return detector.get_accumulation_summary()


def detect_quiet_accumulation(data: pd.DataFrame,
                              window: int = 30) -> pd.Series:
    """
    Quick function to detect quiet accumulation.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    window : int, default=30
        Analysis window

    Returns:
    --------
    pd.Series
        Accumulation scores
    """
    detector = WhaleAccumulationDetector(data)
    return detector.detect_quiet_accumulation(window=window)


if __name__ == "__main__":
    # Example usage
    print("Whale Accumulation/Distribution Detector")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    # Simulate accumulation pattern: sideways price with increasing volume
    base_price = 100
    price_noise = np.random.randn(200) * 2

    # Add accumulation phase (days 50-100)
    accumulation_phase = np.zeros(200)
    accumulation_phase[50:100] = np.linspace(0, 10, 50)

    close = base_price + price_noise + accumulation_phase
    high = close + np.abs(np.random.randn(200) * 1.5)
    low = close - np.abs(np.random.randn(200) * 1.5)
    open_price = close + np.random.randn(200) * 1

    # Volume increases during accumulation
    volume = np.random.randint(1000000, 2000000, 200)
    volume[50:100] = np.random.randint(2000000, 5000000, 50)  # Higher volume

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Initialize detector
    detector = WhaleAccumulationDetector(df)

    # Run analysis
    print("\n1. Analyzing accumulation patterns...")
    analysis = detector.analyze_accumulation_phase(window=30)

    print("\nRecent Accumulation Metrics:")
    print(analysis[['close', 'Accumulation_Score', 'Distribution_Score',
                   'Net_Accumulation']].tail(10))

    print("\n2. Getting accumulation summary...")
    summary = detector.get_accumulation_summary()

    print("\nAccumulation Summary:")
    print(f"Current Phase: {summary['current_phase']}")
    print(f"Net Accumulation Score: {summary['net_accumulation']:.2f}")
    print(f"Accumulation Score: {summary['accumulation_score']:.2f}")
    print(f"Distribution Score: {summary['distribution_score']:.2f}")
    print(f"Price Suppression: {summary['price_suppression_active']}")
    print(f"Bullish Divergence: {summary['bullish_divergence_active']}")
    print(f"A/D Line Trend: {summary['ad_line_trend']}")
    print(f"OBV Trend: {summary['obv_trend']}")

    print("\n3. Volume Profile Analysis...")
    volume_profile = detector.calculate_volume_profile(bins=20)
    print(f"Point of Control (POC): ${volume_profile['poc']:.2f}")
    print(f"Value Area High: ${volume_profile['value_area'][0]:.2f}")
    print(f"Value Area Low: ${volume_profile['value_area'][1]:.2f}")

    print("\n4. Smart Money Divergences...")
    divergences = detector.detect_smart_money_divergence(window=20)
    bullish_count = divergences['bullish_divergence'].sum()
    bearish_count = divergences['bearish_divergence'].sum()
    print(f"Bullish Divergences: {bullish_count}")
    print(f"Bearish Divergences: {bearish_count}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
