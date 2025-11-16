"""
Advanced Technical Indicators

This module provides advanced technical analysis indicators including:
- Volume-weighted indicators (VWAP, VWMA, Money Flow)
- Price action patterns (candlestick patterns, chart patterns)
- Support/Resistance levels (swing highs/lows, consolidation zones)
- Pivot points (Standard, Fibonacci, Woodie, Camarilla, DeMark)

Features:
- Efficient vectorized calculations using numpy and pandas
- Comprehensive candlestick pattern recognition
- Dynamic support/resistance detection
- Multiple pivot point calculation methods
- Configurable parameters for all indicators
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedIndicatorsError(Exception):
    """Base exception for Advanced Indicators errors."""
    pass


class InsufficientDataError(AdvancedIndicatorsError):
    """Raised when there is insufficient data for calculation."""
    pass


class VolumeWeightedIndicators:
    """
    Volume-weighted technical indicators.

    These indicators incorporate volume data to provide more robust signals
    by weighting price movements by their corresponding trading volume.
    """

    @staticmethod
    def vwap(
        df: pd.DataFrame,
        window: Optional[int] = None,
        price_col: str = 'Close'
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP is the ratio of the value traded to total volume traded over
        a particular time period. It's a measure of the average price
        weighted by volume.

        Args:
            df: DataFrame with OHLCV data
            window: Rolling window period (None for cumulative)
            price_col: Price column to use (default: 'Close')

        Returns:
            Series with VWAP values
        """
        # Use typical price for VWAP calculation
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        volume = df['Volume']

        if window is None:
            # Cumulative VWAP (resets daily if intraday data)
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
        else:
            # Rolling VWAP
            vwap = (
                (typical_price * volume).rolling(window=window).sum() /
                volume.rolling(window=window).sum()
            )

        return vwap

    @staticmethod
    def vwma(
        df: pd.DataFrame,
        window: int = 20,
        price_col: str = 'Close'
    ) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA).

        VWMA gives more weight to periods with higher volume.

        Args:
            df: DataFrame with OHLCV data
            window: Period for calculation
            price_col: Price column to use

        Returns:
            Series with VWMA values
        """
        price = df[price_col]
        volume = df['Volume']

        vwma = (
            (price * volume).rolling(window=window).sum() /
            volume.rolling(window=window).sum()
        )

        return vwma

    @staticmethod
    def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).

        MFI is a momentum indicator that uses price and volume to identify
        overbought or oversold conditions. Similar to RSI but volume-weighted.

        Args:
            df: DataFrame with OHLCV data
            window: Period for calculation (default: 14)

        Returns:
            Series with MFI values (0-100)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        # Identify positive and negative money flow
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        # Price increases -> positive flow
        price_diff = typical_price.diff()
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]

        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))

        return mfi

    @staticmethod
    def on_balance_volume(df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a cumulative indicator that adds volume on up days and
        subtracts volume on down days.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with OBV values
        """
        obv = pd.Series(0.0, index=df.index)
        close_diff = df['Close'].diff()

        obv[close_diff > 0] = df['Volume'][close_diff > 0]
        obv[close_diff < 0] = -df['Volume'][close_diff < 0]
        obv[close_diff == 0] = 0

        obv = obv.cumsum()

        return obv

    @staticmethod
    def volume_price_trend(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Price Trend (VPT).

        VPT is similar to OBV but uses percentage price changes.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with VPT values
        """
        price_pct_change = df['Close'].pct_change()
        vpt = (price_pct_change * df['Volume']).cumsum()

        return vpt

    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).

        CMF measures the amount of money flow volume over a specific period.

        Args:
            df: DataFrame with OHLCV data
            window: Period for calculation

        Returns:
            Series with CMF values
        """
        # Money Flow Multiplier
        mf_multiplier = (
            ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) /
            (df['High'] - df['Low'])
        )
        mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero

        # Money Flow Volume
        mf_volume = mf_multiplier * df['Volume']

        # CMF
        cmf = (
            mf_volume.rolling(window=window).sum() /
            df['Volume'].rolling(window=window).sum()
        )

        return cmf


class PriceActionPatterns:
    """
    Price action pattern recognition.

    Identifies candlestick patterns and chart patterns based on OHLC data.
    """

    @staticmethod
    def _body_size(df: pd.DataFrame) -> pd.Series:
        """Calculate candle body size."""
        return abs(df['Close'] - df['Open'])

    @staticmethod
    def _upper_shadow(df: pd.DataFrame) -> pd.Series:
        """Calculate upper shadow/wick size."""
        return df['High'] - df[['Open', 'Close']].max(axis=1)

    @staticmethod
    def _lower_shadow(df: pd.DataFrame) -> pd.Series:
        """Calculate lower shadow/wick size."""
        return df[['Open', 'Close']].min(axis=1) - df['Low']

    @staticmethod
    def _range(df: pd.DataFrame) -> pd.Series:
        """Calculate candle range (high - low)."""
        return df['High'] - df['Low']

    @staticmethod
    def doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Identify Doji candlestick pattern.

        A Doji occurs when open and close are virtually equal.
        Indicates indecision in the market.

        Args:
            df: DataFrame with OHLC data
            threshold: Body size threshold as % of range

        Returns:
            Boolean Series (True = Doji pattern)
        """
        body = PriceActionPatterns._body_size(df)
        candle_range = PriceActionPatterns._range(df)

        # Body is very small relative to range
        is_doji = body <= (candle_range * threshold)

        return is_doji

    @staticmethod
    def hammer(df: pd.DataFrame, body_ratio: float = 0.3) -> pd.Series:
        """
        Identify Hammer candlestick pattern.

        A Hammer has a small body at the top and a long lower shadow.
        Bullish reversal pattern at bottom of downtrend.

        Args:
            df: DataFrame with OHLC data
            body_ratio: Maximum body to range ratio

        Returns:
            Boolean Series (True = Hammer pattern)
        """
        body = PriceActionPatterns._body_size(df)
        lower_shadow = PriceActionPatterns._lower_shadow(df)
        upper_shadow = PriceActionPatterns._upper_shadow(df)
        candle_range = PriceActionPatterns._range(df)

        # Conditions for hammer
        is_hammer = (
            (body / candle_range <= body_ratio) &  # Small body
            (lower_shadow >= 2 * body) &  # Long lower shadow
            (upper_shadow <= 0.3 * body)  # Very small upper shadow
        )

        return is_hammer

    @staticmethod
    def shooting_star(df: pd.DataFrame, body_ratio: float = 0.3) -> pd.Series:
        """
        Identify Shooting Star candlestick pattern.

        A Shooting Star has a small body at the bottom and a long upper shadow.
        Bearish reversal pattern at top of uptrend.

        Args:
            df: DataFrame with OHLC data
            body_ratio: Maximum body to range ratio

        Returns:
            Boolean Series (True = Shooting Star pattern)
        """
        body = PriceActionPatterns._body_size(df)
        lower_shadow = PriceActionPatterns._lower_shadow(df)
        upper_shadow = PriceActionPatterns._upper_shadow(df)
        candle_range = PriceActionPatterns._range(df)

        # Conditions for shooting star
        is_shooting_star = (
            (body / candle_range <= body_ratio) &  # Small body
            (upper_shadow >= 2 * body) &  # Long upper shadow
            (lower_shadow <= 0.3 * body)  # Very small lower shadow
        )

        return is_shooting_star

    @staticmethod
    def engulfing_bullish(df: pd.DataFrame) -> pd.Series:
        """
        Identify Bullish Engulfing pattern.

        A bullish candle completely engulfs the previous bearish candle.
        Strong bullish reversal signal.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Bullish Engulfing pattern)
        """
        # Current candle is bullish
        current_bullish = df['Close'] > df['Open']

        # Previous candle is bearish
        prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)

        # Current candle's body engulfs previous candle's body
        engulfs = (
            (df['Open'] < df['Close'].shift(1)) &  # Opens below prev close
            (df['Close'] > df['Open'].shift(1))    # Closes above prev open
        )

        is_bullish_engulfing = current_bullish & prev_bearish & engulfs

        return is_bullish_engulfing

    @staticmethod
    def engulfing_bearish(df: pd.DataFrame) -> pd.Series:
        """
        Identify Bearish Engulfing pattern.

        A bearish candle completely engulfs the previous bullish candle.
        Strong bearish reversal signal.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Bearish Engulfing pattern)
        """
        # Current candle is bearish
        current_bearish = df['Close'] < df['Open']

        # Previous candle is bullish
        prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)

        # Current candle's body engulfs previous candle's body
        engulfs = (
            (df['Open'] > df['Close'].shift(1)) &  # Opens above prev close
            (df['Close'] < df['Open'].shift(1))    # Closes below prev open
        )

        is_bearish_engulfing = current_bearish & prev_bullish & engulfs

        return is_bearish_engulfing

    @staticmethod
    def morning_star(df: pd.DataFrame) -> pd.Series:
        """
        Identify Morning Star pattern (3-candle bullish reversal).

        Pattern: Large bearish candle, small-bodied candle (gap down),
        large bullish candle (gap up).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Morning Star pattern)
        """
        # First candle: Large bearish
        first_bearish = df['Close'].shift(2) < df['Open'].shift(2)
        first_large = (
            abs(df['Close'].shift(2) - df['Open'].shift(2)) >
            (df['High'].shift(2) - df['Low'].shift(2)) * 0.6
        )

        # Second candle: Small body
        second_small = (
            abs(df['Close'].shift(1) - df['Open'].shift(1)) <
            (df['High'].shift(1) - df['Low'].shift(1)) * 0.3
        )

        # Third candle: Large bullish
        third_bullish = df['Close'] > df['Open']
        third_large = (
            abs(df['Close'] - df['Open']) >
            (df['High'] - df['Low']) * 0.6
        )

        # Third candle closes above midpoint of first candle
        closes_high = df['Close'] > (
            (df['Open'].shift(2) + df['Close'].shift(2)) / 2
        )

        is_morning_star = (
            first_bearish & first_large &
            second_small &
            third_bullish & third_large & closes_high
        )

        return is_morning_star

    @staticmethod
    def evening_star(df: pd.DataFrame) -> pd.Series:
        """
        Identify Evening Star pattern (3-candle bearish reversal).

        Pattern: Large bullish candle, small-bodied candle (gap up),
        large bearish candle (gap down).

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Evening Star pattern)
        """
        # First candle: Large bullish
        first_bullish = df['Close'].shift(2) > df['Open'].shift(2)
        first_large = (
            abs(df['Close'].shift(2) - df['Open'].shift(2)) >
            (df['High'].shift(2) - df['Low'].shift(2)) * 0.6
        )

        # Second candle: Small body
        second_small = (
            abs(df['Close'].shift(1) - df['Open'].shift(1)) <
            (df['High'].shift(1) - df['Low'].shift(1)) * 0.3
        )

        # Third candle: Large bearish
        third_bearish = df['Close'] < df['Open']
        third_large = (
            abs(df['Close'] - df['Open']) >
            (df['High'] - df['Low']) * 0.6
        )

        # Third candle closes below midpoint of first candle
        closes_low = df['Close'] < (
            (df['Open'].shift(2) + df['Close'].shift(2)) / 2
        )

        is_evening_star = (
            first_bullish & first_large &
            second_small &
            third_bearish & third_large & closes_low
        )

        return is_evening_star

    @staticmethod
    def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """
        Identify Three White Soldiers pattern (strong bullish continuation).

        Pattern: Three consecutive bullish candles with higher closes,
        each opening within previous candle's body.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Three White Soldiers pattern)
        """
        # All three candles are bullish
        all_bullish = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'].shift(2) > df['Open'].shift(2))
        )

        # Each close is higher than previous
        higher_closes = (
            (df['Close'] > df['Close'].shift(1)) &
            (df['Close'].shift(1) > df['Close'].shift(2))
        )

        # Each opens within previous body
        opens_in_body = (
            (df['Open'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Open'].shift(1) > df['Open'].shift(2)) &
            (df['Open'].shift(1) < df['Close'].shift(2))
        )

        is_three_white_soldiers = all_bullish & higher_closes & opens_in_body

        return is_three_white_soldiers

    @staticmethod
    def three_black_crows(df: pd.DataFrame) -> pd.Series:
        """
        Identify Three Black Crows pattern (strong bearish continuation).

        Pattern: Three consecutive bearish candles with lower closes,
        each opening within previous candle's body.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Boolean Series (True = Three Black Crows pattern)
        """
        # All three candles are bearish
        all_bearish = (
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'].shift(2) < df['Open'].shift(2))
        )

        # Each close is lower than previous
        lower_closes = (
            (df['Close'] < df['Close'].shift(1)) &
            (df['Close'].shift(1) < df['Close'].shift(2))
        )

        # Each opens within previous body
        opens_in_body = (
            (df['Open'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Open'].shift(1) < df['Open'].shift(2)) &
            (df['Open'].shift(1) > df['Close'].shift(2))
        )

        is_three_black_crows = all_bearish & lower_closes & opens_in_body

        return is_three_black_crows


class SupportResistance:
    """
    Support and Resistance level detection.

    Identifies key price levels where the market has historically
    shown a tendency to reverse or consolidate.
    """

    @staticmethod
    def swing_highs_lows(
        df: pd.DataFrame,
        window: int = 5,
        order: int = 5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and swing lows.

        Swing highs/lows are local extrema that can act as support/resistance.

        Args:
            df: DataFrame with OHLC data
            window: Window for rolling max/min
            order: Order for extrema detection (higher = more significant)

        Returns:
            Tuple of (swing_highs, swing_lows) as Series
        """
        # Use scipy to find local maxima and minima
        high_indices = argrelextrema(
            df['High'].values,
            np.greater,
            order=order
        )[0]

        low_indices = argrelextrema(
            df['Low'].values,
            np.less,
            order=order
        )[0]

        # Create series with swing points
        swing_highs = pd.Series(np.nan, index=df.index)
        swing_lows = pd.Series(np.nan, index=df.index)

        swing_highs.iloc[high_indices] = df['High'].iloc[high_indices]
        swing_lows.iloc[low_indices] = df['Low'].iloc[low_indices]

        return swing_highs, swing_lows

    @staticmethod
    def support_resistance_levels(
        df: pd.DataFrame,
        num_levels: int = 5,
        lookback: int = 100
    ) -> Dict[str, List[float]]:
        """
        Identify key support and resistance levels using clustering.

        Args:
            df: DataFrame with OHLC data
            num_levels: Number of support/resistance levels to identify
            lookback: Number of periods to look back

        Returns:
            Dictionary with 'support' and 'resistance' level lists
        """
        if len(df) < lookback:
            lookback = len(df)

        recent_data = df.tail(lookback)

        # Get swing points
        swing_highs, swing_lows = SupportResistance.swing_highs_lows(
            recent_data,
            order=3
        )

        # Extract valid swing points
        resistance_points = swing_highs.dropna().values
        support_points = swing_lows.dropna().values

        # Cluster nearby levels
        def cluster_levels(points, n_clusters):
            if len(points) == 0:
                return []

            if len(points) <= n_clusters:
                return sorted(points.tolist())

            # Simple clustering by sorting and grouping
            sorted_points = np.sort(points)

            # Use histogram to find clusters
            hist, bin_edges = np.histogram(sorted_points, bins=n_clusters)

            # Get cluster centers
            levels = []
            for i in range(len(hist)):
                if hist[i] > 0:
                    # Find points in this bin
                    mask = (sorted_points >= bin_edges[i]) & (
                        sorted_points < bin_edges[i + 1]
                    )
                    if mask.any():
                        levels.append(float(sorted_points[mask].mean()))

            return sorted(levels)

        resistance_levels = cluster_levels(resistance_points, num_levels)
        support_levels = cluster_levels(support_points, num_levels)

        return {
            'resistance': resistance_levels,
            'support': support_levels
        }

    @staticmethod
    def consolidation_zones(
        df: pd.DataFrame,
        window: int = 20,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Identify consolidation zones (price ranges with low volatility).

        Args:
            df: DataFrame with OHLC data
            window: Rolling window for volatility calculation
            threshold: Maximum price range as % for consolidation

        Returns:
            DataFrame with consolidation zone information
        """
        # Calculate price range
        price_range = (df['High'] - df['Low']) / df['Close']

        # Rolling average of price range
        avg_range = price_range.rolling(window=window).mean()

        # Identify consolidation (low range periods)
        is_consolidation = price_range < (avg_range * threshold)

        # Get zone boundaries
        zones = []
        in_zone = False
        zone_start = None

        for i, (idx, val) in enumerate(is_consolidation.items()):
            if val and not in_zone:
                # Start of consolidation zone
                zone_start = idx
                in_zone = True
            elif not val and in_zone:
                # End of consolidation zone
                zone_data = df.loc[zone_start:idx]
                zones.append({
                    'start': zone_start,
                    'end': idx,
                    'high': zone_data['High'].max(),
                    'low': zone_data['Low'].min(),
                    'midpoint': (
                        zone_data['High'].max() + zone_data['Low'].min()
                    ) / 2,
                    'duration': len(zone_data)
                })
                in_zone = False

        return pd.DataFrame(zones)


class PivotPoints:
    """
    Pivot Point calculations.

    Pivot points are price levels that can act as support or resistance.
    Multiple calculation methods are provided.
    """

    @staticmethod
    def standard_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Standard Pivot Points.

        Most common pivot point calculation method.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with pivot levels (PP, R1-R3, S1-S3)
        """
        result = pd.DataFrame(index=df.index)

        # Use previous period's high, low, close
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)

        # Pivot Point
        result['PP'] = (high + low + close) / 3

        # Support levels
        result['S1'] = 2 * result['PP'] - high
        result['S2'] = result['PP'] - (high - low)
        result['S3'] = low - 2 * (high - result['PP'])

        # Resistance levels
        result['R1'] = 2 * result['PP'] - low
        result['R2'] = result['PP'] + (high - low)
        result['R3'] = high + 2 * (result['PP'] - low)

        return result

    @staticmethod
    def fibonacci_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Pivot Points.

        Uses Fibonacci ratios (0.382, 0.618, 1.000) for levels.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with Fibonacci pivot levels
        """
        result = pd.DataFrame(index=df.index)

        # Use previous period's high, low, close
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)

        # Pivot Point (same as standard)
        result['PP'] = (high + low + close) / 3

        # Range
        range_hl = high - low

        # Support levels
        result['S1'] = result['PP'] - 0.382 * range_hl
        result['S2'] = result['PP'] - 0.618 * range_hl
        result['S3'] = result['PP'] - 1.000 * range_hl

        # Resistance levels
        result['R1'] = result['PP'] + 0.382 * range_hl
        result['R2'] = result['PP'] + 0.618 * range_hl
        result['R3'] = result['PP'] + 1.000 * range_hl

        return result

    @staticmethod
    def woodie_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Woodie Pivot Points.

        Gives more weight to the closing price.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with Woodie pivot levels
        """
        result = pd.DataFrame(index=df.index)

        # Use previous period's high, low, close
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)

        # Pivot Point (different from standard)
        result['PP'] = (high + low + 2 * close) / 4

        # Support levels
        result['S1'] = 2 * result['PP'] - high
        result['S2'] = result['PP'] - high + low
        result['S3'] = low - 2 * (high - result['PP'])
        result['S4'] = result['S3'] - (high - low)

        # Resistance levels
        result['R1'] = 2 * result['PP'] - low
        result['R2'] = result['PP'] + high - low
        result['R3'] = high + 2 * (result['PP'] - low)
        result['R4'] = result['R3'] + (high - low)

        return result

    @staticmethod
    def camarilla_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Camarilla Pivot Points.

        Uses multipliers of the previous day's range.
        Popular for intraday trading.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with Camarilla pivot levels
        """
        result = pd.DataFrame(index=df.index)

        # Use previous period's high, low, close
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)

        # Range
        range_hl = high - low

        # Pivot Point
        result['PP'] = (high + low + close) / 3

        # Support levels (L1-L4)
        result['S1'] = close - range_hl * 1.1 / 12
        result['S2'] = close - range_hl * 1.1 / 6
        result['S3'] = close - range_hl * 1.1 / 4
        result['S4'] = close - range_hl * 1.1 / 2

        # Resistance levels (H1-H4)
        result['R1'] = close + range_hl * 1.1 / 12
        result['R2'] = close + range_hl * 1.1 / 6
        result['R3'] = close + range_hl * 1.1 / 4
        result['R4'] = close + range_hl * 1.1 / 2

        return result

    @staticmethod
    def demark_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DeMark Pivot Points.

        Conditional calculation based on open/close relationship.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with DeMark pivot levels
        """
        result = pd.DataFrame(index=df.index)

        # Use previous period's high, low, close, open
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)
        open_price = df['Open'].shift(1)

        # X value depends on open/close relationship
        x = pd.Series(0.0, index=df.index)

        # If close < open: X = H + 2L + C
        x[close < open_price] = (
            high[close < open_price] +
            2 * low[close < open_price] +
            close[close < open_price]
        )

        # If close > open: X = 2H + L + C
        x[close > open_price] = (
            2 * high[close > open_price] +
            low[close > open_price] +
            close[close > open_price]
        )

        # If close == open: X = H + L + 2C
        x[close == open_price] = (
            high[close == open_price] +
            low[close == open_price] +
            2 * close[close == open_price]
        )

        # Pivot Point
        result['PP'] = x / 4

        # Support
        result['S1'] = x / 2 - high

        # Resistance
        result['R1'] = x / 2 - low

        return result


class AdvancedIndicators:
    """
    Main class combining all advanced technical indicators.

    Provides a unified interface for calculating volume-weighted indicators,
    price action patterns, support/resistance levels, and pivot points.
    """

    def __init__(self):
        """Initialize Advanced Indicators calculator."""
        self.volume_weighted = VolumeWeightedIndicators()
        self.price_action = PriceActionPatterns()
        self.support_resistance = SupportResistance()
        self.pivots = PivotPoints()
        logger.info("AdvancedIndicators initialized")

    def calculate_all_volume_indicators(
        self,
        df: pd.DataFrame,
        vwap_window: Optional[int] = None,
        vwma_window: int = 20,
        mfi_window: int = 14,
        cmf_window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate all volume-weighted indicators.

        Args:
            df: DataFrame with OHLCV data
            vwap_window: VWAP rolling window (None for cumulative)
            vwma_window: VWMA period
            mfi_window: MFI period
            cmf_window: CMF period

        Returns:
            DataFrame with all volume indicators
        """
        result = df.copy()

        try:
            result['VWAP'] = self.volume_weighted.vwap(df, vwap_window)
            result['VWMA'] = self.volume_weighted.vwma(df, vwma_window)
            result['MFI'] = self.volume_weighted.money_flow_index(df, mfi_window)
            result['OBV'] = self.volume_weighted.on_balance_volume(df)
            result['VPT'] = self.volume_weighted.volume_price_trend(df)
            result['CMF'] = self.volume_weighted.chaikin_money_flow(df, cmf_window)

            logger.info("All volume indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            raise

        return result

    def calculate_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all price action patterns.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with pattern columns (True/False)
        """
        result = df.copy()

        try:
            # Single candle patterns
            result['Doji'] = self.price_action.doji(df)
            result['Hammer'] = self.price_action.hammer(df)
            result['ShootingStar'] = self.price_action.shooting_star(df)

            # Two candle patterns
            result['BullishEngulfing'] = self.price_action.engulfing_bullish(df)
            result['BearishEngulfing'] = self.price_action.engulfing_bearish(df)

            # Three candle patterns
            result['MorningStar'] = self.price_action.morning_star(df)
            result['EveningStar'] = self.price_action.evening_star(df)
            result['ThreeWhiteSoldiers'] = self.price_action.three_white_soldiers(df)
            result['ThreeBlackCrows'] = self.price_action.three_black_crows(df)

            logger.info("All price action patterns identified successfully")
        except Exception as e:
            logger.error(f"Error identifying patterns: {str(e)}")
            raise

        return result

    def calculate_all_pivots(
        self,
        df: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate pivot points using multiple methods.

        Args:
            df: DataFrame with OHLC data
            methods: List of methods to use. Options:
                    ['standard', 'fibonacci', 'woodie', 'camarilla', 'demark']
                    If None, calculates all methods.

        Returns:
            DataFrame with pivot points from all requested methods
        """
        if methods is None:
            methods = ['standard', 'fibonacci', 'woodie', 'camarilla', 'demark']

        result = df.copy()

        try:
            if 'standard' in methods:
                standard = self.pivots.standard_pivots(df)
                for col in standard.columns:
                    result[f'Std_{col}'] = standard[col]

            if 'fibonacci' in methods:
                fibonacci = self.pivots.fibonacci_pivots(df)
                for col in fibonacci.columns:
                    result[f'Fib_{col}'] = fibonacci[col]

            if 'woodie' in methods:
                woodie = self.pivots.woodie_pivots(df)
                for col in woodie.columns:
                    result[f'Wood_{col}'] = woodie[col]

            if 'camarilla' in methods:
                camarilla = self.pivots.camarilla_pivots(df)
                for col in camarilla.columns:
                    result[f'Cam_{col}'] = camarilla[col]

            if 'demark' in methods:
                demark = self.pivots.demark_pivots(df)
                for col in demark.columns:
                    result[f'DeM_{col}'] = demark[col]

            logger.info(f"Pivot points calculated using methods: {methods}")
        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            raise

        return result

    def get_support_resistance(
        self,
        df: pd.DataFrame,
        num_levels: int = 5,
        lookback: int = 100
    ) -> Dict[str, any]:
        """
        Get comprehensive support and resistance analysis.

        Args:
            df: DataFrame with OHLC data
            num_levels: Number of key levels to identify
            lookback: Number of periods to analyze

        Returns:
            Dictionary with swing points, levels, and consolidation zones
        """
        try:
            # Swing highs and lows
            swing_highs, swing_lows = self.support_resistance.swing_highs_lows(df)

            # Key support/resistance levels
            levels = self.support_resistance.support_resistance_levels(
                df,
                num_levels,
                lookback
            )

            # Consolidation zones
            zones = self.support_resistance.consolidation_zones(df)

            result = {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'levels': levels,
                'consolidation_zones': zones
            }

            logger.info("Support/resistance analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {str(e)}")
            raise


def example_usage():
    """Example usage of Advanced Indicators."""

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Generate realistic OHLCV data
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100) * 1.5)
    low = close - np.abs(np.random.randn(100) * 1.5)
    open_price = close + np.random.randn(100) * 1
    volume = np.random.randint(1000000, 10000000, 100)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    # Initialize indicators
    indicators = AdvancedIndicators()

    print("=== Volume-Weighted Indicators ===")
    vol_indicators = indicators.calculate_all_volume_indicators(df)
    print(vol_indicators[['Close', 'VWAP', 'VWMA', 'MFI', 'OBV', 'CMF']].tail())

    print("\n=== Price Action Patterns ===")
    patterns = indicators.calculate_all_patterns(df)
    pattern_cols = [
        'Doji', 'Hammer', 'ShootingStar',
        'BullishEngulfing', 'BearishEngulfing'
    ]
    print(patterns[pattern_cols].tail(20))

    print("\n=== Pivot Points (Standard) ===")
    pivots = indicators.calculate_all_pivots(df, methods=['standard'])
    pivot_cols = [col for col in pivots.columns if 'Std_' in col]
    print(pivots[pivot_cols].tail())

    print("\n=== Support/Resistance Analysis ===")
    sr_analysis = indicators.get_support_resistance(df)
    print("Key Levels:")
    print(f"Resistance: {sr_analysis['levels']['resistance']}")
    print(f"Support: {sr_analysis['levels']['support']}")
    print(f"\nConsolidation Zones: {len(sr_analysis['consolidation_zones'])}")


if __name__ == "__main__":
    example_usage()
