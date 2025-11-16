"""
Trend Indicators Module for BIST Trading System

This module provides comprehensive trend analysis indicators for technical analysis
of financial time series data, specifically designed for the Borsa Istanbul (BIST)
equity market.

Indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - WMA (Weighted Moving Average)
    - HMA (Hull Moving Average)
    - Ichimoku Cloud (complete cloud components)

Features:
    - Multiple timeframe support
    - Pandas-ta and TA-Lib backend support
    - Vectorized operations for performance
    - Comprehensive parameter validation
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import technical analysis libraries
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available. Install with: pip install pandas-ta")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Install with: pip install ta-lib")


class TrendIndicators:
    """
    Comprehensive trend indicator calculator for technical analysis.

    This class provides methods to calculate various trend indicators
    with support for multiple timeframes and library backends.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        price_column: str = 'close',
        backend: str = 'auto'
    ):
        """
        Initialize the trend indicator calculator.

        Args:
            data: DataFrame with OHLCV columns and datetime index
            price_column: Column to use for price-based calculations (default: 'close')
            backend: Library to use ('pandas_ta', 'talib', 'auto', 'numpy')
                    'auto' will prefer pandas_ta, then talib, then numpy
        """
        self.data = data
        self.price_column = price_column
        self.backend = self._validate_backend(backend)

    def _validate_backend(self, backend: str) -> str:
        """
        Validate and select the best available backend.

        Args:
            backend: Requested backend

        Returns:
            Selected backend name
        """
        if backend == 'auto':
            if PANDAS_TA_AVAILABLE:
                return 'pandas_ta'
            elif TALIB_AVAILABLE:
                return 'talib'
            else:
                logger.info("Using numpy backend (pandas-ta and TA-Lib not available)")
                return 'numpy'

        if backend == 'pandas_ta' and not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta requested but not available")
        if backend == 'talib' and not TALIB_AVAILABLE:
            raise ImportError("TA-Lib requested but not available")

        return backend

    def _get_price_series(
        self,
        data: Optional[pd.DataFrame] = None,
        column: Optional[str] = None
    ) -> pd.Series:
        """
        Extract price series from data.

        Args:
            data: DataFrame to use (uses self.data if None)
            column: Column name to extract (uses self.price_column if None)

        Returns:
            Price series
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        if column is None:
            column = self.price_column

        # Handle case-insensitive column names
        data_cols_lower = {col.lower(): col for col in data.columns}
        column_lower = column.lower()

        if column_lower in data_cols_lower:
            return data[data_cols_lower[column_lower]]
        else:
            raise ValueError(f"Column '{column}' not found in data")

    def sma(
        self,
        data: Optional[pd.DataFrame] = None,
        period: int = 20,
        column: Optional[str] = None,
        timeframes: Optional[List[int]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Simple Moving Average (SMA).

        SMA is the arithmetic mean of prices over a specified period.
        It's the most basic trend indicator, smoothing out price fluctuations.

        Args:
            data: DataFrame with price data (uses self.data if None)
            period: Number of periods for MA calculation (default: 20)
            column: Column to use for calculation (uses self.price_column if None)
            timeframes: List of periods to calculate (overrides period if provided)
                       Example: [10, 20, 50, 200]

        Returns:
            Series with SMA values or DataFrame if multiple timeframes

        Example:
            >>> calculator = TrendIndicators(data)
            >>> sma_20 = calculator.sma(period=20)
            >>> sma_multi = calculator.sma(timeframes=[10, 20, 50, 200])
        """
        price = self._get_price_series(data, column)

        if timeframes is not None:
            result = pd.DataFrame(index=price.index)
            for tf in timeframes:
                result[f'SMA_{tf}'] = self._calculate_sma(price, tf)
            return result

        return self._calculate_sma(price, period)

    def _calculate_sma(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using selected backend."""
        if self.backend == 'pandas_ta':
            return ta.sma(price, length=period)
        elif self.backend == 'talib':
            return pd.Series(
                talib.SMA(price.values, timeperiod=period),
                index=price.index,
                name=f'SMA_{period}'
            )
        else:  # numpy backend
            return price.rolling(window=period, min_periods=1).mean()

    def ema(
        self,
        data: Optional[pd.DataFrame] = None,
        period: int = 20,
        column: Optional[str] = None,
        timeframes: Optional[List[int]] = None,
        adjust: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Exponential Moving Average (EMA).

        EMA gives more weight to recent prices, making it more responsive
        to new information compared to SMA.

        Args:
            data: DataFrame with price data (uses self.data if None)
            period: Number of periods for MA calculation (default: 20)
            column: Column to use for calculation (uses self.price_column if None)
            timeframes: List of periods to calculate (overrides period if provided)
            adjust: Use adjusted EMA calculation (default: True)

        Returns:
            Series with EMA values or DataFrame if multiple timeframes

        Example:
            >>> calculator = TrendIndicators(data)
            >>> ema_12 = calculator.ema(period=12)
            >>> ema_multi = calculator.ema(timeframes=[12, 26, 50])
        """
        price = self._get_price_series(data, column)

        if timeframes is not None:
            result = pd.DataFrame(index=price.index)
            for tf in timeframes:
                result[f'EMA_{tf}'] = self._calculate_ema(price, tf, adjust)
            return result

        return self._calculate_ema(price, period, adjust)

    def _calculate_ema(
        self,
        price: pd.Series,
        period: int,
        adjust: bool = True
    ) -> pd.Series:
        """Calculate EMA using selected backend."""
        if self.backend == 'pandas_ta':
            return ta.ema(price, length=period, adjust=adjust)
        elif self.backend == 'talib':
            return pd.Series(
                talib.EMA(price.values, timeperiod=period),
                index=price.index,
                name=f'EMA_{period}'
            )
        else:  # numpy backend
            return price.ewm(span=period, adjust=adjust, min_periods=1).mean()

    def wma(
        self,
        data: Optional[pd.DataFrame] = None,
        period: int = 20,
        column: Optional[str] = None,
        timeframes: Optional[List[int]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Weighted Moving Average (WMA).

        WMA assigns linearly increasing weights to recent prices.
        Most recent price gets weight n, previous gets n-1, etc.

        Args:
            data: DataFrame with price data (uses self.data if None)
            period: Number of periods for MA calculation (default: 20)
            column: Column to use for calculation (uses self.price_column if None)
            timeframes: List of periods to calculate (overrides period if provided)

        Returns:
            Series with WMA values or DataFrame if multiple timeframes

        Example:
            >>> calculator = TrendIndicators(data)
            >>> wma_20 = calculator.wma(period=20)
            >>> wma_multi = calculator.wma(timeframes=[10, 20, 30])
        """
        price = self._get_price_series(data, column)

        if timeframes is not None:
            result = pd.DataFrame(index=price.index)
            for tf in timeframes:
                result[f'WMA_{tf}'] = self._calculate_wma(price, tf)
            return result

        return self._calculate_wma(price, period)

    def _calculate_wma(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate WMA using selected backend."""
        if self.backend == 'pandas_ta':
            return ta.wma(price, length=period)
        elif self.backend == 'talib':
            return pd.Series(
                talib.WMA(price.values, timeperiod=period),
                index=price.index,
                name=f'WMA_{period}'
            )
        else:  # numpy backend
            weights = np.arange(1, period + 1)

            def weighted_mean(x):
                if len(x) < period:
                    # Use available data with adjusted weights
                    w = weights[:len(x)]
                    return np.sum(x * w) / np.sum(w)
                return np.sum(x * weights) / np.sum(weights)

            return price.rolling(window=period, min_periods=1).apply(
                weighted_mean, raw=True
            )

    def hma(
        self,
        data: Optional[pd.DataFrame] = None,
        period: int = 20,
        column: Optional[str] = None,
        timeframes: Optional[List[int]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Hull Moving Average (HMA).

        HMA is designed to reduce lag while maintaining smoothness.
        Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

        Args:
            data: DataFrame with price data (uses self.data if None)
            period: Number of periods for MA calculation (default: 20)
            column: Column to use for calculation (uses self.price_column if None)
            timeframes: List of periods to calculate (overrides period if provided)

        Returns:
            Series with HMA values or DataFrame if multiple timeframes

        Example:
            >>> calculator = TrendIndicators(data)
            >>> hma_20 = calculator.hma(period=20)
            >>> hma_multi = calculator.hma(timeframes=[9, 16, 25])
        """
        price = self._get_price_series(data, column)

        if timeframes is not None:
            result = pd.DataFrame(index=price.index)
            for tf in timeframes:
                result[f'HMA_{tf}'] = self._calculate_hma(price, tf)
            return result

        return self._calculate_hma(price, period)

    def _calculate_hma(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate HMA using selected backend or custom implementation."""
        if self.backend == 'pandas_ta':
            return ta.hma(price, length=period)
        else:
            # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))

            wma_half = self._calculate_wma(price, half_period)
            wma_full = self._calculate_wma(price, period)

            raw_hma = 2 * wma_half - wma_full
            hma = self._calculate_wma(raw_hma, sqrt_period)

            return pd.Series(hma, index=price.index, name=f'HMA_{period}')

    def ichimoku(
        self,
        data: Optional[pd.DataFrame] = None,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud (Ichimoku Kinko Hyo).

        A comprehensive indicator system that provides support/resistance,
        trend direction, and momentum in a single glance.

        Components:
            - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward
            - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
            - Chikou Span (Lagging Span): Close price shifted backward 26 periods

        Args:
            data: DataFrame with OHLCV data (uses self.data if None)
            tenkan_period: Period for Tenkan-sen (default: 9)
            kijun_period: Period for Kijun-sen (default: 26)
            senkou_b_period: Period for Senkou Span B (default: 52)
            displacement: Periods to shift Senkou spans forward / Chikou backward (default: 26)

        Returns:
            DataFrame with all Ichimoku components

        Example:
            >>> calculator = TrendIndicators(data)
            >>> ichimoku = calculator.ichimoku()
            >>> # Access components
            >>> tenkan = ichimoku['tenkan_sen']
            >>> cloud_top = ichimoku['senkou_span_a']
            >>> cloud_bottom = ichimoku['senkou_span_b']
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        # Normalize column names
        data_cols_lower = {col.lower(): col for col in data.columns}

        # Get high and low columns
        high_col = data_cols_lower.get('high')
        low_col = data_cols_lower.get('low')
        close_col = data_cols_lower.get('close')

        if high_col is None or low_col is None or close_col is None:
            raise ValueError("Data must contain 'high', 'low', and 'close' columns")

        high = data[high_col]
        low = data[low_col]
        close = data[close_col]

        if self.backend == 'pandas_ta':
            # Use pandas-ta implementation
            ichimoku_df = ta.ichimoku(
                high=high,
                low=low,
                close=close,
                tenkan=tenkan_period,
                kijun=kijun_period,
                senkou=senkou_b_period
            )[0]  # pandas-ta returns tuple

            return ichimoku_df
        else:
            # Custom implementation
            result = pd.DataFrame(index=data.index)

            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan_high = high.rolling(window=tenkan_period, min_periods=1).max()
            tenkan_low = low.rolling(window=tenkan_period, min_periods=1).min()
            result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun_high = high.rolling(window=kijun_period, min_periods=1).max()
            kijun_low = low.rolling(window=kijun_period, min_periods=1).min()
            result['kijun_sen'] = (kijun_high + kijun_low) / 2

            # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward
            senkou_span_a = (result['tenkan_sen'] + result['kijun_sen']) / 2
            result['senkou_span_a'] = senkou_span_a.shift(displacement)

            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
            senkou_b_high = high.rolling(window=senkou_b_period, min_periods=1).max()
            senkou_b_low = low.rolling(window=senkou_b_period, min_periods=1).min()
            senkou_span_b = (senkou_b_high + senkou_b_low) / 2
            result['senkou_span_b'] = senkou_span_b.shift(displacement)

            # Chikou Span (Lagging Span): Close price shifted backward
            result['chikou_span'] = close.shift(-displacement)

            # Additional helper columns
            result['cloud_top'] = result[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            result['cloud_bottom'] = result[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            result['cloud_green'] = result['senkou_span_a'] > result['senkou_span_b']

            return result

    def multi_timeframe_analysis(
        self,
        data: Optional[pd.DataFrame] = None,
        indicators: Optional[List[str]] = None,
        timeframes: Optional[Dict[str, List[int]]] = None,
        column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators across multiple timeframes.

        This is a convenience method for comprehensive trend analysis
        across various time horizons.

        Args:
            data: DataFrame with price data (uses self.data if None)
            indicators: List of indicators to calculate
                       Options: ['sma', 'ema', 'wma', 'hma']
                       Default: all indicators
            timeframes: Dict mapping indicator names to list of periods
                       Example: {'sma': [20, 50, 200], 'ema': [12, 26]}
                       Default: standard timeframes for each indicator
            column: Column to use for calculation (uses self.price_column if None)

        Returns:
            DataFrame with all calculated indicators

        Example:
            >>> calculator = TrendIndicators(data)
            >>> analysis = calculator.multi_timeframe_analysis(
            ...     indicators=['sma', 'ema'],
            ...     timeframes={'sma': [20, 50, 200], 'ema': [12, 26, 50]}
            ... )
        """
        if data is None:
            data = self.data
        if data is None:
            raise ValueError("No data provided")

        # Default indicators
        if indicators is None:
            indicators = ['sma', 'ema', 'wma', 'hma']

        # Default timeframes
        if timeframes is None:
            timeframes = {
                'sma': [10, 20, 50, 100, 200],
                'ema': [8, 12, 21, 26, 50],
                'wma': [10, 20, 30],
                'hma': [9, 16, 25]
            }

        result = pd.DataFrame(index=data.index)

        # Calculate each indicator
        for indicator in indicators:
            if indicator not in timeframes:
                logger.warning(f"No timeframes specified for {indicator}, skipping")
                continue

            try:
                if indicator == 'sma':
                    indicator_df = self.sma(
                        data=data,
                        timeframes=timeframes['sma'],
                        column=column
                    )
                elif indicator == 'ema':
                    indicator_df = self.ema(
                        data=data,
                        timeframes=timeframes['ema'],
                        column=column
                    )
                elif indicator == 'wma':
                    indicator_df = self.wma(
                        data=data,
                        timeframes=timeframes['wma'],
                        column=column
                    )
                elif indicator == 'hma':
                    indicator_df = self.hma(
                        data=data,
                        timeframes=timeframes['hma'],
                        column=column
                    )
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
                    continue

                # Add to result
                if isinstance(indicator_df, pd.DataFrame):
                    result = pd.concat([result, indicator_df], axis=1)
                else:
                    result[indicator_df.name] = indicator_df

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                continue

        return result

    def get_trend_signal(
        self,
        data: Optional[pd.DataFrame] = None,
        fast_period: int = 10,
        slow_period: int = 20,
        indicator: str = 'ema'
    ) -> pd.Series:
        """
        Generate trend signals based on moving average crossovers.

        Args:
            data: DataFrame with price data (uses self.data if None)
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            indicator: Type of MA to use ('sma', 'ema', 'wma', 'hma')

        Returns:
            Series with signals: 1 (bullish), -1 (bearish), 0 (neutral)

        Example:
            >>> calculator = TrendIndicators(data)
            >>> signals = calculator.get_trend_signal(fast_period=12, slow_period=26, indicator='ema')
        """
        price = self._get_price_series(data)

        # Calculate fast and slow MAs
        if indicator == 'sma':
            fast_ma = self._calculate_sma(price, fast_period)
            slow_ma = self._calculate_sma(price, slow_period)
        elif indicator == 'ema':
            fast_ma = self._calculate_ema(price, fast_period)
            slow_ma = self._calculate_ema(price, slow_period)
        elif indicator == 'wma':
            fast_ma = self._calculate_wma(price, fast_period)
            slow_ma = self._calculate_wma(price, slow_period)
        elif indicator == 'hma':
            fast_ma = self._calculate_hma(price, fast_period)
            slow_ma = self._calculate_hma(price, slow_period)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")

        # Generate signals
        signal = pd.Series(0, index=price.index, name='trend_signal')
        signal[fast_ma > slow_ma] = 1  # Bullish
        signal[fast_ma < slow_ma] = -1  # Bearish

        return signal


# Convenience functions for standalone use

def calculate_sma(
    data: pd.DataFrame,
    period: int = 20,
    column: str = 'close',
    timeframes: Optional[List[int]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convenience function to calculate SMA.

    Args:
        data: DataFrame with price data
        period: Number of periods (default: 20)
        column: Column to use (default: 'close')
        timeframes: List of periods for multiple timeframes

    Returns:
        Series or DataFrame with SMA values
    """
    calculator = TrendIndicators(data, price_column=column)
    return calculator.sma(period=period, timeframes=timeframes)


def calculate_ema(
    data: pd.DataFrame,
    period: int = 20,
    column: str = 'close',
    timeframes: Optional[List[int]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convenience function to calculate EMA.

    Args:
        data: DataFrame with price data
        period: Number of periods (default: 20)
        column: Column to use (default: 'close')
        timeframes: List of periods for multiple timeframes

    Returns:
        Series or DataFrame with EMA values
    """
    calculator = TrendIndicators(data, price_column=column)
    return calculator.ema(period=period, timeframes=timeframes)


def calculate_wma(
    data: pd.DataFrame,
    period: int = 20,
    column: str = 'close',
    timeframes: Optional[List[int]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convenience function to calculate WMA.

    Args:
        data: DataFrame with price data
        period: Number of periods (default: 20)
        column: Column to use (default: 'close')
        timeframes: List of periods for multiple timeframes

    Returns:
        Series or DataFrame with WMA values
    """
    calculator = TrendIndicators(data, price_column=column)
    return calculator.wma(period=period, timeframes=timeframes)


def calculate_hma(
    data: pd.DataFrame,
    period: int = 20,
    column: str = 'close',
    timeframes: Optional[List[int]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Convenience function to calculate HMA.

    Args:
        data: DataFrame with price data
        period: Number of periods (default: 20)
        column: Column to use (default: 'close')
        timeframes: List of periods for multiple timeframes

    Returns:
        Series or DataFrame with HMA values
    """
    calculator = TrendIndicators(data, price_column=column)
    return calculator.hma(period=period, timeframes=timeframes)


def calculate_ichimoku(
    data: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> pd.DataFrame:
    """
    Convenience function to calculate Ichimoku Cloud.

    Args:
        data: DataFrame with OHLCV data
        tenkan_period: Period for Tenkan-sen (default: 9)
        kijun_period: Period for Kijun-sen (default: 26)
        senkou_b_period: Period for Senkou Span B (default: 52)
        displacement: Displacement for cloud (default: 26)

    Returns:
        DataFrame with all Ichimoku components
    """
    calculator = TrendIndicators(data)
    return calculator.ichimoku(
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_b_period=senkou_b_period,
        displacement=displacement
    )


def calculate_trend_signals(
    data: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 20,
    indicator: str = 'ema',
    column: str = 'close'
) -> pd.Series:
    """
    Convenience function to generate trend signals.

    Args:
        data: DataFrame with price data
        fast_period: Fast MA period (default: 10)
        slow_period: Slow MA period (default: 20)
        indicator: MA type ('sma', 'ema', 'wma', 'hma')
        column: Column to use (default: 'close')

    Returns:
        Series with trend signals (1: bullish, -1: bearish, 0: neutral)
    """
    calculator = TrendIndicators(data, price_column=column)
    return calculator.get_trend_signal(
        fast_period=fast_period,
        slow_period=slow_period,
        indicator=indicator
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Trend Indicators Module")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)

    # Simulate realistic price data with trend
    trend = np.linspace(100, 150, 200)
    noise = np.random.randn(200) * 5
    close = trend + noise

    sample_data = pd.DataFrame({
        'open': close + np.random.randn(200) * 2,
        'high': close + abs(np.random.randn(200) * 3),
        'low': close - abs(np.random.randn(200) * 3),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    # Ensure OHLC relationships
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)

    print("\nSample data created")
    print(f"Shape: {sample_data.shape}")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")

    # Initialize calculator
    calculator = TrendIndicators(sample_data)
    print(f"\nUsing backend: {calculator.backend}")

    # Test SMA
    print("\n1. Testing SMA...")
    sma_20 = calculator.sma(period=20)
    print(f"   SMA(20) calculated, last value: {sma_20.iloc[-1]:.2f}")

    # Test multiple SMAs
    sma_multi = calculator.sma(timeframes=[10, 20, 50])
    print(f"   Multiple SMAs calculated: {list(sma_multi.columns)}")

    # Test EMA
    print("\n2. Testing EMA...")
    ema_12 = calculator.ema(period=12)
    print(f"   EMA(12) calculated, last value: {ema_12.iloc[-1]:.2f}")

    # Test WMA
    print("\n3. Testing WMA...")
    wma_20 = calculator.wma(period=20)
    print(f"   WMA(20) calculated, last value: {wma_20.iloc[-1]:.2f}")

    # Test HMA
    print("\n4. Testing HMA...")
    hma_20 = calculator.hma(period=20)
    print(f"   HMA(20) calculated, last value: {hma_20.iloc[-1]:.2f}")

    # Test Ichimoku
    print("\n5. Testing Ichimoku Cloud...")
    ichimoku = calculator.ichimoku()
    print(f"   Ichimoku components: {list(ichimoku.columns)}")
    print(f"   Tenkan-sen: {ichimoku['tenkan_sen'].iloc[-1]:.2f}")
    print(f"   Kijun-sen: {ichimoku['kijun_sen'].iloc[-1]:.2f}")

    # Test multi-timeframe analysis
    print("\n6. Testing multi-timeframe analysis...")
    analysis = calculator.multi_timeframe_analysis(
        indicators=['sma', 'ema'],
        timeframes={'sma': [20, 50], 'ema': [12, 26]}
    )
    print(f"   Calculated {len(analysis.columns)} indicators")
    print(f"   Columns: {list(analysis.columns)}")

    # Test trend signals
    print("\n7. Testing trend signals...")
    signals = calculator.get_trend_signal(fast_period=12, slow_period=26, indicator='ema')
    bullish_count = (signals == 1).sum()
    bearish_count = (signals == -1).sum()
    print(f"   Bullish signals: {bullish_count}")
    print(f"   Bearish signals: {bearish_count}")
    print(f"   Current signal: {signals.iloc[-1]}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
