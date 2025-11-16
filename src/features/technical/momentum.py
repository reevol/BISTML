"""
Momentum Indicators for Technical Analysis

This module provides a comprehensive set of momentum-based technical indicators
for the BIST AI Trading System. These indicators help identify the speed and
magnitude of price changes, overbought/oversold conditions, and trend strength.

Indicators included:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- ADX (Average Directional Index)
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import warnings

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using pandas implementations.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    warnings.warn("pandas-ta not available. Using custom implementations.")


class MomentumIndicators:
    """
    A comprehensive class for calculating momentum-based technical indicators.

    This class provides both TA-Lib and pandas-based implementations with
    automatic fallback to ensure robust functionality across different environments.
    """

    def __init__(self, data: pd.DataFrame, use_talib: bool = True):
        """
        Initialize the MomentumIndicators class.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing OHLCV data with columns: 'open', 'high', 'low', 'close', 'volume'
        use_talib : bool, default=True
            Whether to use TA-Lib if available, otherwise use pandas implementations
        """
        self.data = data.copy()
        self.use_talib = use_talib and TALIB_AVAILABLE

        # Standardize column names
        self._standardize_columns()

    def _standardize_columns(self):
        """Standardize column names to lowercase for consistency."""
        column_mapping = {}
        for col in self.data.columns:
            lower_col = col.lower()
            if lower_col in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = lower_col

        if column_mapping:
            self.data.rename(columns=column_mapping, inplace=True)

    def calculate_rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude of
        price changes. It oscillates between 0 and 100, with readings above 70
        indicating overbought conditions and below 30 indicating oversold conditions.

        Parameters:
        -----------
        period : int, default=14
            The lookback period for RSI calculation
        column : str, default='close'
            The column to use for calculation

        Returns:
        --------
        pd.Series
            RSI values
        """
        if self.use_talib:
            return pd.Series(
                talib.RSI(self.data[column].values, timeperiod=period),
                index=self.data.index,
                name=f'RSI_{period}'
            )
        else:
            return self._rsi_pandas(self.data[column], period)

    def _rsi_pandas(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using pandas.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            Lookback period

        Returns:
        --------
        pd.Series
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.Series(rsi, index=prices.index, name=f'RSI_{period}')

    def calculate_macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.

        Parameters:
        -----------
        fast_period : int, default=12
            Fast EMA period
        slow_period : int, default=26
            Slow EMA period
        signal_period : int, default=9
            Signal line EMA period
        column : str, default='close'
            The column to use for calculation

        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, Signal line, and Histogram
        """
        if self.use_talib:
            macd, signal, hist = talib.MACD(
                self.data[column].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return (
                pd.Series(macd, index=self.data.index, name='MACD'),
                pd.Series(signal, index=self.data.index, name='MACD_Signal'),
                pd.Series(hist, index=self.data.index, name='MACD_Hist')
            )
        else:
            return self._macd_pandas(
                self.data[column],
                fast_period,
                slow_period,
                signal_period
            )

    def _macd_pandas(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD using pandas.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        fast : int
            Fast EMA period
        slow : int
            Slow EMA period
        signal : int
            Signal line period

        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, Signal line, and Histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            pd.Series(macd_line, name='MACD'),
            pd.Series(signal_line, name='MACD_Signal'),
            pd.Series(histogram, name='MACD_Hist')
        )

    def calculate_stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        The Stochastic Oscillator compares a closing price to its price range
        over a given time period. Values range from 0 to 100, with readings
        above 80 considered overbought and below 20 considered oversold.

        Parameters:
        -----------
        k_period : int, default=14
            Lookback period for %K calculation
        d_period : int, default=3
            Moving average period for %D (signal line)
        smooth_k : int, default=3
            Smoothing period for %K

        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            %K line and %D line
        """
        if self.use_talib:
            slowk, slowd = talib.STOCH(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                fastk_period=k_period,
                slowk_period=smooth_k,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            return (
                pd.Series(slowk, index=self.data.index, name='Stoch_K'),
                pd.Series(slowd, index=self.data.index, name='Stoch_D')
            )
        else:
            return self._stochastic_pandas(k_period, d_period, smooth_k)

    def _stochastic_pandas(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator using pandas.

        Parameters:
        -----------
        k_period : int
            Lookback period for %K
        d_period : int
            Period for %D
        smooth_k : int
            Smoothing for %K

        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            %K and %D lines
        """
        # Calculate raw %K
        lowest_low = self.data['low'].rolling(window=k_period).min()
        highest_high = self.data['high'].rolling(window=k_period).max()

        raw_k = 100 * (self.data['close'] - lowest_low) / (highest_high - lowest_low)

        # Smooth %K
        k_line = raw_k.rolling(window=smooth_k).mean()

        # Calculate %D (signal line)
        d_line = k_line.rolling(window=d_period).mean()

        return (
            pd.Series(k_line, name='Stoch_K'),
            pd.Series(d_line, name='Stoch_D')
        )

    def calculate_adx(
        self,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX (Average Directional Index) and directional indicators.

        ADX measures the strength of a trend regardless of direction. Values
        above 25 indicate a strong trend, while values below 20 suggest a
        weak or ranging market.

        Parameters:
        -----------
        period : int, default=14
            Lookback period for ADX calculation

        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            ADX, +DI (Plus Directional Indicator), -DI (Minus Directional Indicator)
        """
        if self.use_talib:
            adx = talib.ADX(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=period
            )
            plus_di = talib.PLUS_DI(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=period
            )
            minus_di = talib.MINUS_DI(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=period
            )
            return (
                pd.Series(adx, index=self.data.index, name=f'ADX_{period}'),
                pd.Series(plus_di, index=self.data.index, name=f'Plus_DI_{period}'),
                pd.Series(minus_di, index=self.data.index, name=f'Minus_DI_{period}')
            )
        else:
            return self._adx_pandas(period)

    def _adx_pandas(self, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX using pandas.

        Parameters:
        -----------
        period : int
            Lookback period

        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            ADX, +DI, -DI
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Smooth TR and DM using Wilder's smoothing (EMA with alpha = 1/period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

        # Calculate Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return (
            pd.Series(adx, name=f'ADX_{period}'),
            pd.Series(plus_di, name=f'Plus_DI_{period}'),
            pd.Series(minus_di, name=f'Minus_DI_{period}')
        )

    def calculate_williams_r(
        self,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R.

        Williams %R is a momentum indicator that measures overbought and
        oversold levels. It ranges from 0 to -100, with readings above -20
        considered overbought and below -80 considered oversold.

        Parameters:
        -----------
        period : int, default=14
            Lookback period

        Returns:
        --------
        pd.Series
            Williams %R values
        """
        if self.use_talib:
            willr = talib.WILLR(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=period
            )
            return pd.Series(willr, index=self.data.index, name=f'Williams_R_{period}')
        else:
            return self._williams_r_pandas(period)

    def _williams_r_pandas(self, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R using pandas.

        Parameters:
        -----------
        period : int
            Lookback period

        Returns:
        --------
        pd.Series
            Williams %R values
        """
        highest_high = self.data['high'].rolling(window=period).max()
        lowest_low = self.data['low'].rolling(window=period).min()

        williams_r = -100 * (highest_high - self.data['close']) / (highest_high - lowest_low)

        return pd.Series(williams_r, name=f'Williams_R_{period}')

    def calculate_cci(
        self,
        period: int = 20,
        constant: float = 0.015
    ) -> pd.Series:
        """
        Calculate CCI (Commodity Channel Index).

        CCI measures the current price level relative to an average price level
        over a given period. It can be used to identify overbought/oversold
        conditions and trend reversals.

        Parameters:
        -----------
        period : int, default=20
            Lookback period
        constant : float, default=0.015
            Scaling constant (typically 0.015)

        Returns:
        --------
        pd.Series
            CCI values
        """
        if self.use_talib:
            cci = talib.CCI(
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values,
                timeperiod=period
            )
            return pd.Series(cci, index=self.data.index, name=f'CCI_{period}')
        else:
            return self._cci_pandas(period, constant)

    def _cci_pandas(self, period: int = 20, constant: float = 0.015) -> pd.Series:
        """
        Calculate CCI using pandas.

        Parameters:
        -----------
        period : int
            Lookback period
        constant : float
            Scaling constant

        Returns:
        --------
        pd.Series
            CCI values
        """
        # Typical Price
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3

        # Simple Moving Average of Typical Price
        sma_tp = tp.rolling(window=period).mean()

        # Mean Deviation
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(),
            raw=True
        )

        # CCI
        cci = (tp - sma_tp) / (constant * mad)

        return pd.Series(cci, name=f'CCI_{period}')

    def calculate_roc(
        self,
        period: int = 12,
        column: str = 'close'
    ) -> pd.Series:
        """
        Calculate ROC (Rate of Change).

        ROC measures the percentage change in price between the current price
        and the price n periods ago. It oscillates above and below zero.

        Parameters:
        -----------
        period : int, default=12
            Lookback period
        column : str, default='close'
            The column to use for calculation

        Returns:
        --------
        pd.Series
            ROC values (percentage)
        """
        if self.use_talib:
            roc = talib.ROC(
                self.data[column].values,
                timeperiod=period
            )
            return pd.Series(roc, index=self.data.index, name=f'ROC_{period}')
        else:
            return self._roc_pandas(self.data[column], period)

    def _roc_pandas(self, prices: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate ROC using pandas.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            Lookback period

        Returns:
        --------
        pd.Series
            ROC values
        """
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100

        return pd.Series(roc, name=f'ROC_{period}')

    def calculate_all_momentum_indicators(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        adx_period: int = 14,
        williams_period: int = 14,
        cci_period: int = 20,
        roc_period: int = 12
    ) -> pd.DataFrame:
        """
        Calculate all momentum indicators at once and return as DataFrame.

        Parameters:
        -----------
        rsi_period : int, default=14
            RSI lookback period
        macd_fast : int, default=12
            MACD fast period
        macd_slow : int, default=26
            MACD slow period
        macd_signal : int, default=9
            MACD signal period
        stoch_k : int, default=14
            Stochastic %K period
        stoch_d : int, default=3
            Stochastic %D period
        stoch_smooth : int, default=3
            Stochastic smoothing period
        adx_period : int, default=14
            ADX period
        williams_period : int, default=14
            Williams %R period
        cci_period : int, default=20
            CCI period
        roc_period : int, default=12
            ROC period

        Returns:
        --------
        pd.DataFrame
            DataFrame with all momentum indicators
        """
        result = self.data.copy()

        # RSI
        result[f'RSI_{rsi_period}'] = self.calculate_rsi(rsi_period)

        # MACD
        macd, signal, hist = self.calculate_macd(macd_fast, macd_slow, macd_signal)
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Hist'] = hist

        # Stochastic
        stoch_k_line, stoch_d_line = self.calculate_stochastic(
            stoch_k, stoch_d, stoch_smooth
        )
        result['Stoch_K'] = stoch_k_line
        result['Stoch_D'] = stoch_d_line

        # ADX
        adx, plus_di, minus_di = self.calculate_adx(adx_period)
        result[f'ADX_{adx_period}'] = adx
        result[f'Plus_DI_{adx_period}'] = plus_di
        result[f'Minus_DI_{adx_period}'] = minus_di

        # Williams %R
        result[f'Williams_R_{williams_period}'] = self.calculate_williams_r(williams_period)

        # CCI
        result[f'CCI_{cci_period}'] = self.calculate_cci(cci_period)

        # ROC
        result[f'ROC_{roc_period}'] = self.calculate_roc(roc_period)

        return result


# Convenience functions for quick calculations
def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Quick function to calculate RSI.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    period : int, default=14
        Lookback period
    column : str, default='close'
        Column to use

    Returns:
    --------
    pd.Series
        RSI values
    """
    indicator = MomentumIndicators(data)
    return indicator.calculate_rsi(period, column)


def calculate_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = 'close'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Quick function to calculate MACD.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    fast : int, default=12
        Fast period
    slow : int, default=26
        Slow period
    signal : int, default=9
        Signal period
    column : str, default='close'
        Column to use

    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series]
        MACD, Signal, Histogram
    """
    indicator = MomentumIndicators(data)
    return indicator.calculate_macd(fast, slow, signal, column)


def calculate_all_momentum(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Quick function to calculate all momentum indicators.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    **kwargs : dict
        Optional parameters for each indicator

    Returns:
    --------
    pd.DataFrame
        Data with all momentum indicators
    """
    indicator = MomentumIndicators(data)
    return indicator.calculate_all_momentum_indicators(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Momentum Indicators Module")
    print("=" * 50)
    print("\nAvailable indicators:")
    print("- RSI (Relative Strength Index)")
    print("- MACD (Moving Average Convergence Divergence)")
    print("- Stochastic Oscillator")
    print("- ADX (Average Directional Index)")
    print("- Williams %R")
    print("- CCI (Commodity Channel Index)")
    print("- ROC (Rate of Change)")
    print("\nTA-Lib available:", TALIB_AVAILABLE)
    print("pandas-ta available:", PANDAS_TA_AVAILABLE)
