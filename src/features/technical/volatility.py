"""
Volatility Indicators for Technical Analysis

This module provides various volatility indicators used in technical analysis:
- Bollinger Bands: Price bands based on standard deviation
- ATR (Average True Range): Measure of market volatility
- Donchian Channels: Highest high and lowest low over a period
- Keltner Channels: ATR-based volatility bands
- Historical Volatility: Statistical measure of price dispersion

Author: BISTML Trading System
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union


class VolatilityIndicators:
    """
    A comprehensive class for calculating volatility-based technical indicators.
    """

    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
        ddof: int = 0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and two outer bands
        positioned at a specified number of standard deviations above and below
        the middle band.

        Parameters
        ----------
        data : pd.Series
            Price data (typically closing prices)
        period : int, default=20
            Number of periods for moving average
        std_dev : float, default=2.0
            Number of standard deviations for bands
        ddof : int, default=0
            Delta degrees of freedom for std calculation

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (upper_band, middle_band, lower_band)

        Examples
        --------
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> upper, middle, lower = VolatilityIndicators.bollinger_bands(prices, period=5)
        """
        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std(ddof=ddof)

        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def bollinger_bandwidth(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Calculate Bollinger Band Width.

        Bandwidth measures the percentage difference between upper and lower bands,
        useful for identifying periods of high or low volatility.

        Parameters
        ----------
        data : pd.Series
            Price data (typically closing prices)
        period : int, default=20
            Number of periods for moving average
        std_dev : float, default=2.0
            Number of standard deviations for bands

        Returns
        -------
        pd.Series
            Bollinger Band Width values
        """
        upper, middle, lower = VolatilityIndicators.bollinger_bands(data, period, std_dev)
        bandwidth = ((upper - lower) / middle) * 100
        return bandwidth

    @staticmethod
    def bollinger_percent_b(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Calculate Bollinger %B indicator.

        %B shows where price is relative to the bands:
        - Above 1: Price is above upper band
        - 0.5: Price is at middle band
        - Below 0: Price is below lower band

        Parameters
        ----------
        data : pd.Series
            Price data (typically closing prices)
        period : int, default=20
            Number of periods for moving average
        std_dev : float, default=2.0
            Number of standard deviations for bands

        Returns
        -------
        pd.Series
            %B values
        """
        upper, middle, lower = VolatilityIndicators.bollinger_bands(data, period, std_dev)
        percent_b = (data - lower) / (upper - lower)
        return percent_b

    @staticmethod
    def true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calculate True Range.

        True Range is the greatest of:
        - Current high minus current low
        - Absolute value of current high minus previous close
        - Absolute value of current low minus previous close

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices

        Returns
        -------
        pd.Series
            True Range values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return true_range

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        method: str = 'wilder'
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR is a measure of volatility that accounts for gaps and limit moves.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int, default=14
            Number of periods for averaging
        method : str, default='wilder'
            Smoothing method: 'wilder' (Wilder's smoothing), 'ema', or 'sma'

        Returns
        -------
        pd.Series
            ATR values

        Examples
        --------
        >>> atr_values = VolatilityIndicators.atr(df['High'], df['Low'], df['Close'])
        """
        tr = VolatilityIndicators.true_range(high, low, close)

        if method == 'wilder':
            # Wilder's smoothing: ATR = (Previous ATR * (n-1) + Current TR) / n
            atr_values = pd.Series(index=tr.index, dtype=float)
            atr_values.iloc[period-1] = tr.iloc[:period].mean()

            for i in range(period, len(tr)):
                atr_values.iloc[i] = (atr_values.iloc[i-1] * (period - 1) + tr.iloc[i]) / period

        elif method == 'ema':
            atr_values = tr.ewm(span=period, adjust=False).mean()
        elif method == 'sma':
            atr_values = tr.rolling(window=period).mean()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'wilder', 'ema', or 'sma'")

        return atr_values

    @staticmethod
    def atr_percent(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        method: str = 'wilder'
    ) -> pd.Series:
        """
        Calculate ATR as a percentage of closing price.

        This normalized version of ATR allows for better comparison across
        different price levels and securities.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int, default=14
            Number of periods for averaging
        method : str, default='wilder'
            Smoothing method: 'wilder', 'ema', or 'sma'

        Returns
        -------
        pd.Series
            ATR percentage values
        """
        atr_values = VolatilityIndicators.atr(high, low, close, period, method)
        atr_pct = (atr_values / close) * 100
        return atr_pct

    @staticmethod
    def donchian_channels(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels.

        Donchian Channels track the highest high and lowest low over a period,
        with the middle line being the average of the two.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        period : int, default=20
            Lookback period for channels

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (upper_channel, middle_channel, lower_channel)

        Examples
        --------
        >>> upper, middle, lower = VolatilityIndicators.donchian_channels(
        ...     df['High'], df['Low'], period=20
        ... )
        """
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2

        return upper_channel, middle_channel, lower_channel

    @staticmethod
    def donchian_width(
        high: pd.Series,
        low: pd.Series,
        period: int = 20,
        normalized: bool = True
    ) -> pd.Series:
        """
        Calculate Donchian Channel Width.

        Measures the width of the Donchian Channel, useful for identifying
        volatility expansion and contraction.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        period : int, default=20
            Lookback period for channels
        normalized : bool, default=True
            If True, return as percentage of middle channel

        Returns
        -------
        pd.Series
            Channel width values
        """
        upper, middle, lower = VolatilityIndicators.donchian_channels(high, low, period)
        width = upper - lower

        if normalized:
            width = (width / middle) * 100

        return width

    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        ma_type: str = 'ema'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.

        Keltner Channels use ATR to set channel distance from a moving average,
        combining trend and volatility.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int, default=20
            Period for the moving average (middle line)
        atr_period : int, default=10
            Period for ATR calculation
        multiplier : float, default=2.0
            ATR multiplier for channel distance
        ma_type : str, default='ema'
            Type of moving average: 'ema' or 'sma'

        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            (upper_channel, middle_line, lower_channel)

        Examples
        --------
        >>> upper, middle, lower = VolatilityIndicators.keltner_channels(
        ...     df['High'], df['Low'], df['Close']
        ... )
        """
        # Calculate middle line
        if ma_type == 'ema':
            middle_line = close.ewm(span=period, adjust=False).mean()
        elif ma_type == 'sma':
            middle_line = close.rolling(window=period).mean()
        else:
            raise ValueError(f"Unknown ma_type: {ma_type}. Use 'ema' or 'sma'")

        # Calculate ATR
        atr_values = VolatilityIndicators.atr(high, low, close, atr_period)

        # Calculate channels
        upper_channel = middle_line + (multiplier * atr_values)
        lower_channel = middle_line - (multiplier * atr_values)

        return upper_channel, middle_line, lower_channel

    @staticmethod
    def keltner_width(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> pd.Series:
        """
        Calculate Keltner Channel Width.

        Measures the width of the Keltner Channel as a percentage of the middle line.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int, default=20
            Period for the moving average
        atr_period : int, default=10
            Period for ATR calculation
        multiplier : float, default=2.0
            ATR multiplier for channel distance

        Returns
        -------
        pd.Series
            Channel width as percentage
        """
        upper, middle, lower = VolatilityIndicators.keltner_channels(
            high, low, close, period, atr_period, multiplier
        )
        width = ((upper - lower) / middle) * 100
        return width

    @staticmethod
    def historical_volatility(
        data: pd.Series,
        period: int = 30,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculate Historical Volatility (HV).

        Measures the rate at which price has been changing, calculated as
        the standard deviation of logarithmic returns.

        Parameters
        ----------
        data : pd.Series
            Price data (typically closing prices)
        period : int, default=30
            Lookback period for volatility calculation
        annualize : bool, default=True
            If True, annualize the volatility
        trading_periods : int, default=252
            Number of trading periods per year (252 for daily, 52 for weekly)

        Returns
        -------
        pd.Series
            Historical volatility values

        Examples
        --------
        >>> hv = VolatilityIndicators.historical_volatility(df['Close'], period=30)
        """
        # Calculate log returns
        log_returns = np.log(data / data.shift(1))

        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=period).std()

        # Annualize if requested
        if annualize:
            volatility = volatility * np.sqrt(trading_periods)

        return volatility * 100  # Convert to percentage

    @staticmethod
    def parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        period: int = 30,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculate Parkinson's Historical Volatility.

        Uses high and low prices to estimate volatility, which can be more
        efficient than close-to-close volatility.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        period : int, default=30
            Lookback period for volatility calculation
        annualize : bool, default=True
            If True, annualize the volatility
        trading_periods : int, default=252
            Number of trading periods per year

        Returns
        -------
        pd.Series
            Parkinson volatility values
        """
        # Parkinson formula: sqrt(1/(4*ln(2)) * E[(ln(H/L))^2])
        hl_ratio = np.log(high / low)
        hl_squared = hl_ratio ** 2

        volatility = np.sqrt(
            hl_squared.rolling(window=period).mean() / (4 * np.log(2))
        )

        if annualize:
            volatility = volatility * np.sqrt(trading_periods)

        return volatility * 100

    @staticmethod
    def garman_klass_volatility(
        high: pd.Series,
        low: pd.Series,
        open_price: pd.Series,
        close: pd.Series,
        period: int = 30,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculate Garman-Klass Historical Volatility.

        A more sophisticated volatility estimator that uses OHLC data,
        more efficient than standard historical volatility.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        open_price : pd.Series
            Open prices
        close : pd.Series
            Close prices
        period : int, default=30
            Lookback period for volatility calculation
        annualize : bool, default=True
            If True, annualize the volatility
        trading_periods : int, default=252
            Number of trading periods per year

        Returns
        -------
        pd.Series
            Garman-Klass volatility values
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)

        # Garman-Klass formula
        rs = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

        volatility = np.sqrt(rs.rolling(window=period).mean())

        if annualize:
            volatility = volatility * np.sqrt(trading_periods)

        return volatility * 100

    @staticmethod
    def yang_zhang_volatility(
        high: pd.Series,
        low: pd.Series,
        open_price: pd.Series,
        close: pd.Series,
        period: int = 30,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculate Yang-Zhang Historical Volatility.

        Combines overnight volatility, open-to-close volatility, and
        Rogers-Satchell volatility. Handles overnight jumps and trending markets.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        open_price : pd.Series
            Open prices
        close : pd.Series
            Close prices
        period : int, default=30
            Lookback period for volatility calculation
        annualize : bool, default=True
            If True, annualize the volatility
        trading_periods : int, default=252
            Number of trading periods per year

        Returns
        -------
        pd.Series
            Yang-Zhang volatility values
        """
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)

        log_oc = np.log(open_price / close.shift(1))
        log_cc = np.log(close / close.shift(1))

        # Overnight volatility
        overnight_vol = log_oc ** 2

        # Open-to-close volatility
        open_close_vol = log_co ** 2

        # Rogers-Satchell component
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        k = 0.34 / (1.34 + (period + 1) / (period - 1))

        volatility = np.sqrt(
            overnight_vol.rolling(window=period).mean() +
            k * open_close_vol.rolling(window=period).mean() +
            (1 - k) * rs.rolling(window=period).mean()
        )

        if annualize:
            volatility = volatility * np.sqrt(trading_periods)

        return volatility * 100

    @staticmethod
    def ulcer_index(
        data: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Ulcer Index.

        Measures downside volatility, focusing on the depth and duration
        of price declines from recent highs.

        Parameters
        ----------
        data : pd.Series
            Price data (typically closing prices)
        period : int, default=14
            Lookback period

        Returns
        -------
        pd.Series
            Ulcer Index values
        """
        # Calculate running maximum
        running_max = data.rolling(window=period, min_periods=1).max()

        # Calculate percentage drawdown
        pct_drawdown = ((data - running_max) / running_max) * 100

        # Square the drawdowns
        squared_drawdown = pct_drawdown ** 2

        # Calculate Ulcer Index
        ulcer = np.sqrt(squared_drawdown.rolling(window=period).mean())

        return ulcer

    @staticmethod
    def mass_index(
        high: pd.Series,
        low: pd.Series,
        fast_period: int = 9,
        slow_period: int = 25
    ) -> pd.Series:
        """
        Calculate Mass Index.

        Identifies trend reversals by measuring the narrowing and widening
        of the range between high and low prices.

        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        fast_period : int, default=9
            Period for first EMA
        slow_period : int, default=25
            Period for summation

        Returns
        -------
        pd.Series
            Mass Index values
        """
        price_range = high - low

        # Single EMA of range
        ema1 = price_range.ewm(span=fast_period, adjust=False).mean()

        # Double EMA of range
        ema2 = ema1.ewm(span=fast_period, adjust=False).mean()

        # Mass Index is the sum of the ratio over the slow period
        mass = (ema1 / ema2).rolling(window=slow_period).sum()

        return mass


def calculate_all_volatility_indicators(
    df: pd.DataFrame,
    high_col: str = 'High',
    low_col: str = 'Low',
    open_col: str = 'Open',
    close_col: str = 'Close'
) -> pd.DataFrame:
    """
    Calculate all volatility indicators and add them to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC price data
    high_col : str, default='High'
        Name of high price column
    low_col : str, default='Low'
        Name of low price column
    open_col : str, default='Open'
        Name of open price column
    close_col : str, default='Close'
        Name of close price column

    Returns
    -------
    pd.DataFrame
        Dataframe with all volatility indicators added

    Examples
    --------
    >>> df_with_volatility = calculate_all_volatility_indicators(price_df)
    """
    result = df.copy()

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = VolatilityIndicators.bollinger_bands(
        result[close_col], period=20
    )
    result['BB_Upper'] = bb_upper
    result['BB_Middle'] = bb_middle
    result['BB_Lower'] = bb_lower
    result['BB_Width'] = VolatilityIndicators.bollinger_bandwidth(result[close_col])
    result['BB_PercentB'] = VolatilityIndicators.bollinger_percent_b(result[close_col])

    # ATR
    result['ATR'] = VolatilityIndicators.atr(
        result[high_col], result[low_col], result[close_col]
    )
    result['ATR_Percent'] = VolatilityIndicators.atr_percent(
        result[high_col], result[low_col], result[close_col]
    )

    # Donchian Channels
    dc_upper, dc_middle, dc_lower = VolatilityIndicators.donchian_channels(
        result[high_col], result[low_col], period=20
    )
    result['DC_Upper'] = dc_upper
    result['DC_Middle'] = dc_middle
    result['DC_Lower'] = dc_lower
    result['DC_Width'] = VolatilityIndicators.donchian_width(
        result[high_col], result[low_col]
    )

    # Keltner Channels
    kc_upper, kc_middle, kc_lower = VolatilityIndicators.keltner_channels(
        result[high_col], result[low_col], result[close_col]
    )
    result['KC_Upper'] = kc_upper
    result['KC_Middle'] = kc_middle
    result['KC_Lower'] = kc_lower
    result['KC_Width'] = VolatilityIndicators.keltner_width(
        result[high_col], result[low_col], result[close_col]
    )

    # Historical Volatility
    result['HV_30'] = VolatilityIndicators.historical_volatility(
        result[close_col], period=30
    )
    result['HV_Parkinson'] = VolatilityIndicators.parkinson_volatility(
        result[high_col], result[low_col], period=30
    )
    result['HV_GarmanKlass'] = VolatilityIndicators.garman_klass_volatility(
        result[high_col], result[low_col], result[open_col], result[close_col]
    )
    result['HV_YangZhang'] = VolatilityIndicators.yang_zhang_volatility(
        result[high_col], result[low_col], result[open_col], result[close_col]
    )

    # Additional volatility metrics
    result['Ulcer_Index'] = VolatilityIndicators.ulcer_index(result[close_col])
    result['Mass_Index'] = VolatilityIndicators.mass_index(
        result[high_col], result[low_col]
    )

    return result


if __name__ == "__main__":
    # Example usage
    print("Volatility Indicators Module")
    print("=" * 50)
    print("\nAvailable indicators:")
    print("- Bollinger Bands (with Width and %B)")
    print("- Average True Range (ATR)")
    print("- Donchian Channels")
    print("- Keltner Channels")
    print("- Historical Volatility (Close-to-Close)")
    print("- Parkinson's Historical Volatility")
    print("- Garman-Klass Historical Volatility")
    print("- Yang-Zhang Historical Volatility")
    print("- Ulcer Index")
    print("- Mass Index")
    print("\nExample usage:")
    print(">>> from features.technical.volatility import VolatilityIndicators")
    print(">>> upper, middle, lower = VolatilityIndicators.bollinger_bands(df['Close'])")
    print(">>> atr = VolatilityIndicators.atr(df['High'], df['Low'], df['Close'])")
