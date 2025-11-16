"""
Utility helper functions for BIST ML Trading System.

This module provides various utility functions including:
- Date and time helpers
- BIST market hours checking
- Turkish Lira formatting
- Percentage calculations
- Data conversion utilities
"""

from datetime import datetime, time, timedelta
from typing import Union, Optional, List, Any
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo


# =============================================================================
# Date and Time Helpers
# =============================================================================

def get_istanbul_time() -> datetime:
    """
    Get current time in Istanbul timezone.

    Returns:
        datetime: Current Istanbul time
    """
    return datetime.now(ZoneInfo("Europe/Istanbul"))


def convert_to_istanbul_time(dt: datetime) -> datetime:
    """
    Convert a datetime object to Istanbul timezone.

    Args:
        dt: Datetime object to convert

    Returns:
        datetime: Datetime in Istanbul timezone
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone info
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("Europe/Istanbul"))


def is_weekday(date: Optional[datetime] = None) -> bool:
    """
    Check if a given date is a weekday (Monday-Friday).

    Args:
        date: Date to check. If None, uses current Istanbul time.

    Returns:
        bool: True if weekday, False if weekend
    """
    if date is None:
        date = get_istanbul_time()
    return date.weekday() < 5


def get_next_trading_day(date: Optional[datetime] = None) -> datetime:
    """
    Get the next trading day (next weekday).

    Args:
        date: Starting date. If None, uses current Istanbul time.

    Returns:
        datetime: Next trading day
    """
    if date is None:
        date = get_istanbul_time()

    next_day = date + timedelta(days=1)
    while not is_weekday(next_day):
        next_day += timedelta(days=1)

    return next_day


def get_previous_trading_day(date: Optional[datetime] = None) -> datetime:
    """
    Get the previous trading day (previous weekday).

    Args:
        date: Starting date. If None, uses current Istanbul time.

    Returns:
        datetime: Previous trading day
    """
    if date is None:
        date = get_istanbul_time()

    prev_day = date - timedelta(days=1)
    while not is_weekday(prev_day):
        prev_day -= timedelta(days=1)

    return prev_day


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object to string.

    Args:
        dt: Datetime to format
        format_str: Format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        str: Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse a datetime string to datetime object.

    Args:
        date_str: String to parse
        format_str: Format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        datetime: Parsed datetime object
    """
    return datetime.strptime(date_str, format_str)


def get_date_range(start_date: datetime, end_date: datetime,
                   trading_days_only: bool = True) -> List[datetime]:
    """
    Get a list of dates between start and end date.

    Args:
        start_date: Start date
        end_date: End date
        trading_days_only: If True, only include weekdays

    Returns:
        List[datetime]: List of dates
    """
    dates = []
    current = start_date

    while current <= end_date:
        if not trading_days_only or is_weekday(current):
            dates.append(current)
        current += timedelta(days=1)

    return dates


# =============================================================================
# BIST Market Hours Checking
# =============================================================================

# BIST trading sessions
BIST_MORNING_OPEN = time(10, 0)   # 10:00 AM
BIST_MORNING_CLOSE = time(12, 30)  # 12:30 PM
BIST_AFTERNOON_OPEN = time(14, 0)  # 2:00 PM
BIST_AFTERNOON_CLOSE = time(18, 0) # 6:00 PM

# Pre-market and after-market
BIST_PREMARKET_OPEN = time(9, 30)   # 9:30 AM
BIST_AFTERMARKET_CLOSE = time(18, 10) # 6:10 PM


def is_bist_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if BIST is currently open for trading.

    BIST trading hours:
    - Morning session: 10:00 - 12:30
    - Afternoon session: 14:00 - 18:00
    - Monday to Friday only

    Args:
        dt: Datetime to check. If None, uses current Istanbul time.

    Returns:
        bool: True if BIST is open, False otherwise
    """
    if dt is None:
        dt = get_istanbul_time()

    # Check if it's a weekday
    if not is_weekday(dt):
        return False

    current_time = dt.time()

    # Morning session
    if BIST_MORNING_OPEN <= current_time <= BIST_MORNING_CLOSE:
        return True

    # Afternoon session
    if BIST_AFTERNOON_OPEN <= current_time <= BIST_AFTERNOON_CLOSE:
        return True

    return False


def is_premarket(dt: Optional[datetime] = None) -> bool:
    """
    Check if it's pre-market hours (9:30 - 10:00).

    Args:
        dt: Datetime to check. If None, uses current Istanbul time.

    Returns:
        bool: True if pre-market, False otherwise
    """
    if dt is None:
        dt = get_istanbul_time()

    if not is_weekday(dt):
        return False

    current_time = dt.time()
    return BIST_PREMARKET_OPEN <= current_time < BIST_MORNING_OPEN


def is_aftermarket(dt: Optional[datetime] = None) -> bool:
    """
    Check if it's after-market hours (18:00 - 18:10).

    Args:
        dt: Datetime to check. If None, uses current Istanbul time.

    Returns:
        bool: True if after-market, False otherwise
    """
    if dt is None:
        dt = get_istanbul_time()

    if not is_weekday(dt):
        return False

    current_time = dt.time()
    return BIST_AFTERNOON_CLOSE < current_time <= BIST_AFTERMARKET_CLOSE


def get_market_session(dt: Optional[datetime] = None) -> str:
    """
    Get the current market session.

    Args:
        dt: Datetime to check. If None, uses current Istanbul time.

    Returns:
        str: One of 'premarket', 'morning', 'midday_break', 'afternoon',
             'aftermarket', 'closed'
    """
    if dt is None:
        dt = get_istanbul_time()

    if not is_weekday(dt):
        return 'closed'

    current_time = dt.time()

    if current_time < BIST_PREMARKET_OPEN:
        return 'closed'
    elif current_time < BIST_MORNING_OPEN:
        return 'premarket'
    elif current_time <= BIST_MORNING_CLOSE:
        return 'morning'
    elif current_time < BIST_AFTERNOON_OPEN:
        return 'midday_break'
    elif current_time <= BIST_AFTERNOON_CLOSE:
        return 'afternoon'
    elif current_time <= BIST_AFTERMARKET_CLOSE:
        return 'aftermarket'
    else:
        return 'closed'


def time_until_market_open(dt: Optional[datetime] = None) -> Optional[timedelta]:
    """
    Calculate time until next market opening.

    Args:
        dt: Reference datetime. If None, uses current Istanbul time.

    Returns:
        timedelta: Time until market opens, or None if market is open
    """
    if dt is None:
        dt = get_istanbul_time()

    if is_bist_open(dt):
        return None

    current_time = dt.time()

    # If during midday break, return time until afternoon session
    if BIST_MORNING_CLOSE < current_time < BIST_AFTERNOON_OPEN:
        afternoon_open = datetime.combine(dt.date(), BIST_AFTERNOON_OPEN)
        return afternoon_open - dt

    # If after market close or before premarket, return time until next day's open
    if current_time >= BIST_AFTERNOON_CLOSE or current_time < BIST_MORNING_OPEN:
        next_trading = get_next_trading_day(dt)
        next_open = datetime.combine(next_trading.date(), BIST_MORNING_OPEN)
        next_open = next_open.replace(tzinfo=dt.tzinfo)
        return next_open - dt

    # Otherwise, market opens same day
    morning_open = datetime.combine(dt.date(), BIST_MORNING_OPEN)
    morning_open = morning_open.replace(tzinfo=dt.tzinfo)
    return morning_open - dt


def time_until_market_close(dt: Optional[datetime] = None) -> Optional[timedelta]:
    """
    Calculate time until market closing.

    Args:
        dt: Reference datetime. If None, uses current Istanbul time.

    Returns:
        timedelta: Time until market closes, or None if market is closed
    """
    if dt is None:
        dt = get_istanbul_time()

    if not is_bist_open(dt):
        return None

    current_time = dt.time()

    # If in morning session
    if BIST_MORNING_OPEN <= current_time <= BIST_MORNING_CLOSE:
        morning_close = datetime.combine(dt.date(), BIST_MORNING_CLOSE)
        morning_close = morning_close.replace(tzinfo=dt.tzinfo)
        return morning_close - dt

    # If in afternoon session
    if BIST_AFTERNOON_OPEN <= current_time <= BIST_AFTERNOON_CLOSE:
        afternoon_close = datetime.combine(dt.date(), BIST_AFTERNOON_CLOSE)
        afternoon_close = afternoon_close.replace(tzinfo=dt.tzinfo)
        return afternoon_close - dt

    return None


# =============================================================================
# Turkish Lira Formatting
# =============================================================================

def format_try(amount: float, decimals: int = 2, symbol: bool = True,
               thousands_sep: str = '.', decimal_sep: str = ',') -> str:
    """
    Format amount as Turkish Lira.

    Turkish number formatting uses:
    - Period (.) as thousands separator
    - Comma (,) as decimal separator

    Args:
        amount: Amount to format
        decimals: Number of decimal places (default: 2)
        symbol: Include ₺ symbol (default: True)
        thousands_sep: Thousands separator (default: '.')
        decimal_sep: Decimal separator (default: ',')

    Returns:
        str: Formatted currency string

    Examples:
        >>> format_try(1234567.89)
        '₺1.234.567,89'
        >>> format_try(1234.5, decimals=2, symbol=False)
        '1.234,50'
    """
    # Format with specified decimals
    formatted = f"{abs(amount):,.{decimals}f}"

    # Replace default separators with Turkish format
    formatted = formatted.replace(',', 'TEMP')
    formatted = formatted.replace('.', thousands_sep)
    formatted = formatted.replace('TEMP', decimal_sep)

    # Add currency symbol
    if symbol:
        formatted = f"₺{formatted}"

    # Add negative sign if needed
    if amount < 0:
        formatted = f"-{formatted}"

    return formatted


def parse_try(amount_str: str) -> float:
    """
    Parse Turkish Lira formatted string to float.

    Args:
        amount_str: Formatted currency string

    Returns:
        float: Parsed amount

    Examples:
        >>> parse_try('₺1.234.567,89')
        1234567.89
        >>> parse_try('1.234,50')
        1234.5
    """
    # Remove currency symbol and whitespace
    clean = amount_str.replace('₺', '').replace(' ', '').strip()

    # Handle negative numbers
    is_negative = clean.startswith('-')
    clean = clean.lstrip('-')

    # Replace Turkish separators
    clean = clean.replace('.', '')  # Remove thousands separator
    clean = clean.replace(',', '.')  # Replace decimal separator

    result = float(clean)
    return -result if is_negative else result


# =============================================================================
# Percentage Calculations
# =============================================================================

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        float: Percentage change

    Examples:
        >>> calculate_percentage_change(100, 110)
        10.0
        >>> calculate_percentage_change(100, 90)
        -10.0
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')

    return ((new_value - old_value) / old_value) * 100


def calculate_return(initial_price: float, final_price: float) -> float:
    """
    Calculate return on investment as percentage.

    Args:
        initial_price: Initial/entry price
        final_price: Final/exit price

    Returns:
        float: Return percentage
    """
    return calculate_percentage_change(initial_price, final_price)


def calculate_drawdown(peak: float, trough: float) -> float:
    """
    Calculate drawdown percentage from peak to trough.

    Args:
        peak: Peak value
        trough: Trough value

    Returns:
        float: Drawdown percentage (negative value)
    """
    return calculate_percentage_change(peak, trough)


def format_percentage(value: float, decimals: int = 2,
                     include_sign: bool = True) -> str:
    """
    Format a number as percentage.

    Args:
        value: Value to format (e.g., 10 for 10%)
        decimals: Number of decimal places
        include_sign: Include + sign for positive values

    Returns:
        str: Formatted percentage string

    Examples:
        >>> format_percentage(10.5)
        '+10.50%'
        >>> format_percentage(-5.2)
        '-5.20%'
    """
    sign = ''
    if include_sign and value > 0:
        sign = '+'
    elif value < 0:
        sign = '-'

    return f"{sign}{abs(value):.{decimals}f}%"


def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray, pd.Series],
                          risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        float: Sharpe ratio
    """
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_cagr(initial_value: float, final_value: float,
                  years: float) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        initial_value: Initial investment value
        final_value: Final investment value
        years: Number of years

    Returns:
        float: CAGR as percentage
    """
    if initial_value <= 0 or years <= 0:
        return 0.0

    return (((final_value / initial_value) ** (1 / years)) - 1) * 100


# =============================================================================
# Data Conversion Helpers
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        float: Converted value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        int: Converted value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def normalize_symbol(symbol: str) -> str:
    """
    Normalize stock symbol to standard format.

    Args:
        symbol: Stock symbol

    Returns:
        str: Normalized symbol (uppercase, trimmed)

    Examples:
        >>> normalize_symbol('  tuprs  ')
        'TUPRS'
        >>> normalize_symbol('garan')
        'GARAN'
    """
    return symbol.strip().upper()


def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       errors: str = 'coerce') -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric type.

    Args:
        df: DataFrame to convert
        columns: List of columns to convert. If None, converts all columns.
        errors: How to handle errors ('raise', 'coerce', 'ignore')

    Returns:
        pd.DataFrame: DataFrame with numeric columns
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.columns.tolist()

    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors=errors)

    return df_copy


def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing values in DataFrame.

    Args:
        df: DataFrame with missing values
        method: Fill method ('ffill', 'bfill', 'mean', 'median', 'zero')

    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_copy = df.copy()

    if method == 'ffill':
        df_copy = df_copy.fillna(method='ffill')
    elif method == 'bfill':
        df_copy = df_copy.fillna(method='bfill')
    elif method == 'mean':
        df_copy = df_copy.fillna(df_copy.mean())
    elif method == 'median':
        df_copy = df_copy.fillna(df_copy.median())
    elif method == 'zero':
        df_copy = df_copy.fillna(0)

    return df_copy


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLC data to different timeframe.

    Args:
        df: DataFrame with OHLC data (must have DatetimeIndex)
        timeframe: Target timeframe ('1D', '1H', '5T', etc.)

    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    # Only include columns that exist in the DataFrame
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    resampled = df.resample(timeframe).agg(agg_dict)
    return resampled.dropna()


def calculate_lot_size(position_size: float, price: float,
                      lot_multiplier: int = 1) -> int:
    """
    Calculate number of lots to trade.

    Args:
        position_size: Desired position size in TRY
        price: Current price
        lot_multiplier: Lot size multiplier (default: 1)

    Returns:
        int: Number of lots
    """
    if price <= 0:
        return 0

    shares = position_size / price
    lots = int(shares / lot_multiplier)

    return lots


def round_price(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to nearest tick size.

    Args:
        price: Price to round
        tick_size: Minimum price movement (default: 0.01)

    Returns:
        float: Rounded price
    """
    return round(price / tick_size) * tick_size


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return True


def clip_outliers(data: Union[pd.Series, np.ndarray],
                 n_std: float = 3.0) -> Union[pd.Series, np.ndarray]:
    """
    Clip outliers beyond n standard deviations.

    Args:
        data: Data to clip
        n_std: Number of standard deviations (default: 3.0)

    Returns:
        Clipped data (same type as input)
    """
    mean = np.mean(data)
    std = np.std(data)

    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std

    if isinstance(data, pd.Series):
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        return np.clip(data, lower_bound, upper_bound)


# =============================================================================
# Additional Utility Functions
# =============================================================================

def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """
    Split list into chunks of size n.

    Args:
        lst: List to split
        n: Chunk size

    Returns:
        List of chunks
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.

    Args:
        nested_list: Nested list to flatten

    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def dict_to_namespace(d: dict) -> object:
    """
    Convert dictionary to namespace object.

    Args:
        d: Dictionary to convert

    Returns:
        Namespace object with dict keys as attributes
    """
    from types import SimpleNamespace
    return SimpleNamespace(**d)


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path
    """
    import os
    os.makedirs(directory, exist_ok=True)
