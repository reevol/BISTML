"""
Validation Utilities for BIST AI Trading System

This module provides validation functions for individual parameters and inputs including:
- Stock codes/symbols (BIST format)
- Date and time inputs
- Price and financial values
- Portfolio parameters (shares, allocations, percentages)
- Configuration parameters (timeframes, methods, thresholds)

This complements src/data/processors/validator.py which validates DataFrames.
This module focuses on validating individual values and configuration parameters.

Author: BIST AI Trading System
Date: 2025-11-16
"""

import re
import logging
from datetime import datetime, timedelta, date
from typing import Union, List, Optional, Any, Tuple
from enum import Enum
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation errors"""
    pass


class InvalidStockCodeError(ValidationError):
    """Raised when stock code is invalid"""
    pass


class InvalidDateError(ValidationError):
    """Raised when date is invalid"""
    pass


class InvalidPriceError(ValidationError):
    """Raised when price is invalid"""
    pass


class InvalidQuantityError(ValidationError):
    """Raised when quantity/shares is invalid"""
    pass


class InvalidPercentageError(ValidationError):
    """Raised when percentage is invalid"""
    pass


class InvalidConfigurationError(ValidationError):
    """Raised when configuration parameter is invalid"""
    pass


# ============================================================================
# Stock Code Validators
# ============================================================================

def validate_bist_symbol(symbol: str, allow_suffix: bool = True, auto_fix: bool = False) -> str:
    """
    Validate BIST stock symbol format.

    Valid formats:
    - 3-6 uppercase letters (e.g., "THYAO", "GARAN", "ISCTR")
    - Optionally followed by .IS suffix (e.g., "THYAO.IS")

    Args:
        symbol: Stock symbol to validate
        allow_suffix: Whether to allow .IS suffix
        auto_fix: If True, automatically fix common issues (strip whitespace, uppercase)

    Returns:
        Validated (and potentially fixed) symbol

    Raises:
        InvalidStockCodeError: If symbol format is invalid

    Examples:
        >>> validate_bist_symbol("THYAO")
        'THYAO'
        >>> validate_bist_symbol("thyao", auto_fix=True)
        'THYAO'
        >>> validate_bist_symbol("THYAO.IS")
        'THYAO.IS'
        >>> validate_bist_symbol("123ABC")
        InvalidStockCodeError: Invalid BIST symbol format
    """
    if not symbol or not isinstance(symbol, str):
        raise InvalidStockCodeError(f"Symbol must be a non-empty string, got: {type(symbol)}")

    # Auto-fix if requested
    if auto_fix:
        symbol = symbol.strip().upper()

    # Check for .IS suffix
    has_suffix = symbol.endswith('.IS')

    if has_suffix:
        if not allow_suffix:
            raise InvalidStockCodeError(f"Symbol '{symbol}' has .IS suffix but suffix not allowed")
        base_symbol = symbol[:-3]  # Remove .IS
    else:
        base_symbol = symbol

    # Validate base symbol: 3-6 uppercase letters
    if not re.match(r'^[A-Z]{3,6}$', base_symbol):
        raise InvalidStockCodeError(
            f"Invalid BIST symbol format: '{symbol}'. "
            f"Must be 3-6 uppercase letters, optionally followed by .IS"
        )

    return symbol


def validate_bist_symbols(symbols: List[str], allow_suffix: bool = True,
                          auto_fix: bool = False, skip_invalid: bool = False) -> List[str]:
    """
    Validate a list of BIST stock symbols.

    Args:
        symbols: List of stock symbols
        allow_suffix: Whether to allow .IS suffix
        auto_fix: If True, automatically fix common issues
        skip_invalid: If True, skip invalid symbols instead of raising error

    Returns:
        List of validated symbols

    Raises:
        InvalidStockCodeError: If any symbol is invalid (unless skip_invalid=True)
    """
    if not isinstance(symbols, (list, tuple)):
        raise InvalidStockCodeError(f"Symbols must be a list or tuple, got: {type(symbols)}")

    validated = []
    invalid = []

    for symbol in symbols:
        try:
            validated_symbol = validate_bist_symbol(symbol, allow_suffix=allow_suffix, auto_fix=auto_fix)
            validated.append(validated_symbol)
        except InvalidStockCodeError as e:
            if skip_invalid:
                invalid.append(symbol)
                logger.warning(f"Skipping invalid symbol '{symbol}': {str(e)}")
            else:
                raise

    if invalid and skip_invalid:
        logger.info(f"Skipped {len(invalid)} invalid symbols: {invalid}")

    return validated


def normalize_bist_symbol(symbol: str, add_suffix: bool = False, remove_suffix: bool = False) -> str:
    """
    Normalize BIST symbol to standard format.

    Args:
        symbol: Stock symbol
        add_suffix: If True, add .IS suffix if not present
        remove_suffix: If True, remove .IS suffix if present

    Returns:
        Normalized symbol

    Examples:
        >>> normalize_bist_symbol("thyao")
        'THYAO'
        >>> normalize_bist_symbol("thyao", add_suffix=True)
        'THYAO.IS'
        >>> normalize_bist_symbol("THYAO.IS", remove_suffix=True)
        'THYAO'
    """
    # Validate and auto-fix
    symbol = validate_bist_symbol(symbol, allow_suffix=True, auto_fix=True)

    # Handle suffix
    has_suffix = symbol.endswith('.IS')

    if add_suffix and not has_suffix:
        return symbol + '.IS'
    elif remove_suffix and has_suffix:
        return symbol[:-3]
    else:
        return symbol


def validate_bist_index(index_name: str) -> str:
    """
    Validate BIST index name.

    Valid indices: XU100, XU030, XU050, XBANK, XUSIN, etc.

    Args:
        index_name: Index name to validate

    Returns:
        Validated index name

    Raises:
        InvalidStockCodeError: If index name is invalid
    """
    if not index_name or not isinstance(index_name, str):
        raise InvalidStockCodeError(f"Index name must be a non-empty string")

    index_name = index_name.upper().strip()

    # Common BIST indices
    known_indices = {
        'XU100', 'XU030', 'XU050', 'XBANK', 'XUSIN', 'XHOLD', 'XUMAL',
        'XGIDA', 'XTEKS', 'XKURY', 'XUHIZ', 'XUTEK', 'XULAS', 'XYORT'
    }

    # Check if it's a known index
    if index_name in known_indices:
        return index_name

    # Otherwise validate format: starts with X, followed by letters/numbers
    if not re.match(r'^X[A-Z0-9]{2,8}$', index_name):
        raise InvalidStockCodeError(
            f"Invalid BIST index format: '{index_name}'. "
            f"Must start with 'X' followed by 2-8 letters/numbers. "
            f"Known indices: {', '.join(sorted(known_indices))}"
        )

    return index_name


# ============================================================================
# Date and Time Validators
# ============================================================================

def validate_date(
    date_value: Union[str, datetime, date],
    date_format: Optional[str] = None,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    allow_future: bool = True
) -> datetime:
    """
    Validate and convert date to datetime object.

    Args:
        date_value: Date as string, datetime, or date object
        date_format: Expected date format for string parsing (e.g., '%Y-%m-%d')
        min_date: Minimum allowed date
        max_date: Maximum allowed date
        allow_future: Whether to allow future dates

    Returns:
        Validated datetime object

    Raises:
        InvalidDateError: If date is invalid or out of range

    Examples:
        >>> validate_date("2024-01-15")
        datetime(2024, 1, 15, 0, 0)
        >>> validate_date("2024-01-15", allow_future=False)  # May raise if in future
    """
    # Convert to datetime
    try:
        if isinstance(date_value, datetime):
            dt = date_value
        elif isinstance(date_value, date):
            dt = datetime.combine(date_value, datetime.min.time())
        elif isinstance(date_value, str):
            if date_format:
                dt = datetime.strptime(date_value, date_format)
            else:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        dt = datetime.strptime(date_value, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # Try pandas parsing as last resort
                    import pandas as pd
                    dt = pd.to_datetime(date_value).to_pydatetime()
        else:
            raise InvalidDateError(f"Invalid date type: {type(date_value)}")
    except Exception as e:
        raise InvalidDateError(f"Failed to parse date '{date_value}': {str(e)}")

    # Validate range
    if not allow_future and dt > datetime.now():
        raise InvalidDateError(f"Future dates not allowed: {dt}")

    if min_date and dt < min_date:
        raise InvalidDateError(f"Date {dt} is before minimum date {min_date}")

    if max_date and dt > max_date:
        raise InvalidDateError(f"Date {dt} is after maximum date {max_date}")

    return dt


def validate_date_range(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    min_days: Optional[int] = None,
    max_days: Optional[int] = None,
    allow_same_day: bool = True
) -> Tuple[datetime, datetime]:
    """
    Validate a date range.

    Args:
        start_date: Start date
        end_date: End date
        min_days: Minimum number of days in range
        max_days: Maximum number of days in range
        allow_same_day: Whether start and end can be the same day

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        InvalidDateError: If date range is invalid
    """
    start_dt = validate_date(start_date)
    end_dt = validate_date(end_date)

    # Check order
    if start_dt > end_dt:
        raise InvalidDateError(f"Start date {start_dt} is after end date {end_dt}")

    if not allow_same_day and start_dt == end_dt:
        raise InvalidDateError(f"Start and end dates cannot be the same")

    # Check duration
    days_diff = (end_dt - start_dt).days

    if min_days and days_diff < min_days:
        raise InvalidDateError(
            f"Date range too short: {days_diff} days, minimum {min_days} days required"
        )

    if max_days and days_diff > max_days:
        raise InvalidDateError(
            f"Date range too long: {days_diff} days, maximum {max_days} days allowed"
        )

    return start_dt, end_dt


def validate_timeframe(timeframe: str) -> str:
    """
    Validate trading timeframe.

    Args:
        timeframe: Timeframe string (e.g., '1d', 'daily', '1h', '30min')

    Returns:
        Validated timeframe

    Raises:
        InvalidConfigurationError: If timeframe is invalid
    """
    if not timeframe or not isinstance(timeframe, str):
        raise InvalidConfigurationError("Timeframe must be a non-empty string")

    timeframe = timeframe.lower().strip()

    # Valid timeframes
    valid_timeframes = {
        # Daily
        'daily', '1d', 'd',
        # Hourly
        'hourly', '1h', 'h',
        # Minutes
        '1m', '1min', '5m', '5min', '15m', '15min', '30m', '30min',
        # Weekly/Monthly
        'weekly', '1w', 'w', 'monthly', '1mo', 'mo'
    }

    if timeframe not in valid_timeframes:
        raise InvalidConfigurationError(
            f"Invalid timeframe '{timeframe}'. Valid options: {', '.join(sorted(valid_timeframes))}"
        )

    return timeframe


# ============================================================================
# Price and Financial Value Validators
# ============================================================================

def validate_price(
    price: Union[int, float],
    min_price: float = 0.0,
    max_price: Optional[float] = None,
    allow_zero: bool = False,
    decimals: Optional[int] = None
) -> float:
    """
    Validate price value.

    Args:
        price: Price to validate
        min_price: Minimum allowed price
        max_price: Maximum allowed price
        allow_zero: Whether zero price is allowed
        decimals: If specified, round to this many decimals

    Returns:
        Validated price

    Raises:
        InvalidPriceError: If price is invalid

    Examples:
        >>> validate_price(100.50)
        100.5
        >>> validate_price(-10)
        InvalidPriceError: Price cannot be negative
        >>> validate_price(0, allow_zero=True)
        0.0
    """
    try:
        price = float(price)
    except (TypeError, ValueError):
        raise InvalidPriceError(f"Price must be a number, got: {type(price)}")

    # Check for NaN or infinity
    if price != price:  # NaN check
        raise InvalidPriceError("Price cannot be NaN")

    if abs(price) == float('inf'):
        raise InvalidPriceError("Price cannot be infinite")

    # Check sign
    if price < 0:
        raise InvalidPriceError(f"Price cannot be negative: {price}")

    if price == 0 and not allow_zero:
        raise InvalidPriceError("Price cannot be zero")

    # Check range
    if price < min_price:
        raise InvalidPriceError(f"Price {price} is below minimum {min_price}")

    if max_price is not None and price > max_price:
        raise InvalidPriceError(f"Price {price} exceeds maximum {max_price}")

    # Round if requested
    if decimals is not None:
        price = round(price, decimals)

    return price


def validate_ohlc_prices(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float
) -> Tuple[float, float, float, float]:
    """
    Validate OHLC price relationships.

    Ensures: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close

    Args:
        open_price: Opening price
        high_price: High price
        low_price: Low price
        close_price: Closing price

    Returns:
        Tuple of validated (open, high, low, close) prices

    Raises:
        InvalidPriceError: If price relationships are invalid
    """
    # Validate individual prices
    open_price = validate_price(open_price)
    high_price = validate_price(high_price)
    low_price = validate_price(low_price)
    close_price = validate_price(close_price)

    # Validate relationships
    if high_price < low_price:
        raise InvalidPriceError(f"High price {high_price} is less than low price {low_price}")

    if high_price < open_price:
        raise InvalidPriceError(f"High price {high_price} is less than open price {open_price}")

    if high_price < close_price:
        raise InvalidPriceError(f"High price {high_price} is less than close price {close_price}")

    if low_price > open_price:
        raise InvalidPriceError(f"Low price {low_price} is greater than open price {open_price}")

    if low_price > close_price:
        raise InvalidPriceError(f"Low price {low_price} is greater than close price {close_price}")

    return open_price, high_price, low_price, close_price


def validate_volume(
    volume: Union[int, float],
    min_volume: float = 0,
    allow_zero: bool = True
) -> float:
    """
    Validate trading volume.

    Args:
        volume: Trading volume
        min_volume: Minimum allowed volume
        allow_zero: Whether zero volume is allowed

    Returns:
        Validated volume

    Raises:
        InvalidQuantityError: If volume is invalid
    """
    try:
        volume = float(volume)
    except (TypeError, ValueError):
        raise InvalidQuantityError(f"Volume must be a number, got: {type(volume)}")

    if volume < 0:
        raise InvalidQuantityError(f"Volume cannot be negative: {volume}")

    if volume == 0 and not allow_zero:
        raise InvalidQuantityError("Volume cannot be zero")

    if volume < min_volume:
        raise InvalidQuantityError(f"Volume {volume} is below minimum {min_volume}")

    return volume


def validate_amount(
    amount: Union[int, float],
    min_amount: float = 0.0,
    max_amount: Optional[float] = None,
    allow_negative: bool = False,
    allow_zero: bool = True
) -> float:
    """
    Validate monetary amount (cash, cost, etc.).

    Args:
        amount: Amount to validate
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount
        allow_negative: Whether negative amounts are allowed
        allow_zero: Whether zero amount is allowed

    Returns:
        Validated amount

    Raises:
        InvalidPriceError: If amount is invalid
    """
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise InvalidPriceError(f"Amount must be a number, got: {type(amount)}")

    if amount != amount:  # NaN check
        raise InvalidPriceError("Amount cannot be NaN")

    if not allow_negative and amount < 0:
        raise InvalidPriceError(f"Amount cannot be negative: {amount}")

    if not allow_zero and amount == 0:
        raise InvalidPriceError("Amount cannot be zero")

    if amount < min_amount:
        raise InvalidPriceError(f"Amount {amount} is below minimum {min_amount}")

    if max_amount is not None and amount > max_amount:
        raise InvalidPriceError(f"Amount {amount} exceeds maximum {max_amount}")

    return amount


# ============================================================================
# Portfolio Input Validators
# ============================================================================

def validate_shares(
    shares: Union[int, float],
    min_shares: float = 0.0,
    allow_fractional: bool = True,
    allow_zero: bool = False
) -> float:
    """
    Validate number of shares.

    Args:
        shares: Number of shares
        min_shares: Minimum allowed shares
        allow_fractional: Whether fractional shares are allowed
        allow_zero: Whether zero shares is allowed

    Returns:
        Validated shares

    Raises:
        InvalidQuantityError: If shares is invalid

    Examples:
        >>> validate_shares(100)
        100.0
        >>> validate_shares(100.5, allow_fractional=False)
        InvalidQuantityError: Fractional shares not allowed
    """
    try:
        shares = float(shares)
    except (TypeError, ValueError):
        raise InvalidQuantityError(f"Shares must be a number, got: {type(shares)}")

    if shares < 0:
        raise InvalidQuantityError(f"Shares cannot be negative: {shares}")

    if shares == 0 and not allow_zero:
        raise InvalidQuantityError("Shares cannot be zero")

    if not allow_fractional and shares != int(shares):
        raise InvalidQuantityError(f"Fractional shares not allowed: {shares}")

    if shares < min_shares:
        raise InvalidQuantityError(f"Shares {shares} is below minimum {min_shares}")

    return shares


def validate_percentage(
    percentage: Union[int, float],
    min_pct: float = 0.0,
    max_pct: float = 100.0,
    allow_negative: bool = False
) -> float:
    """
    Validate percentage value.

    Args:
        percentage: Percentage to validate (0-100 scale)
        min_pct: Minimum allowed percentage
        max_pct: Maximum allowed percentage
        allow_negative: Whether negative percentages are allowed

    Returns:
        Validated percentage

    Raises:
        InvalidPercentageError: If percentage is invalid

    Examples:
        >>> validate_percentage(50.5)
        50.5
        >>> validate_percentage(150)
        InvalidPercentageError: Percentage exceeds maximum
    """
    try:
        percentage = float(percentage)
    except (TypeError, ValueError):
        raise InvalidPercentageError(f"Percentage must be a number, got: {type(percentage)}")

    if not allow_negative and percentage < 0:
        raise InvalidPercentageError(f"Percentage cannot be negative: {percentage}")

    if percentage < min_pct:
        raise InvalidPercentageError(f"Percentage {percentage} is below minimum {min_pct}")

    if percentage > max_pct:
        raise InvalidPercentageError(f"Percentage {percentage} exceeds maximum {max_pct}")

    return percentage


def validate_allocation(
    allocation: Union[float, dict],
    total_should_equal: float = 100.0,
    tolerance: float = 0.01
) -> Union[float, dict]:
    """
    Validate portfolio allocation (either single value or dict of allocations).

    Args:
        allocation: Allocation percentage or dict of {symbol: percentage}
        total_should_equal: What the allocations should sum to (typically 100.0)
        tolerance: Tolerance for sum check (as percentage points)

    Returns:
        Validated allocation

    Raises:
        InvalidPercentageError: If allocation is invalid

    Examples:
        >>> validate_allocation(50.0)
        50.0
        >>> validate_allocation({'THYAO': 30, 'GARAN': 70})
        {'THYAO': 30, 'GARAN': 70}
        >>> validate_allocation({'THYAO': 30, 'GARAN': 80})
        InvalidPercentageError: Allocations sum to 110.0, expected 100.0
    """
    if isinstance(allocation, (int, float)):
        return validate_percentage(allocation, min_pct=0.0, max_pct=100.0)

    elif isinstance(allocation, dict):
        total = 0.0
        validated = {}

        for symbol, pct in allocation.items():
            validated_pct = validate_percentage(pct, min_pct=0.0, max_pct=100.0)
            validated[symbol] = validated_pct
            total += validated_pct

        # Check if total equals expected value (with tolerance)
        if abs(total - total_should_equal) > tolerance:
            raise InvalidPercentageError(
                f"Allocations sum to {total:.2f}, expected {total_should_equal:.2f} "
                f"(tolerance: {tolerance})"
            )

        return validated

    else:
        raise InvalidPercentageError(
            f"Allocation must be a number or dict, got: {type(allocation)}"
        )


def validate_position_size(
    position_size: Union[int, float],
    portfolio_value: float,
    max_position_pct: float = 20.0
) -> float:
    """
    Validate position size relative to portfolio value.

    Args:
        position_size: Size of position in currency
        portfolio_value: Total portfolio value
        max_position_pct: Maximum allowed position size as percentage of portfolio

    Returns:
        Validated position size

    Raises:
        InvalidQuantityError: If position size is too large
    """
    position_size = validate_amount(position_size, min_amount=0.0)
    portfolio_value = validate_amount(portfolio_value, min_amount=0.0)

    if portfolio_value == 0:
        raise InvalidQuantityError("Portfolio value cannot be zero")

    position_pct = (position_size / portfolio_value) * 100

    if position_pct > max_position_pct:
        raise InvalidQuantityError(
            f"Position size {position_size:.2f} is {position_pct:.2f}% of portfolio, "
            f"exceeds maximum {max_position_pct}%"
        )

    return position_size


# ============================================================================
# Configuration Parameter Validators
# ============================================================================

def validate_cost_basis_method(method: str) -> str:
    """
    Validate cost basis calculation method.

    Args:
        method: Cost basis method ('FIFO', 'LIFO', 'AVERAGE', 'SPECIFIC')

    Returns:
        Validated method

    Raises:
        InvalidConfigurationError: If method is invalid
    """
    if not method or not isinstance(method, str):
        raise InvalidConfigurationError("Cost basis method must be a non-empty string")

    method = method.upper().strip()

    valid_methods = ['FIFO', 'LIFO', 'AVERAGE', 'SPECIFIC']

    if method not in valid_methods:
        raise InvalidConfigurationError(
            f"Invalid cost basis method '{method}'. Valid options: {', '.join(valid_methods)}"
        )

    return method


def validate_period(period: str) -> str:
    """
    Validate period string for historical data.

    Args:
        period: Period string (e.g., '1d', '5d', '1mo', '3mo', '1y', 'max')

    Returns:
        Validated period

    Raises:
        InvalidConfigurationError: If period is invalid
    """
    if not period or not isinstance(period, str):
        raise InvalidConfigurationError("Period must be a non-empty string")

    period = period.lower().strip()

    # Valid period patterns
    valid_periods = {
        '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    }

    # Also accept number + unit format
    if period not in valid_periods:
        if not re.match(r'^\d+[dmy]$', period):
            raise InvalidConfigurationError(
                f"Invalid period '{period}'. "
                f"Valid options: {', '.join(sorted(valid_periods))} or number+unit (e.g., '30d', '6m', '2y')"
            )

    return period


def validate_threshold(
    threshold: Union[int, float],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    threshold_name: str = "threshold"
) -> float:
    """
    Validate a threshold value.

    Args:
        threshold: Threshold value to validate
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
        threshold_name: Name of threshold for error messages

    Returns:
        Validated threshold

    Raises:
        InvalidConfigurationError: If threshold is invalid
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise InvalidConfigurationError(
            f"{threshold_name} must be a number, got: {type(threshold)}"
        )

    if min_threshold is not None and threshold < min_threshold:
        raise InvalidConfigurationError(
            f"{threshold_name} {threshold} is below minimum {min_threshold}"
        )

    if max_threshold is not None and threshold > max_threshold:
        raise InvalidConfigurationError(
            f"{threshold_name} {threshold} exceeds maximum {max_threshold}"
        )

    return threshold


def validate_signal_confidence(confidence: Union[int, float]) -> float:
    """
    Validate signal confidence score (0-1 or 0-100).

    Args:
        confidence: Confidence score

    Returns:
        Validated confidence (normalized to 0-1)

    Raises:
        InvalidConfigurationError: If confidence is invalid
    """
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        raise InvalidConfigurationError(
            f"Confidence must be a number, got: {type(confidence)}"
        )

    # Normalize if 0-100 scale
    if confidence > 1.0:
        if confidence > 100.0:
            raise InvalidConfigurationError(
                f"Confidence score {confidence} exceeds 100"
            )
        confidence = confidence / 100.0

    if confidence < 0.0:
        raise InvalidConfigurationError(
            f"Confidence score cannot be negative: {confidence}"
        )

    return confidence


def validate_risk_level(risk_level: Union[str, int, float]) -> str:
    """
    Validate risk level parameter.

    Args:
        risk_level: Risk level as string ('low', 'medium', 'high') or number (1-5)

    Returns:
        Validated risk level as string

    Raises:
        InvalidConfigurationError: If risk level is invalid
    """
    if isinstance(risk_level, str):
        risk_level = risk_level.lower().strip()
        valid_levels = ['low', 'medium', 'high', 'conservative', 'moderate', 'aggressive']

        if risk_level not in valid_levels:
            raise InvalidConfigurationError(
                f"Invalid risk level '{risk_level}'. Valid options: {', '.join(valid_levels)}"
            )

        return risk_level

    elif isinstance(risk_level, (int, float)):
        # Convert numeric risk level (1-5) to string
        risk_level = int(risk_level)

        if risk_level < 1 or risk_level > 5:
            raise InvalidConfigurationError(
                f"Numeric risk level must be 1-5, got: {risk_level}"
            )

        # Map to string
        risk_map = {1: 'low', 2: 'low', 3: 'medium', 4: 'high', 5: 'high'}
        return risk_map[risk_level]

    else:
        raise InvalidConfigurationError(
            f"Risk level must be string or number, got: {type(risk_level)}"
        )


# ============================================================================
# Validator Decorators
# ============================================================================

def validate_inputs(**validators):
    """
    Decorator to validate function inputs.

    Usage:
        @validate_inputs(
            symbol=validate_bist_symbol,
            price=lambda p: validate_price(p, min_price=0),
            shares=validate_shares
        )
        def buy_stock(symbol, price, shares):
            ...

    Args:
        **validators: Keyword arguments mapping parameter names to validator functions

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value is not None:  # Skip None values
                        try:
                            validated_value = validator(value)
                            bound.arguments[param_name] = validated_value
                        except ValidationError as e:
                            raise ValidationError(
                                f"Validation failed for parameter '{param_name}': {str(e)}"
                            )

            return func(*bound.args, **bound.kwargs)

        return wrapper
    return decorator


def validate_output(validator):
    """
    Decorator to validate function output.

    Usage:
        @validate_output(lambda x: validate_price(x, min_price=0))
        def calculate_price():
            return some_price

    Args:
        validator: Validator function to apply to return value

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            try:
                validated_result = validator(result)
                return validated_result
            except ValidationError as e:
                raise ValidationError(
                    f"Output validation failed for {func.__name__}: {str(e)}"
                )

        return wrapper
    return decorator


# ============================================================================
# Composite Validators
# ============================================================================

def validate_trade_parameters(
    symbol: str,
    shares: Union[int, float],
    price: float,
    order_type: str = 'MARKET',
    commission: float = 0.0
) -> dict:
    """
    Validate all parameters for a trade order.

    Args:
        symbol: Stock symbol
        shares: Number of shares
        price: Price per share
        order_type: Order type ('MARKET' or 'LIMIT')
        commission: Commission/fees

    Returns:
        Dictionary of validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    validated = {}

    # Validate symbol
    validated['symbol'] = validate_bist_symbol(symbol, auto_fix=True)

    # Validate shares
    validated['shares'] = validate_shares(shares, allow_fractional=True)

    # Validate price
    validated['price'] = validate_price(price, min_price=0.01)

    # Validate order type
    order_type = order_type.upper().strip()
    if order_type not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
        raise InvalidConfigurationError(
            f"Invalid order type '{order_type}'. Valid options: MARKET, LIMIT, STOP, STOP_LIMIT"
        )
    validated['order_type'] = order_type

    # Validate commission
    validated['commission'] = validate_amount(commission, min_amount=0.0, allow_negative=False)

    return validated


def validate_portfolio_config(config: dict) -> dict:
    """
    Validate portfolio configuration dictionary.

    Args:
        config: Configuration dictionary with portfolio parameters

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If any configuration parameter is invalid
    """
    validated = config.copy()

    # Validate initial cash if present
    if 'initial_cash' in config:
        validated['initial_cash'] = validate_amount(
            config['initial_cash'], min_amount=0.0
        )

    # Validate cost basis method if present
    if 'cost_basis_method' in config:
        validated['cost_basis_method'] = validate_cost_basis_method(
            config['cost_basis_method']
        )

    # Validate max position size if present
    if 'max_position_pct' in config:
        validated['max_position_pct'] = validate_percentage(
            config['max_position_pct'], min_pct=0.0, max_pct=100.0
        )

    # Validate risk level if present
    if 'risk_level' in config:
        validated['risk_level'] = validate_risk_level(config['risk_level'])

    return validated


# ============================================================================
# Convenience Functions
# ============================================================================

def is_valid_bist_symbol(symbol: str) -> bool:
    """Check if symbol is valid without raising exception."""
    try:
        validate_bist_symbol(symbol)
        return True
    except InvalidStockCodeError:
        return False


def is_valid_date(date_value: Union[str, datetime, date]) -> bool:
    """Check if date is valid without raising exception."""
    try:
        validate_date(date_value)
        return True
    except InvalidDateError:
        return False


def is_valid_price(price: Union[int, float]) -> bool:
    """Check if price is valid without raising exception."""
    try:
        validate_price(price)
        return True
    except InvalidPriceError:
        return False


# ============================================================================
# Examples and Tests
# ============================================================================

def example_usage():
    """Example usage of validators."""
    print("=" * 80)
    print("BIST Trading System - Validators Example Usage")
    print("=" * 80)

    # Stock symbol validation
    print("\n1. Stock Symbol Validation")
    print("-" * 80)

    try:
        symbol1 = validate_bist_symbol("thyao", auto_fix=True)
        print(f"✓ Valid symbol: {symbol1}")

        symbol2 = validate_bist_symbol("GARAN.IS")
        print(f"✓ Valid symbol with suffix: {symbol2}")

        symbols = validate_bist_symbols(["thyao", "garan", "isctr"], auto_fix=True)
        print(f"✓ Valid symbols: {symbols}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Date validation
    print("\n2. Date Validation")
    print("-" * 80)

    try:
        date1 = validate_date("2024-01-15")
        print(f"✓ Valid date: {date1}")

        start, end = validate_date_range("2024-01-01", "2024-12-31", min_days=30)
        print(f"✓ Valid date range: {start.date()} to {end.date()}")

        timeframe = validate_timeframe("1h")
        print(f"✓ Valid timeframe: {timeframe}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Price validation
    print("\n3. Price and Financial Value Validation")
    print("-" * 80)

    try:
        price = validate_price(250.50, min_price=0.01)
        print(f"✓ Valid price: {price}")

        o, h, l, c = validate_ohlc_prices(100.0, 105.0, 98.0, 103.0)
        print(f"✓ Valid OHLC: O={o}, H={h}, L={l}, C={c}")

        volume = validate_volume(1000000)
        print(f"✓ Valid volume: {volume:,.0f}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Portfolio input validation
    print("\n4. Portfolio Input Validation")
    print("-" * 80)

    try:
        shares = validate_shares(100.5, allow_fractional=True)
        print(f"✓ Valid shares: {shares}")

        pct = validate_percentage(25.5, min_pct=0, max_pct=100)
        print(f"✓ Valid percentage: {pct}%")

        allocation = validate_allocation({'THYAO': 30, 'GARAN': 40, 'ISCTR': 30})
        print(f"✓ Valid allocation: {allocation}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Configuration validation
    print("\n5. Configuration Parameter Validation")
    print("-" * 80)

    try:
        method = validate_cost_basis_method("FIFO")
        print(f"✓ Valid cost basis method: {method}")

        period = validate_period("1y")
        print(f"✓ Valid period: {period}")

        risk = validate_risk_level("medium")
        print(f"✓ Valid risk level: {risk}")

        confidence = validate_signal_confidence(0.85)
        print(f"✓ Valid confidence: {confidence}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Composite validation
    print("\n6. Composite Trade Validation")
    print("-" * 80)

    try:
        trade_params = validate_trade_parameters(
            symbol="thyao",
            shares=100,
            price=250.50,
            order_type="LIMIT",
            commission=5.0
        )
        print(f"✓ Valid trade parameters:")
        for key, value in trade_params.items():
            print(f"  {key}: {value}")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Error examples
    print("\n7. Error Examples")
    print("-" * 80)

    # Invalid symbol
    try:
        validate_bist_symbol("123")
    except InvalidStockCodeError as e:
        print(f"✓ Caught expected error: {e}")

    # Invalid price
    try:
        validate_price(-100)
    except InvalidPriceError as e:
        print(f"✓ Caught expected error: {e}")

    # Invalid allocation
    try:
        validate_allocation({'THYAO': 60, 'GARAN': 50})  # Sums to 110%
    except InvalidPercentageError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
