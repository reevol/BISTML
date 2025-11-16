"""
Valuation Metrics Calculator

This module provides functions to calculate various valuation metrics from financial statements
for BIST (Borsa Istanbul) stocks. It computes key ratios used in fundamental analysis including:

Valuation Ratios:
- P/E (Price-to-Earnings) Ratio
- P/B (Price-to-Book) Ratio
- P/S (Price-to-Sales) Ratio
- EV/EBITDA (Enterprise Value to EBITDA)
- EV/Sales (Enterprise Value to Sales)
- PEG (Price/Earnings to Growth) Ratio
- Dividend Yield

The module is designed to work with financial statements obtained from various data sources
including Yahoo Finance, and can handle both quarterly and annual reporting periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValuationError(Exception):
    """Base exception for valuation calculation errors"""
    pass


class InsufficientDataError(ValuationError):
    """Raised when insufficient data is available for calculations"""
    pass


@dataclass
class ValuationMetrics:
    """Data class for storing comprehensive valuation metrics"""
    ticker: str
    date: datetime

    # Price-based ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None

    # Enterprise value ratios
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None

    # Growth-adjusted ratios
    peg_ratio: Optional[float] = None

    # Dividend metrics
    dividend_yield: Optional[float] = None

    # Supporting data
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    current_price: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        """String representation of valuation metrics"""
        return (
            f"ValuationMetrics(ticker={self.ticker}, date={self.date.strftime('%Y-%m-%d')}, "
            f"P/E={self.pe_ratio:.2f if self.pe_ratio else 'N/A'}, "
            f"P/B={self.pb_ratio:.2f if self.pb_ratio else 'N/A'}, "
            f"P/S={self.ps_ratio:.2f if self.ps_ratio else 'N/A'})"
        )


class ValuationCalculator:
    """
    Calculator for fundamental valuation metrics.

    This class provides methods to calculate various valuation ratios from
    financial statements and market data.
    """

    def __init__(self, ticker: str, current_price: Optional[float] = None):
        """
        Initialize the ValuationCalculator.

        Args:
            ticker: Stock ticker symbol
            current_price: Current market price of the stock
        """
        self.ticker = ticker.upper().strip()
        self.current_price = current_price
        logger.info(f"Initialized ValuationCalculator for {self.ticker}")

    @staticmethod
    def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """
        Safely divide two numbers, handling None and zero values.

        Args:
            numerator: Numerator value
            denominator: Denominator value

        Returns:
            Division result or None if calculation is not possible
        """
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        if pd.isna(numerator) or pd.isna(denominator):
            return None
        return numerator / denominator

    @staticmethod
    def _get_field_value(
        df: pd.DataFrame,
        field_names: list,
        period: int = 0
    ) -> Optional[float]:
        """
        Get value from DataFrame trying multiple field name variations.

        Args:
            df: DataFrame containing financial data
            field_names: List of possible field names to try
            period: Column index to use (0 = most recent)

        Returns:
            Field value or None if not found
        """
        if df is None or df.empty:
            return None

        if period >= len(df.columns):
            return None

        for field in field_names:
            if field in df.index:
                value = df.loc[field, df.columns[period]]
                if pd.notna(value):
                    return float(value)

        return None

    def calculate_pe_ratio(
        self,
        price: Optional[float] = None,
        earnings_per_share: Optional[float] = None,
        net_income: Optional[float] = None,
        shares_outstanding: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Price-to-Earnings (P/E) Ratio.

        Formula: P/E = Price / Earnings Per Share

        Args:
            price: Current stock price (uses self.current_price if not provided)
            earnings_per_share: Earnings per share (EPS)
            net_income: Net income (used to calculate EPS if not provided)
            shares_outstanding: Number of shares outstanding

        Returns:
            P/E ratio or None if calculation fails
        """
        # Use provided price or instance price
        price = price or self.current_price

        if price is None:
            logger.warning(f"No price available for P/E calculation for {self.ticker}")
            return None

        # Calculate EPS if not provided
        eps = earnings_per_share
        if eps is None and net_income is not None and shares_outstanding is not None:
            eps = self._safe_divide(net_income, shares_outstanding)

        if eps is None or eps <= 0:
            logger.warning(f"Invalid or negative EPS for {self.ticker}")
            return None

        pe_ratio = self._safe_divide(price, eps)

        if pe_ratio is not None:
            logger.info(f"P/E Ratio for {self.ticker}: {pe_ratio:.2f}")

        return pe_ratio

    def calculate_pb_ratio(
        self,
        price: Optional[float] = None,
        book_value_per_share: Optional[float] = None,
        total_equity: Optional[float] = None,
        shares_outstanding: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Price-to-Book (P/B) Ratio.

        Formula: P/B = Price / Book Value Per Share

        Args:
            price: Current stock price
            book_value_per_share: Book value per share
            total_equity: Total shareholders' equity
            shares_outstanding: Number of shares outstanding

        Returns:
            P/B ratio or None if calculation fails
        """
        price = price or self.current_price

        if price is None:
            logger.warning(f"No price available for P/B calculation for {self.ticker}")
            return None

        # Calculate book value per share if not provided
        bvps = book_value_per_share
        if bvps is None and total_equity is not None and shares_outstanding is not None:
            bvps = self._safe_divide(total_equity, shares_outstanding)

        if bvps is None or bvps <= 0:
            logger.warning(f"Invalid or negative book value per share for {self.ticker}")
            return None

        pb_ratio = self._safe_divide(price, bvps)

        if pb_ratio is not None:
            logger.info(f"P/B Ratio for {self.ticker}: {pb_ratio:.2f}")

        return pb_ratio

    def calculate_ps_ratio(
        self,
        price: Optional[float] = None,
        sales_per_share: Optional[float] = None,
        total_revenue: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
        market_cap: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Price-to-Sales (P/S) Ratio.

        Formula: P/S = Price / Sales Per Share
        Alternative: P/S = Market Cap / Total Revenue

        Args:
            price: Current stock price
            sales_per_share: Sales (revenue) per share
            total_revenue: Total revenue
            shares_outstanding: Number of shares outstanding
            market_cap: Market capitalization

        Returns:
            P/S ratio or None if calculation fails
        """
        # Method 1: Use market cap and total revenue
        if market_cap is not None and total_revenue is not None:
            ps_ratio = self._safe_divide(market_cap, total_revenue)
            if ps_ratio is not None:
                logger.info(f"P/S Ratio for {self.ticker}: {ps_ratio:.2f}")
                return ps_ratio

        # Method 2: Use price and sales per share
        price = price or self.current_price

        if price is None:
            logger.warning(f"No price available for P/S calculation for {self.ticker}")
            return None

        # Calculate sales per share if not provided
        sps = sales_per_share
        if sps is None and total_revenue is not None and shares_outstanding is not None:
            sps = self._safe_divide(total_revenue, shares_outstanding)

        if sps is None or sps <= 0:
            logger.warning(f"Invalid or negative sales per share for {self.ticker}")
            return None

        ps_ratio = self._safe_divide(price, sps)

        if ps_ratio is not None:
            logger.info(f"P/S Ratio for {self.ticker}: {ps_ratio:.2f}")

        return ps_ratio

    def calculate_ev_ebitda(
        self,
        market_cap: Optional[float] = None,
        total_debt: Optional[float] = None,
        cash: Optional[float] = None,
        ebitda: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Enterprise Value to EBITDA Ratio.

        Formula: EV/EBITDA = (Market Cap + Total Debt - Cash) / EBITDA

        Args:
            market_cap: Market capitalization
            total_debt: Total debt (long-term + short-term)
            cash: Cash and cash equivalents
            ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization

        Returns:
            EV/EBITDA ratio or None if calculation fails
        """
        if market_cap is None:
            logger.warning(f"Market cap required for EV/EBITDA calculation for {self.ticker}")
            return None

        # Calculate enterprise value
        enterprise_value = market_cap

        if total_debt is not None:
            enterprise_value += total_debt

        if cash is not None:
            enterprise_value -= cash

        if ebitda is None or ebitda <= 0:
            logger.warning(f"Invalid or negative EBITDA for {self.ticker}")
            return None

        ev_ebitda = self._safe_divide(enterprise_value, ebitda)

        if ev_ebitda is not None:
            logger.info(f"EV/EBITDA for {self.ticker}: {ev_ebitda:.2f}")

        return ev_ebitda

    def calculate_ev_sales(
        self,
        market_cap: Optional[float] = None,
        total_debt: Optional[float] = None,
        cash: Optional[float] = None,
        total_revenue: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Enterprise Value to Sales Ratio.

        Formula: EV/Sales = (Market Cap + Total Debt - Cash) / Total Revenue

        Args:
            market_cap: Market capitalization
            total_debt: Total debt (long-term + short-term)
            cash: Cash and cash equivalents
            total_revenue: Total revenue/sales

        Returns:
            EV/Sales ratio or None if calculation fails
        """
        if market_cap is None:
            logger.warning(f"Market cap required for EV/Sales calculation for {self.ticker}")
            return None

        # Calculate enterprise value
        enterprise_value = market_cap

        if total_debt is not None:
            enterprise_value += total_debt

        if cash is not None:
            enterprise_value -= cash

        if total_revenue is None or total_revenue <= 0:
            logger.warning(f"Invalid or negative revenue for {self.ticker}")
            return None

        ev_sales = self._safe_divide(enterprise_value, total_revenue)

        if ev_sales is not None:
            logger.info(f"EV/Sales for {self.ticker}: {ev_sales:.2f}")

        return ev_sales

    def calculate_peg_ratio(
        self,
        pe_ratio: Optional[float] = None,
        earnings_growth_rate: Optional[float] = None,
        price: Optional[float] = None,
        eps_current: Optional[float] = None,
        eps_previous: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate PEG (Price/Earnings to Growth) Ratio.

        Formula: PEG = P/E Ratio / Earnings Growth Rate (%)

        Args:
            pe_ratio: Price-to-Earnings ratio
            earnings_growth_rate: Annual earnings growth rate (as percentage, e.g., 15 for 15%)
            price: Current stock price (for calculating P/E if not provided)
            eps_current: Current EPS (for calculating growth if not provided)
            eps_previous: Previous year EPS (for calculating growth if not provided)

        Returns:
            PEG ratio or None if calculation fails
        """
        # Calculate earnings growth rate if not provided
        growth_rate = earnings_growth_rate
        if growth_rate is None and eps_current is not None and eps_previous is not None:
            if eps_previous > 0:
                # Calculate annual growth rate as percentage
                growth_rate = ((eps_current - eps_previous) / eps_previous) * 100
            else:
                logger.warning(f"Cannot calculate growth rate with negative previous EPS for {self.ticker}")
                return None

        # Use provided P/E or calculate it
        pe = pe_ratio
        if pe is None and price is not None and eps_current is not None:
            pe = self.calculate_pe_ratio(price=price, earnings_per_share=eps_current)

        if pe is None:
            logger.warning(f"P/E ratio required for PEG calculation for {self.ticker}")
            return None

        if growth_rate is None or growth_rate <= 0:
            logger.warning(f"Invalid or negative growth rate for {self.ticker}")
            return None

        peg_ratio = self._safe_divide(pe, growth_rate)

        if peg_ratio is not None:
            logger.info(f"PEG Ratio for {self.ticker}: {peg_ratio:.2f}")

        return peg_ratio

    def calculate_dividend_yield(
        self,
        annual_dividend_per_share: Optional[float] = None,
        price: Optional[float] = None,
        total_dividends: Optional[float] = None,
        shares_outstanding: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate Dividend Yield.

        Formula: Dividend Yield = (Annual Dividend Per Share / Price) * 100

        Args:
            annual_dividend_per_share: Annual dividend per share
            price: Current stock price
            total_dividends: Total dividends paid
            shares_outstanding: Number of shares outstanding

        Returns:
            Dividend yield as percentage or None if calculation fails
        """
        price = price or self.current_price

        if price is None:
            logger.warning(f"No price available for dividend yield calculation for {self.ticker}")
            return None

        # Calculate dividend per share if not provided
        dps = annual_dividend_per_share
        if dps is None and total_dividends is not None and shares_outstanding is not None:
            dps = self._safe_divide(total_dividends, shares_outstanding)

        if dps is None:
            logger.info(f"No dividend data available for {self.ticker}")
            return 0.0  # Return 0 for stocks that don't pay dividends

        if dps < 0:
            logger.warning(f"Negative dividend per share for {self.ticker}")
            return None

        # Calculate yield as percentage
        dividend_yield = self._safe_divide(dps, price)
        if dividend_yield is not None:
            dividend_yield *= 100
            logger.info(f"Dividend Yield for {self.ticker}: {dividend_yield:.2f}%")

        return dividend_yield

    def calculate_from_statements(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict] = None
    ) -> ValuationMetrics:
        """
        Calculate all valuation metrics from financial statements.

        Args:
            income_statement: Income statement DataFrame (columns = periods, rows = line items)
            balance_sheet: Balance sheet DataFrame
            cash_flow: Cash flow statement DataFrame (optional)
            market_data: Dictionary with market data (price, market_cap, shares_outstanding)

        Returns:
            ValuationMetrics object with all calculated ratios
        """
        logger.info(f"Calculating comprehensive valuation metrics for {self.ticker}")

        # Initialize metrics object
        metrics = ValuationMetrics(
            ticker=self.ticker,
            date=datetime.now()
        )

        # Extract market data
        if market_data:
            price = market_data.get('current_price', self.current_price)
            market_cap = market_data.get('market_cap')
            shares_outstanding = market_data.get('shares_outstanding')
        else:
            price = self.current_price
            market_cap = None
            shares_outstanding = None

        metrics.current_price = price
        metrics.market_cap = market_cap

        try:
            # Extract data from income statement
            net_income = self._get_field_value(
                income_statement,
                ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operations']
            )

            total_revenue = self._get_field_value(
                income_statement,
                ['Total Revenue', 'Revenue', 'Total Operating Revenue']
            )

            ebitda = self._get_field_value(
                income_statement,
                ['EBITDA', 'Normalized EBITDA']
            )

            # If EBITDA not available, try to calculate it
            if ebitda is None:
                operating_income = self._get_field_value(
                    income_statement,
                    ['Operating Income', 'Operating Revenue']
                )
                depreciation = self._get_field_value(
                    income_statement,
                    ['Reconciled Depreciation', 'Depreciation And Amortization']
                )
                if operating_income is not None:
                    ebitda = operating_income + (depreciation or 0)

            # Extract data from balance sheet
            total_equity = self._get_field_value(
                balance_sheet,
                ['Total Equity Gross Minority Interest', 'Stockholders Equity',
                 'Total Stockholder Equity', 'Common Stock Equity']
            )

            total_debt = 0
            long_term_debt = self._get_field_value(
                balance_sheet,
                ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation']
            )
            short_term_debt = self._get_field_value(
                balance_sheet,
                ['Current Debt', 'Short Term Debt', 'Short Long Term Debt']
            )

            if long_term_debt:
                total_debt += long_term_debt
            if short_term_debt:
                total_debt += short_term_debt

            cash = self._get_field_value(
                balance_sheet,
                ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash']
            )

            # Extract dividends from cash flow statement
            dividends = None
            if cash_flow is not None and not cash_flow.empty:
                dividends = self._get_field_value(
                    cash_flow,
                    ['Cash Dividends Paid', 'Common Stock Dividend Paid', 'Dividends Paid']
                )
                # Dividends are typically negative in cash flow statements
                if dividends is not None and dividends < 0:
                    dividends = abs(dividends)

            # Calculate enterprise value
            if market_cap is not None:
                enterprise_value = market_cap + total_debt - (cash or 0)
                metrics.enterprise_value = enterprise_value

            # Calculate P/E Ratio
            if price and net_income and shares_outstanding:
                eps = net_income / shares_outstanding
                metrics.pe_ratio = self.calculate_pe_ratio(
                    price=price,
                    earnings_per_share=eps
                )

            # Calculate P/B Ratio
            if price and total_equity and shares_outstanding:
                bvps = total_equity / shares_outstanding
                metrics.pb_ratio = self.calculate_pb_ratio(
                    price=price,
                    book_value_per_share=bvps
                )

            # Calculate P/S Ratio
            if market_cap and total_revenue:
                metrics.ps_ratio = self.calculate_ps_ratio(
                    market_cap=market_cap,
                    total_revenue=total_revenue
                )
            elif price and total_revenue and shares_outstanding:
                sps = total_revenue / shares_outstanding
                metrics.ps_ratio = self.calculate_ps_ratio(
                    price=price,
                    sales_per_share=sps
                )

            # Calculate EV/EBITDA
            if market_cap and ebitda:
                metrics.ev_ebitda = self.calculate_ev_ebitda(
                    market_cap=market_cap,
                    total_debt=total_debt,
                    cash=cash,
                    ebitda=ebitda
                )

            # Calculate EV/Sales
            if market_cap and total_revenue:
                metrics.ev_sales = self.calculate_ev_sales(
                    market_cap=market_cap,
                    total_debt=total_debt,
                    cash=cash,
                    total_revenue=total_revenue
                )

            # Calculate PEG Ratio (requires historical data)
            if len(income_statement.columns) >= 2:
                # Get current and previous year net income
                net_income_current = self._get_field_value(income_statement, ['Net Income'], period=0)
                net_income_previous = self._get_field_value(income_statement, ['Net Income'], period=1)

                if net_income_current and net_income_previous and shares_outstanding:
                    eps_current = net_income_current / shares_outstanding
                    eps_previous = net_income_previous / shares_outstanding

                    metrics.peg_ratio = self.calculate_peg_ratio(
                        pe_ratio=metrics.pe_ratio,
                        eps_current=eps_current,
                        eps_previous=eps_previous
                    )

            # Calculate Dividend Yield
            if price and dividends and shares_outstanding:
                metrics.dividend_yield = self.calculate_dividend_yield(
                    annual_dividend_per_share=dividends / shares_outstanding,
                    price=price
                )
            elif price:
                # Check if company pays no dividends
                metrics.dividend_yield = 0.0

            logger.info(f"Successfully calculated valuation metrics for {self.ticker}")

        except Exception as e:
            logger.error(f"Error calculating metrics from statements for {self.ticker}: {str(e)}")

        return metrics


def compare_valuations(metrics_list: list) -> pd.DataFrame:
    """
    Compare valuation metrics across multiple stocks.

    Args:
        metrics_list: List of ValuationMetrics objects

    Returns:
        DataFrame with comparative metrics
    """
    if not metrics_list:
        return pd.DataFrame()

    data = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame(data)

    # Select relevant columns
    columns = [
        'ticker', 'pe_ratio', 'pb_ratio', 'ps_ratio',
        'ev_ebitda', 'ev_sales', 'peg_ratio', 'dividend_yield'
    ]

    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]

    return df


def identify_value_stocks(
    metrics_list: list,
    pe_threshold: float = 15.0,
    pb_threshold: float = 2.0,
    peg_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Identify potentially undervalued stocks based on valuation thresholds.

    Args:
        metrics_list: List of ValuationMetrics objects
        pe_threshold: Maximum P/E ratio for value stocks
        pb_threshold: Maximum P/B ratio for value stocks
        peg_threshold: Maximum PEG ratio for value stocks

    Returns:
        DataFrame with stocks meeting value criteria
    """
    df = compare_valuations(metrics_list)

    if df.empty:
        return df

    # Apply filters
    value_stocks = df[
        ((df['pe_ratio'] <= pe_threshold) | df['pe_ratio'].isna()) &
        ((df['pb_ratio'] <= pb_threshold) | df['pb_ratio'].isna()) &
        ((df['peg_ratio'] <= peg_threshold) | df['peg_ratio'].isna())
    ].copy()

    # Add value score (lower is better)
    value_stocks['value_score'] = (
        value_stocks['pe_ratio'].fillna(pe_threshold) / pe_threshold +
        value_stocks['pb_ratio'].fillna(pb_threshold) / pb_threshold +
        value_stocks['peg_ratio'].fillna(peg_threshold) / peg_threshold
    ) / 3

    value_stocks = value_stocks.sort_values('value_score')

    return value_stocks


# Example usage
if __name__ == "__main__":
    # Example: Calculate valuation metrics from sample data
    print("="*80)
    print("Valuation Metrics Calculator - Example Usage")
    print("="*80)

    # Sample income statement data (simplified)
    income_data = {
        '2023-12-31': {
            'Total Revenue': 10_000_000_000,
            'Net Income': 1_500_000_000,
            'EBITDA': 2_500_000_000
        },
        '2022-12-31': {
            'Total Revenue': 9_000_000_000,
            'Net Income': 1_200_000_000,
            'EBITDA': 2_200_000_000
        }
    }

    # Sample balance sheet data
    balance_data = {
        '2023-12-31': {
            'Total Equity Gross Minority Interest': 8_000_000_000,
            'Long Term Debt': 3_000_000_000,
            'Short Term Debt': 500_000_000,
            'Cash And Cash Equivalents': 1_000_000_000
        }
    }

    # Sample cash flow data
    cashflow_data = {
        '2023-12-31': {
            'Cash Dividends Paid': -300_000_000
        }
    }

    # Create DataFrames
    income_stmt = pd.DataFrame(income_data)
    balance_sheet = pd.DataFrame(balance_data)
    cash_flow = pd.DataFrame(cashflow_data)

    # Market data
    market_data = {
        'current_price': 25.50,
        'market_cap': 15_000_000_000,
        'shares_outstanding': 588_235_294
    }

    # Initialize calculator
    calculator = ValuationCalculator(
        ticker='SAMPLE',
        current_price=market_data['current_price']
    )

    # Calculate metrics
    metrics = calculator.calculate_from_statements(
        income_statement=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        market_data=market_data
    )

    # Display results
    print("\nValuation Metrics:")
    print(f"Ticker: {metrics.ticker}")
    print(f"Date: {metrics.date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Price: ${metrics.current_price:.2f}" if metrics.current_price else "Current Price: N/A")
    print(f"Market Cap: ${metrics.market_cap:,.0f}" if metrics.market_cap else "Market Cap: N/A")
    print(f"Enterprise Value: ${metrics.enterprise_value:,.0f}" if metrics.enterprise_value else "Enterprise Value: N/A")

    print("\nPrice Ratios:")
    print(f"P/E Ratio: {metrics.pe_ratio:.2f}" if metrics.pe_ratio else "P/E Ratio: N/A")
    print(f"P/B Ratio: {metrics.pb_ratio:.2f}" if metrics.pb_ratio else "P/B Ratio: N/A")
    print(f"P/S Ratio: {metrics.ps_ratio:.2f}" if metrics.ps_ratio else "P/S Ratio: N/A")

    print("\nEnterprise Value Ratios:")
    print(f"EV/EBITDA: {metrics.ev_ebitda:.2f}" if metrics.ev_ebitda else "EV/EBITDA: N/A")
    print(f"EV/Sales: {metrics.ev_sales:.2f}" if metrics.ev_sales else "EV/Sales: N/A")

    print("\nGrowth & Income:")
    print(f"PEG Ratio: {metrics.peg_ratio:.2f}" if metrics.peg_ratio else "PEG Ratio: N/A")
    print(f"Dividend Yield: {metrics.dividend_yield:.2f}%" if metrics.dividend_yield else "Dividend Yield: N/A")

    print("\n" + "="*80)
    print("Example calculation completed successfully!")
    print("="*80)
