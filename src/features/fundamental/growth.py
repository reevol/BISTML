"""
Fundamental Growth Analysis Module

This module provides comprehensive fundamental growth metrics and trend analysis
for the BIST AI Trading System. It calculates various growth rates and profitability
metrics over time to identify companies with strong fundamental momentum.

Metrics included:
- Revenue Growth (YoY, QoQ, Multi-period CAGR)
- Earnings Growth (YoY, QoQ, Multi-period CAGR)
- Book Value Growth (YoY, Multi-period)
- ROE (Return on Equity) and trends
- ROA (Return on Assets) and trends
- Profit Margin Trends (Gross, Operating, Net)
- EPS Growth
- Free Cash Flow Growth

Author: BISTML Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GrowthMetrics:
    """
    A comprehensive class for calculating fundamental growth metrics and trends.

    This class analyzes financial statement data to compute growth rates,
    profitability trends, and efficiency metrics over time.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the GrowthMetrics class.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing financial data with datetime index
            Expected columns depend on which metrics are being calculated
        """
        self.data = data.copy() if data is not None else None

    @staticmethod
    def calculate_growth_rate(
        current: float,
        previous: float,
        periods: int = 1,
        annualize: bool = False
    ) -> Optional[float]:
        """
        Calculate growth rate between two values.

        Parameters:
        -----------
        current : float
            Current period value
        previous : float
            Previous period value
        periods : int, default=1
            Number of periods between values
        annualize : bool, default=False
            If True, annualize the growth rate (CAGR)

        Returns:
        --------
        float or None
            Growth rate as decimal (e.g., 0.15 = 15% growth)
        """
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return None

        if annualize and periods > 1:
            # Calculate CAGR: (Ending/Beginning)^(1/periods) - 1
            growth_rate = (current / previous) ** (1 / periods) - 1
        else:
            # Simple growth rate
            growth_rate = (current - previous) / abs(previous)

        return growth_rate

    @staticmethod
    def calculate_revenue_growth(
        revenue_series: pd.Series,
        periods: int = 1,
        method: str = 'yoy'
    ) -> pd.Series:
        """
        Calculate revenue growth rates.

        Parameters:
        -----------
        revenue_series : pd.Series
            Time series of revenue values with datetime index
        periods : int, default=1
            Number of periods to look back
            For quarterly data: 1=QoQ, 4=YoY
            For annual data: 1=YoY
        method : str, default='yoy'
            Growth calculation method: 'yoy', 'qoq', 'mom', or 'cagr'

        Returns:
        --------
        pd.Series
            Revenue growth rates
        """
        if method == 'cagr' and periods > 1:
            # Calculate CAGR for each point using specified lookback
            growth = pd.Series(index=revenue_series.index, dtype=float)
            for i in range(periods, len(revenue_series)):
                current = revenue_series.iloc[i]
                previous = revenue_series.iloc[i - periods]
                growth.iloc[i] = GrowthMetrics.calculate_growth_rate(
                    current, previous, periods, annualize=True
                )
            return growth
        else:
            # Simple period-over-period growth
            shifted = revenue_series.shift(periods)
            growth = (revenue_series - shifted) / shifted.abs()
            return growth

    @staticmethod
    def calculate_earnings_growth(
        earnings_series: pd.Series,
        periods: int = 1,
        method: str = 'yoy'
    ) -> pd.Series:
        """
        Calculate earnings (net income) growth rates.

        Parameters:
        -----------
        earnings_series : pd.Series
            Time series of earnings/net income values
        periods : int, default=1
            Number of periods to look back
        method : str, default='yoy'
            Growth calculation method: 'yoy', 'qoq', or 'cagr'

        Returns:
        --------
        pd.Series
            Earnings growth rates
        """
        # Handle negative earnings carefully
        growth = pd.Series(index=earnings_series.index, dtype=float)

        for i in range(periods, len(earnings_series)):
            current = earnings_series.iloc[i]
            previous = earnings_series.iloc[i - periods]

            if pd.isna(current) or pd.isna(previous):
                growth.iloc[i] = np.nan
                continue

            # Handle cases where earnings go from negative to positive or vice versa
            if previous == 0:
                growth.iloc[i] = np.nan
            elif previous < 0 and current < 0:
                # Both negative: improvement is reduction in loss
                growth.iloc[i] = (current - previous) / abs(previous)
            elif previous < 0 and current >= 0:
                # Turned profitable
                growth.iloc[i] = np.inf if current > 0 else 0
            elif previous > 0 and current < 0:
                # Turned unprofitable
                growth.iloc[i] = -np.inf
            else:
                # Both positive: normal growth calculation
                if method == 'cagr' and periods > 1:
                    growth.iloc[i] = (current / previous) ** (1 / periods) - 1
                else:
                    growth.iloc[i] = (current - previous) / previous

        return growth

    @staticmethod
    def calculate_book_value_growth(
        equity_series: pd.Series,
        periods: int = 1,
        per_share: bool = False,
        shares_outstanding: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate book value (shareholders' equity) growth.

        Parameters:
        -----------
        equity_series : pd.Series
            Time series of total shareholders' equity
        periods : int, default=1
            Number of periods to look back
        per_share : bool, default=False
            If True, calculate book value per share growth
        shares_outstanding : pd.Series, optional
            Required if per_share=True

        Returns:
        --------
        pd.Series
            Book value growth rates
        """
        if per_share:
            if shares_outstanding is None:
                raise ValueError("shares_outstanding required for per-share calculation")
            # Calculate book value per share
            bvps = equity_series / shares_outstanding
            return GrowthMetrics.calculate_revenue_growth(bvps, periods)
        else:
            return GrowthMetrics.calculate_revenue_growth(equity_series, periods)

    @staticmethod
    def calculate_eps_growth(
        eps_series: pd.Series,
        periods: int = 1,
        method: str = 'yoy'
    ) -> pd.Series:
        """
        Calculate Earnings Per Share (EPS) growth.

        Parameters:
        -----------
        eps_series : pd.Series
            Time series of EPS values
        periods : int, default=1
            Number of periods to look back
        method : str, default='yoy'
            Growth calculation method

        Returns:
        --------
        pd.Series
            EPS growth rates
        """
        return GrowthMetrics.calculate_earnings_growth(eps_series, periods, method)

    @staticmethod
    def calculate_fcf_growth(
        fcf_series: pd.Series,
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate Free Cash Flow (FCF) growth.

        Parameters:
        -----------
        fcf_series : pd.Series
            Time series of free cash flow values
        periods : int, default=1
            Number of periods to look back

        Returns:
        --------
        pd.Series
            FCF growth rates
        """
        return GrowthMetrics.calculate_earnings_growth(fcf_series, periods)

    @staticmethod
    def calculate_roe(
        net_income: pd.Series,
        shareholders_equity: pd.Series,
        average_equity: bool = True
    ) -> pd.Series:
        """
        Calculate Return on Equity (ROE).

        ROE = Net Income / Shareholders' Equity

        Parameters:
        -----------
        net_income : pd.Series
            Time series of net income
        shareholders_equity : pd.Series
            Time series of shareholders' equity
        average_equity : bool, default=True
            If True, use average of beginning and ending equity

        Returns:
        --------
        pd.Series
            ROE values (as decimal, e.g., 0.15 = 15%)
        """
        if average_equity:
            # Use average of current and previous period equity
            avg_equity = (shareholders_equity + shareholders_equity.shift(1)) / 2
            roe = net_income / avg_equity
        else:
            roe = net_income / shareholders_equity

        return roe

    @staticmethod
    def calculate_roa(
        net_income: pd.Series,
        total_assets: pd.Series,
        average_assets: bool = True
    ) -> pd.Series:
        """
        Calculate Return on Assets (ROA).

        ROA = Net Income / Total Assets

        Parameters:
        -----------
        net_income : pd.Series
            Time series of net income
        total_assets : pd.Series
            Time series of total assets
        average_assets : bool, default=True
            If True, use average of beginning and ending assets

        Returns:
        --------
        pd.Series
            ROA values (as decimal)
        """
        if average_assets:
            # Use average of current and previous period assets
            avg_assets = (total_assets + total_assets.shift(1)) / 2
            roa = net_income / avg_assets
        else:
            roa = net_income / total_assets

        return roa

    @staticmethod
    def calculate_roic(
        nopat: pd.Series,
        invested_capital: pd.Series,
        average_capital: bool = True
    ) -> pd.Series:
        """
        Calculate Return on Invested Capital (ROIC).

        ROIC = NOPAT / Invested Capital
        where NOPAT = Net Operating Profit After Tax
        Invested Capital = Total Debt + Equity - Cash

        Parameters:
        -----------
        nopat : pd.Series
            Net Operating Profit After Tax
        invested_capital : pd.Series
            Total invested capital
        average_capital : bool, default=True
            If True, use average capital

        Returns:
        --------
        pd.Series
            ROIC values
        """
        if average_capital:
            avg_capital = (invested_capital + invested_capital.shift(1)) / 2
            roic = nopat / avg_capital
        else:
            roic = nopat / invested_capital

        return roic

    @staticmethod
    def calculate_profit_margins(
        revenue: pd.Series,
        gross_profit: Optional[pd.Series] = None,
        operating_income: Optional[pd.Series] = None,
        net_income: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate various profit margins.

        Parameters:
        -----------
        revenue : pd.Series
            Total revenue
        gross_profit : pd.Series, optional
            Gross profit
        operating_income : pd.Series, optional
            Operating income
        net_income : pd.Series, optional
            Net income

        Returns:
        --------
        dict
            Dictionary with 'gross_margin', 'operating_margin', 'net_margin' Series
        """
        margins = {}

        if gross_profit is not None:
            margins['gross_margin'] = gross_profit / revenue

        if operating_income is not None:
            margins['operating_margin'] = operating_income / revenue

        if net_income is not None:
            margins['net_margin'] = net_income / revenue

        return margins

    @staticmethod
    def calculate_margin_trend(
        margin_series: pd.Series,
        periods: int = 4,
        method: str = 'slope'
    ) -> pd.Series:
        """
        Calculate trend in profit margins over time.

        Parameters:
        -----------
        margin_series : pd.Series
            Time series of margin values
        periods : int, default=4
            Number of periods to use for trend calculation
        method : str, default='slope'
            Trend method: 'slope' (linear regression) or 'change' (simple change)

        Returns:
        --------
        pd.Series
            Margin trend indicator
        """
        if method == 'slope':
            # Calculate rolling linear regression slope
            def calc_slope(y_values):
                if len(y_values) < 2 or y_values.isna().any():
                    return np.nan
                x = np.arange(len(y_values))
                slope = np.polyfit(x, y_values, 1)[0]
                return slope

            trend = margin_series.rolling(window=periods).apply(calc_slope, raw=False)
        else:
            # Simple change over period
            trend = margin_series - margin_series.shift(periods)

        return trend

    @staticmethod
    def calculate_roe_trend(
        roe_series: pd.Series,
        periods: int = 4
    ) -> pd.Series:
        """
        Calculate trend in ROE over time.

        Parameters:
        -----------
        roe_series : pd.Series
            Time series of ROE values
        periods : int, default=4
            Number of periods for trend calculation

        Returns:
        --------
        pd.Series
            ROE trend (positive = improving, negative = deteriorating)
        """
        return GrowthMetrics.calculate_margin_trend(roe_series, periods, method='slope')

    @staticmethod
    def calculate_roa_trend(
        roa_series: pd.Series,
        periods: int = 4
    ) -> pd.Series:
        """
        Calculate trend in ROA over time.

        Parameters:
        -----------
        roa_series : pd.Series
            Time series of ROA values
        periods : int, default=4
            Number of periods for trend calculation

        Returns:
        --------
        pd.Series
            ROA trend
        """
        return GrowthMetrics.calculate_margin_trend(roa_series, periods, method='slope')

    @staticmethod
    def calculate_dupont_roe(
        net_margin: pd.Series,
        asset_turnover: pd.Series,
        equity_multiplier: pd.Series
    ) -> pd.Series:
        """
        Calculate ROE using DuPont analysis.

        DuPont ROE = Net Margin × Asset Turnover × Equity Multiplier

        Parameters:
        -----------
        net_margin : pd.Series
            Net profit margin (Net Income / Revenue)
        asset_turnover : pd.Series
            Asset turnover ratio (Revenue / Total Assets)
        equity_multiplier : pd.Series
            Equity multiplier (Total Assets / Shareholders' Equity)

        Returns:
        --------
        pd.Series
            ROE calculated using DuPont formula
        """
        return net_margin * asset_turnover * equity_multiplier

    @staticmethod
    def calculate_sustainable_growth_rate(
        roe: pd.Series,
        payout_ratio: pd.Series
    ) -> pd.Series:
        """
        Calculate Sustainable Growth Rate.

        SGR = ROE × (1 - Payout Ratio)

        This represents the rate at which a company can grow using internally
        generated equity without needing external financing.

        Parameters:
        -----------
        roe : pd.Series
            Return on Equity
        payout_ratio : pd.Series
            Dividend payout ratio (Dividends / Net Income)

        Returns:
        --------
        pd.Series
            Sustainable growth rate
        """
        retention_ratio = 1 - payout_ratio
        sgr = roe * retention_ratio
        return sgr

    @staticmethod
    def detect_growth_acceleration(
        growth_series: pd.Series,
        periods: int = 3
    ) -> pd.Series:
        """
        Detect acceleration or deceleration in growth rates.

        Parameters:
        -----------
        growth_series : pd.Series
            Time series of growth rates
        periods : int, default=3
            Lookback period for acceleration calculation

        Returns:
        --------
        pd.Series
            Growth acceleration (positive = accelerating, negative = decelerating)
        """
        # Calculate change in growth rate
        acceleration = growth_series.diff(periods)
        return acceleration

    @staticmethod
    def calculate_quality_score(
        roe: pd.Series,
        roa: pd.Series,
        revenue_growth: pd.Series,
        margin_trend: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate a composite fundamental quality score.

        Parameters:
        -----------
        roe : pd.Series
            Return on Equity
        roa : pd.Series
            Return on Assets
        revenue_growth : pd.Series
            Revenue growth rate
        margin_trend : pd.Series
            Profit margin trend
        weights : dict, optional
            Custom weights for each component

        Returns:
        --------
        pd.Series
            Quality score (0-100 scale)
        """
        if weights is None:
            weights = {
                'roe': 0.3,
                'roa': 0.25,
                'revenue_growth': 0.25,
                'margin_trend': 0.20
            }

        # Normalize each component to 0-100 scale
        def normalize(series, lower_bound=0, upper_bound=None):
            if upper_bound is None:
                # Use percentile-based normalization
                return series.rank(pct=True) * 100
            else:
                # Clip and normalize to bounds
                clipped = series.clip(lower_bound, upper_bound)
                return ((clipped - lower_bound) / (upper_bound - lower_bound)) * 100

        # Normalize components
        roe_norm = normalize(roe, 0, 0.30)  # 0-30% ROE
        roa_norm = normalize(roa, 0, 0.15)  # 0-15% ROA
        growth_norm = normalize(revenue_growth, -0.1, 0.5)  # -10% to 50% growth
        margin_norm = normalize(margin_trend)

        # Calculate weighted score
        score = (
            weights['roe'] * roe_norm +
            weights['roa'] * roa_norm +
            weights['revenue_growth'] * growth_norm +
            weights['margin_trend'] * margin_norm
        )

        return score

    def calculate_all_growth_metrics(
        self,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        periods_yoy: int = 4,
        periods_qoq: int = 1
    ) -> pd.DataFrame:
        """
        Calculate all growth metrics from financial statements.

        Parameters:
        -----------
        income_statement : pd.DataFrame
            Income statement with columns as time periods
        balance_sheet : pd.DataFrame
            Balance sheet with columns as time periods
        periods_yoy : int, default=4
            Periods for YoY calculation (4 for quarterly data)
        periods_qoq : int, default=1
            Periods for QoQ calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with all calculated growth metrics
        """
        results = pd.DataFrame(index=income_statement.columns)

        try:
            # Extract required fields from income statement
            revenue_fields = ['Total Revenue', 'Revenue', 'Sales']
            net_income_fields = ['Net Income', 'Net Profit']
            gross_profit_fields = ['Gross Profit']
            operating_income_fields = ['Operating Income', 'Operating Profit']

            # Get revenue
            revenue = None
            for field in revenue_fields:
                if field in income_statement.index:
                    revenue = income_statement.loc[field]
                    break

            # Get net income
            net_income = None
            for field in net_income_fields:
                if field in income_statement.index:
                    net_income = income_statement.loc[field]
                    break

            # Get gross profit
            gross_profit = None
            for field in gross_profit_fields:
                if field in income_statement.index:
                    gross_profit = income_statement.loc[field]
                    break

            # Get operating income
            operating_income = None
            for field in operating_income_fields:
                if field in income_statement.index:
                    operating_income = income_statement.loc[field]
                    break

            # Extract from balance sheet
            equity_fields = ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Shareholders Equity']
            asset_fields = ['Total Assets']

            equity = None
            for field in equity_fields:
                if field in balance_sheet.index:
                    equity = balance_sheet.loc[field]
                    break

            total_assets = None
            for field in asset_fields:
                if field in balance_sheet.index:
                    total_assets = balance_sheet.loc[field]
                    break

            # Calculate growth rates
            if revenue is not None:
                results['revenue_growth_yoy'] = self.calculate_revenue_growth(revenue, periods_yoy)
                results['revenue_growth_qoq'] = self.calculate_revenue_growth(revenue, periods_qoq)
                results['revenue_cagr_3y'] = self.calculate_revenue_growth(revenue, 12, method='cagr')

            if net_income is not None:
                results['earnings_growth_yoy'] = self.calculate_earnings_growth(net_income, periods_yoy)
                results['earnings_growth_qoq'] = self.calculate_earnings_growth(net_income, periods_qoq)

            if equity is not None:
                results['book_value_growth_yoy'] = self.calculate_book_value_growth(equity, periods_yoy)

            # Calculate profitability metrics
            if revenue is not None and net_income is not None:
                margins = self.calculate_profit_margins(
                    revenue, gross_profit, operating_income, net_income
                )
                for key, value in margins.items():
                    results[key] = value

                # Calculate margin trends
                if 'net_margin' in results.columns:
                    results['net_margin_trend'] = self.calculate_margin_trend(results['net_margin'])

            # Calculate ROE and ROA
            if net_income is not None and equity is not None:
                results['roe'] = self.calculate_roe(net_income, equity)
                results['roe_trend'] = self.calculate_roe_trend(results['roe'])

            if net_income is not None and total_assets is not None:
                results['roa'] = self.calculate_roa(net_income, total_assets)
                results['roa_trend'] = self.calculate_roa_trend(results['roa'])

            # Calculate quality score if enough data
            if all(col in results.columns for col in ['roe', 'roa', 'revenue_growth_yoy', 'net_margin_trend']):
                results['quality_score'] = self.calculate_quality_score(
                    results['roe'],
                    results['roa'],
                    results['revenue_growth_yoy'],
                    results['net_margin_trend']
                )

        except Exception as e:
            logger.error(f"Error calculating growth metrics: {str(e)}")

        return results


# Convenience functions for quick calculations

def calculate_revenue_growth(
    revenue_series: pd.Series,
    periods: int = 1,
    method: str = 'yoy'
) -> pd.Series:
    """
    Quick function to calculate revenue growth.

    Parameters:
    -----------
    revenue_series : pd.Series
        Time series of revenue
    periods : int
        Lookback periods
    method : str
        Calculation method

    Returns:
    --------
    pd.Series
        Revenue growth rates
    """
    return GrowthMetrics.calculate_revenue_growth(revenue_series, periods, method)


def calculate_earnings_growth(
    earnings_series: pd.Series,
    periods: int = 1
) -> pd.Series:
    """
    Quick function to calculate earnings growth.

    Parameters:
    -----------
    earnings_series : pd.Series
        Time series of earnings
    periods : int
        Lookback periods

    Returns:
    --------
    pd.Series
        Earnings growth rates
    """
    return GrowthMetrics.calculate_earnings_growth(earnings_series, periods)


def calculate_roe(
    net_income: pd.Series,
    shareholders_equity: pd.Series
) -> pd.Series:
    """
    Quick function to calculate ROE.

    Parameters:
    -----------
    net_income : pd.Series
        Net income
    shareholders_equity : pd.Series
        Shareholders' equity

    Returns:
    --------
    pd.Series
        ROE values
    """
    return GrowthMetrics.calculate_roe(net_income, shareholders_equity)


def calculate_roa(
    net_income: pd.Series,
    total_assets: pd.Series
) -> pd.Series:
    """
    Quick function to calculate ROA.

    Parameters:
    -----------
    net_income : pd.Series
        Net income
    total_assets : pd.Series
        Total assets

    Returns:
    --------
    pd.Series
        ROA values
    """
    return GrowthMetrics.calculate_roa(net_income, total_assets)


def calculate_all_fundamental_growth(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame
) -> pd.DataFrame:
    """
    Quick function to calculate all growth metrics.

    Parameters:
    -----------
    income_statement : pd.DataFrame
        Income statement data
    balance_sheet : pd.DataFrame
        Balance sheet data

    Returns:
    --------
    pd.DataFrame
        All growth metrics
    """
    calculator = GrowthMetrics()
    return calculator.calculate_all_growth_metrics(income_statement, balance_sheet)


if __name__ == "__main__":
    # Example usage
    print("Fundamental Growth Metrics Module")
    print("=" * 70)
    print("\nAvailable metrics:")
    print("- Revenue Growth (YoY, QoQ, CAGR)")
    print("- Earnings Growth (YoY, QoQ)")
    print("- Book Value Growth")
    print("- EPS Growth")
    print("- Free Cash Flow Growth")
    print("- ROE (Return on Equity) and trends")
    print("- ROA (Return on Assets) and trends")
    print("- ROIC (Return on Invested Capital)")
    print("- Profit Margins (Gross, Operating, Net)")
    print("- Margin Trends")
    print("- DuPont ROE Analysis")
    print("- Sustainable Growth Rate")
    print("- Growth Acceleration Detection")
    print("- Fundamental Quality Score")

    print("\n" + "=" * 70)
    print("Example: Calculate growth metrics from sample data")
    print("=" * 70)

    # Create sample quarterly data
    dates = pd.date_range('2023-Q1', periods=8, freq='Q')

    # Sample income statement data
    sample_income = pd.DataFrame({
        'Total Revenue': [1000, 1050, 1100, 1180, 1250, 1320, 1400, 1480],
        'Gross Profit': [400, 420, 440, 472, 500, 528, 560, 592],
        'Operating Income': [200, 210, 220, 236, 250, 264, 280, 296],
        'Net Income': [150, 157.5, 165, 177, 187.5, 198, 210, 222],
    }, index=dates).T

    # Sample balance sheet data
    sample_balance = pd.DataFrame({
        'Total Assets': [5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400],
        'Stockholders Equity': [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],
    }, index=dates).T

    # Calculate metrics
    calculator = GrowthMetrics()
    metrics_df = calculator.calculate_all_growth_metrics(
        sample_income,
        sample_balance,
        periods_yoy=4,
        periods_qoq=1
    )

    print("\nCalculated Growth Metrics (Latest Quarter):")
    print("-" * 70)

    latest = metrics_df.iloc[-1]

    if 'revenue_growth_yoy' in metrics_df.columns:
        print(f"Revenue Growth (YoY): {latest['revenue_growth_yoy']*100:.2f}%")
    if 'revenue_growth_qoq' in metrics_df.columns:
        print(f"Revenue Growth (QoQ): {latest['revenue_growth_qoq']*100:.2f}%")
    if 'earnings_growth_yoy' in metrics_df.columns:
        print(f"Earnings Growth (YoY): {latest['earnings_growth_yoy']*100:.2f}%")
    if 'roe' in metrics_df.columns:
        print(f"ROE: {latest['roe']*100:.2f}%")
    if 'roa' in metrics_df.columns:
        print(f"ROA: {latest['roa']*100:.2f}%")
    if 'net_margin' in metrics_df.columns:
        print(f"Net Margin: {latest['net_margin']*100:.2f}%")
    if 'quality_score' in metrics_df.columns:
        print(f"Quality Score: {latest['quality_score']:.2f}/100")

    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)
    print("\n# Calculate revenue growth")
    print(">>> from features.fundamental.growth import calculate_revenue_growth")
    print(">>> growth = calculate_revenue_growth(revenue_series, periods=4, method='yoy')")
    print("\n# Calculate ROE")
    print(">>> from features.fundamental.growth import calculate_roe")
    print(">>> roe = calculate_roe(net_income, shareholders_equity)")
    print("\n# Calculate all metrics at once")
    print(">>> from features.fundamental.growth import GrowthMetrics")
    print(">>> calculator = GrowthMetrics()")
    print(">>> all_metrics = calculator.calculate_all_growth_metrics(income_stmt, balance_sheet)")
