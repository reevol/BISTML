"""
Example Usage of BIST Fundamental Data Collector

This script demonstrates how to use the FundamentalCollector to:
1. Collect financial statements for BIST stocks
2. Calculate fundamental metrics (P/E, P/B, EV/EBITDA, etc.)
3. Analyze multiple stocks
4. Export data for further analysis
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.collectors.fundamental_collector import (
    FundamentalCollector,
    Period,
    StatementType
)
import pandas as pd


def example_1_single_stock_metrics():
    """Example 1: Get comprehensive metrics for a single stock"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Comprehensive Fundamental Metrics for Turkish Airlines (THYAO)")
    print("="*80 + "\n")

    collector = FundamentalCollector()

    # Get all fundamental metrics for Turkish Airlines
    metrics = collector.get_comprehensive_metrics('THYAO', Period.QUARTERLY)

    # Display results
    print(f"Stock: {metrics.ticker}")
    print(f"Analysis Date: {metrics.date.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("VALUATION RATIOS:")
    print("-" * 40)
    if metrics.pe_ratio:
        print(f"  P/E Ratio:        {metrics.pe_ratio:>10.2f}")
    if metrics.pb_ratio:
        print(f"  P/B Ratio:        {metrics.pb_ratio:>10.2f}")
    if metrics.ev_ebitda:
        print(f"  EV/EBITDA:        {metrics.ev_ebitda:>10.2f}")

    print("\nPROFITABILITY METRICS:")
    print("-" * 40)
    if metrics.gross_margin:
        print(f"  Gross Margin:     {metrics.gross_margin*100:>9.2f}%")
    if metrics.operating_margin:
        print(f"  Operating Margin: {metrics.operating_margin*100:>9.2f}%")
    if metrics.net_margin:
        print(f"  Net Margin:       {metrics.net_margin*100:>9.2f}%")
    if metrics.roe:
        print(f"  ROE:              {metrics.roe*100:>9.2f}%")
    if metrics.roa:
        print(f"  ROA:              {metrics.roa*100:>9.2f}%")

    print("\nGROWTH METRICS:")
    print("-" * 40)
    if metrics.revenue_growth_yoy:
        print(f"  Revenue Growth (YoY): {metrics.revenue_growth_yoy*100:>7.2f}%")
    if metrics.revenue_growth_qoq:
        print(f"  Revenue Growth (QoQ): {metrics.revenue_growth_qoq*100:>7.2f}%")

    print("\nLEVERAGE & LIQUIDITY:")
    print("-" * 40)
    if metrics.debt_to_equity:
        print(f"  Debt-to-Equity:   {metrics.debt_to_equity:>10.2f}")
    if metrics.current_ratio:
        print(f"  Current Ratio:    {metrics.current_ratio:>10.2f}")
    if metrics.quick_ratio:
        print(f"  Quick Ratio:      {metrics.quick_ratio:>10.2f}")


def example_2_financial_statements():
    """Example 2: Get specific financial statements"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Financial Statements for Garanti Bank (GARAN)")
    print("="*80 + "\n")

    collector = FundamentalCollector()

    # Get quarterly income statement
    print("Quarterly Income Statement (Last 4 Quarters):")
    print("-" * 80)
    income_stmt = collector.get_financial_statements(
        'GARAN',
        Period.QUARTERLY,
        StatementType.INCOME_STATEMENT
    )

    if not income_stmt.empty:
        # Display key metrics
        key_metrics = [
            'Total Revenue',
            'Gross Profit',
            'Operating Income',
            'Net Income',
        ]

        display_df = pd.DataFrame()
        for metric in key_metrics:
            if metric in income_stmt.index:
                display_df[metric] = income_stmt.loc[metric]

        print(display_df.T.to_string())
    else:
        print("No income statement data available")


def example_3_multiple_stocks():
    """Example 3: Analyze multiple BIST stocks"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparative Analysis of BIST Banking Stocks")
    print("="*80 + "\n")

    collector = FundamentalCollector()

    # BIST banking stocks
    banking_stocks = ['GARAN', 'ISCTR', 'AKBNK', 'YKBNK', 'HALKB']

    print(f"Analyzing {len(banking_stocks)} banking stocks: {', '.join(banking_stocks)}\n")
    print("Please wait, collecting data...\n")

    # Get metrics for all stocks
    metrics_df = collector.get_metrics_for_multiple_stocks(
        banking_stocks,
        Period.QUARTERLY
    )

    # Display comparison table
    if not metrics_df.empty:
        display_columns = [
            'ticker',
            'pe_ratio',
            'pb_ratio',
            'roe',
            'debt_to_equity',
            'net_margin'
        ]

        available_columns = [col for col in display_columns if col in metrics_df.columns]
        comparison_df = metrics_df[available_columns].copy()

        # Format percentages
        if 'roe' in comparison_df.columns:
            comparison_df['roe'] = comparison_df['roe'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
        if 'net_margin' in comparison_df.columns:
            comparison_df['net_margin'] = comparison_df['net_margin'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )

        print("Banking Stocks Comparison:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))

        # Find best values
        print("\n" + "-" * 80)
        print("HIGHLIGHTS:")
        print("-" * 80)

        if 'pe_ratio' in metrics_df.columns:
            lowest_pe = metrics_df.nsmallest(1, 'pe_ratio')
            if not lowest_pe.empty:
                print(f"Lowest P/E:  {lowest_pe.iloc[0]['ticker']} ({lowest_pe.iloc[0]['pe_ratio']:.2f})")

        if 'pb_ratio' in metrics_df.columns:
            lowest_pb = metrics_df.nsmallest(1, 'pb_ratio')
            if not lowest_pb.empty:
                print(f"Lowest P/B:  {lowest_pb.iloc[0]['ticker']} ({lowest_pb.iloc[0]['pb_ratio']:.2f})")


def example_4_sector_analysis():
    """Example 4: Analyze stocks from different sectors"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Sector BIST Analysis")
    print("="*80 + "\n")

    collector = FundamentalCollector()

    # Stocks from different sectors
    diverse_stocks = {
        'THYAO': 'Airlines',
        'TUPRS': 'Oil Refinery',
        'EREGL': 'Steel',
        'ASELS': 'Defense',
        'BIMAS': 'Retail',
    }

    print("Analyzing stocks from different sectors:\n")
    for ticker, sector in diverse_stocks.items():
        print(f"  {ticker} - {sector}")

    print("\nCollecting fundamental metrics...\n")

    metrics_df = collector.get_metrics_for_multiple_stocks(
        list(diverse_stocks.keys()),
        Period.ANNUAL
    )

    if not metrics_df.empty:
        # Add sector information
        metrics_df['sector'] = metrics_df['ticker'].map(diverse_stocks)

        # Display key metrics
        display_df = metrics_df[[
            'ticker', 'sector', 'pe_ratio', 'pb_ratio',
            'debt_to_equity', 'revenue_growth_yoy'
        ]].copy()

        # Format growth rate
        if 'revenue_growth_yoy' in display_df.columns:
            display_df['revenue_growth_yoy'] = display_df['revenue_growth_yoy'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )

        print("Sector Comparison:")
        print("-" * 100)
        print(display_df.to_string(index=False))

        # Export to CSV
        output_file = collector.export_to_csv(
            metrics_df,
            'bist_sector_analysis',
            include_timestamp=True
        )
        print(f"\nFull analysis exported to: {output_file}")


def example_5_individual_calculations():
    """Example 5: Individual metric calculations"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Individual Metric Calculations for Koç Holding (KCHOL)")
    print("="*80 + "\n")

    collector = FundamentalCollector()
    ticker = 'KCHOL'

    # Calculate individual metrics
    print(f"Calculating individual metrics for {ticker}...\n")

    # Valuation
    pe = collector.calculate_pe_ratio(ticker)
    pb = collector.calculate_pb_ratio(ticker)
    ev_ebitda = collector.calculate_ev_ebitda(ticker)

    print("VALUATION METRICS:")
    print("-" * 40)
    print(f"  P/E Ratio:   {pe:.2f}" if pe else "  P/E Ratio:   N/A")
    print(f"  P/B Ratio:   {pb:.2f}" if pb else "  P/B Ratio:   N/A")
    print(f"  EV/EBITDA:   {ev_ebitda:.2f}" if ev_ebitda else "  EV/EBITDA:   N/A")

    # Leverage
    de_ratio = collector.calculate_debt_to_equity(ticker)
    print(f"\nLEVERAGE:")
    print("-" * 40)
    print(f"  Debt-to-Equity: {de_ratio:.2f}" if de_ratio else "  Debt-to-Equity: N/A")

    # Profitability
    margins = collector.calculate_profit_margins(ticker)
    print(f"\nPROFITABILITY:")
    print("-" * 40)
    if margins['gross_margin']:
        print(f"  Gross Margin:     {margins['gross_margin']*100:.2f}%")
    if margins['operating_margin']:
        print(f"  Operating Margin: {margins['operating_margin']*100:.2f}%")
    if margins['net_margin']:
        print(f"  Net Margin:       {margins['net_margin']*100:.2f}%")

    # Growth
    growth = collector.calculate_revenue_growth(ticker, Period.QUARTERLY)
    print(f"\nGROWTH:")
    print("-" * 40)
    if growth['revenue_growth_yoy']:
        print(f"  Revenue Growth (YoY): {growth['revenue_growth_yoy']*100:.2f}%")
    if growth['revenue_growth_qoq']:
        print(f"  Revenue Growth (QoQ): {growth['revenue_growth_qoq']*100:.2f}%")


def main():
    """Run all examples"""
    print("\n")
    print("="*80)
    print(" BIST FUNDAMENTAL DATA COLLECTOR - EXAMPLES")
    print("="*80)

    try:
        # Run examples
        example_1_single_stock_metrics()
        example_2_financial_statements()
        example_3_multiple_stocks()
        example_4_sector_analysis()
        example_5_individual_calculations()

        print("\n" + "="*80)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Note: This script requires internet connection and valid BIST stock tickers
    # Some stocks may not have complete fundamental data available

    print("""
    ⚠️  IMPORTANT NOTES:

    1. This script requires internet connection to fetch data from Yahoo Finance
    2. Some BIST stocks may have limited or missing fundamental data
    3. The script includes rate limiting to avoid API throttling
    4. Each example may take several seconds to complete
    5. Results depend on data availability from Yahoo Finance

    Press Enter to continue or Ctrl+C to cancel...
    """)

    try:
        input()
        main()
    except KeyboardInterrupt:
        print("\n\nExamples cancelled by user.")
