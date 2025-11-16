# BIST Fundamental Collector - Quick Start Guide

## Overview

The `fundamental_collector.py` module provides comprehensive fundamental data collection and analysis for BIST (Borsa Istanbul) stocks. It automatically fetches financial statements and calculates key metrics used in fundamental analysis.

## File Location

```
/home/user/BISTML/src/data/collectors/fundamental_collector.py
```

## Key Features

### 1. Financial Statement Collection
- **Balance Sheet**: Assets, liabilities, equity, debt positions
- **Income Statement**: Revenue, profits, expenses, margins
- **Cash Flow Statement**: Operating, investing, financing activities
- **Periods**: Quarterly and Annual data

### 2. Calculated Metrics

#### Valuation Ratios
- **P/E Ratio**: Price-to-Earnings
- **P/B Ratio**: Price-to-Book Value
- **EV/EBITDA**: Enterprise Value to EBITDA

#### Profitability Metrics
- **Gross Margin**: (Gross Profit / Revenue) × 100%
- **Operating Margin**: (Operating Income / Revenue) × 100%
- **Net Margin**: (Net Income / Revenue) × 100%
- **ROE**: Return on Equity
- **ROA**: Return on Assets

#### Growth Metrics
- **Revenue Growth (YoY)**: Year-over-Year revenue growth %
- **Revenue Growth (QoQ)**: Quarter-over-Quarter revenue growth %
- **Earnings Growth**: Year-over-Year earnings growth %

#### Leverage & Liquidity
- **Debt-to-Equity**: Total Debt / Total Equity
- **Current Ratio**: Current Assets / Current Liabilities
- **Quick Ratio**: (Current Assets - Inventory) / Current Liabilities

## Installation

### Required Dependencies

```bash
pip install pandas numpy yfinance requests beautifulsoup4 lxml
```

Or install all project dependencies:

```bash
pip install -r /home/user/BISTML/requirements.txt
```

## Quick Start Examples

### Example 1: Get All Metrics for a Single Stock

```python
from src.data.collectors.fundamental_collector import FundamentalCollector, Period

# Initialize collector
collector = FundamentalCollector()

# Get comprehensive metrics for Turkish Airlines
metrics = collector.get_comprehensive_metrics('THYAO', Period.QUARTERLY)

# Access individual metrics
print(f"P/E Ratio: {metrics.pe_ratio}")
print(f"Debt-to-Equity: {metrics.debt_to_equity}")
print(f"Net Margin: {metrics.net_margin * 100:.2f}%")
print(f"Revenue Growth (YoY): {metrics.revenue_growth_yoy * 100:.2f}%")
```

### Example 2: Analyze Multiple Stocks

```python
from src.data.collectors.fundamental_collector import FundamentalCollector, Period

collector = FundamentalCollector()

# BIST banking stocks
banks = ['GARAN', 'ISCTR', 'AKBNK', 'YKBNK']

# Get metrics for all banks
df = collector.get_metrics_for_multiple_stocks(banks, Period.QUARTERLY)

# Display comparison
print(df[['ticker', 'pe_ratio', 'pb_ratio', 'roe', 'debt_to_equity']])

# Find best P/E ratios
best_pe = df.nsmallest(3, 'pe_ratio')
print(best_pe)
```

### Example 3: Get Specific Financial Statements

```python
from src.data.collectors.fundamental_collector import (
    FundamentalCollector, Period, StatementType
)

collector = FundamentalCollector()

# Get quarterly income statement for Garanti Bank
income_stmt = collector.get_financial_statements(
    'GARAN',
    Period.QUARTERLY,
    StatementType.INCOME_STATEMENT
)

print(income_stmt)

# Get annual balance sheet
balance_sheet = collector.get_financial_statements(
    'GARAN',
    Period.ANNUAL,
    StatementType.BALANCE_SHEET
)

print(balance_sheet)
```

### Example 4: Calculate Individual Metrics

```python
from src.data.collectors.fundamental_collector import FundamentalCollector

collector = FundamentalCollector()

ticker = 'AKBNK'

# Calculate specific metrics
pe_ratio = collector.calculate_pe_ratio(ticker)
pb_ratio = collector.calculate_pb_ratio(ticker)
ev_ebitda = collector.calculate_ev_ebitda(ticker)
de_ratio = collector.calculate_debt_to_equity(ticker)

# Get profit margins
margins = collector.calculate_profit_margins(ticker)
print(f"Gross Margin: {margins['gross_margin'] * 100:.2f}%")
print(f"Net Margin: {margins['net_margin'] * 100:.2f}%")

# Get revenue growth
growth = collector.calculate_revenue_growth(ticker, Period.QUARTERLY)
print(f"YoY Growth: {growth['revenue_growth_yoy'] * 100:.2f}%")
```

### Example 5: Export to CSV

```python
from src.data.collectors.fundamental_collector import FundamentalCollector, Period

collector = FundamentalCollector()

# BIST 30 stocks
bist_30 = ['THYAO', 'GARAN', 'ISCTR', 'AKBNK', 'YKBNK',
           'SAHOL', 'KCHOL', 'EREGL', 'SISE', 'TUPRS']

# Collect metrics
df = collector.get_metrics_for_multiple_stocks(bist_30, Period.QUARTERLY)

# Export to CSV with timestamp
filename = collector.export_to_csv(df, 'bist30_fundamentals')
print(f"Exported to: {filename}")
```

## Common BIST Stock Tickers

### Major Banks
- **GARAN**: Garanti Bankası
- **ISCTR**: İş Bankası (C)
- **AKBNK**: Akbank
- **YKBNK**: Yapı Kredi
- **HALKB**: Halkbank

### Holdings & Industrials
- **SAHOL**: Sabancı Holding
- **KCHOL**: Koç Holding
- **EREGL**: Ereğli Demir Çelik
- **SISE**: Şişe Cam
- **TUPRS**: Tüpraş

### Technology & Defense
- **ASELS**: Aselsan
- **TTKOM**: Türk Telekom
- **TCELL**: Turkcell

### Transportation
- **THYAO**: Türk Hava Yolları
- **PGSUS**: Pegasus
- **TAVHL**: TAV Havalimanları

### Retail & Consumer
- **BIMAS**: BIM
- **MGROS**: Migros

## Running the Example Script

A comprehensive example script is provided:

```bash
python /home/user/BISTML/examples_fundamental_collector.py
```

This script demonstrates:
1. Single stock comprehensive analysis
2. Financial statement extraction
3. Multiple stock comparison
4. Sector analysis
5. Individual metric calculations

## Data Source

The collector uses **Yahoo Finance** as the primary data source:
- BIST stocks are accessed with `.IS` suffix (e.g., `THYAO.IS`)
- Provides quarterly and annual financial statements
- Free and reliable data source
- Real-time price data

## Ticker Format

Tickers can be provided in two formats:
1. **Short format**: `THYAO`, `GARAN`, `AKBNK`
2. **Yahoo Finance format**: `THYAO.IS`, `GARAN.IS`, `AKBNK.IS`

The collector automatically normalizes to Yahoo Finance format.

## Return Values

### FundamentalMetrics Object

The `get_comprehensive_metrics()` method returns a `FundamentalMetrics` object with:

```python
@dataclass
class FundamentalMetrics:
    ticker: str
    date: datetime

    # Valuation
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    ev_ebitda: Optional[float]

    # Profitability
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    roe: Optional[float]
    roa: Optional[float]

    # Growth
    revenue_growth_yoy: Optional[float]
    revenue_growth_qoq: Optional[float]
    earnings_growth_yoy: Optional[float]

    # Leverage
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
```

### DataFrame Output

The `get_metrics_for_multiple_stocks()` method returns a pandas DataFrame with all metrics as columns.

## Error Handling

The collector handles errors gracefully:
- Returns `None` for unavailable metrics
- Logs warnings for missing data
- Continues processing even if some metrics fail
- Includes automatic rate limiting to avoid API throttling

## Performance Considerations

1. **Rate Limiting**: Built-in delays (0.3-1 sec) between API calls
2. **Batch Processing**: Use `get_metrics_for_multiple_stocks()` for efficiency
3. **Data Caching**: Consider implementing caching for frequently accessed data
4. **Internet Connection**: Requires active internet connection

## Limitations

1. **Data Availability**: Depends on Yahoo Finance data coverage
2. **Historical Data**: Limited to Yahoo Finance's historical range
3. **Real-time Updates**: Data freshness depends on Yahoo Finance updates
4. **Missing Data**: Some BIST stocks may have incomplete fundamental data

## Troubleshooting

### No Data Returned

```python
# Check if stock exists and has data
from src.data.collectors.fundamental_collector import FundamentalCollector
import yfinance as yf

ticker = 'THYAO.IS'
stock = yf.Ticker(ticker)
print(stock.info)  # Check if stock info is available
```

### Import Errors

```bash
# Ensure you're in the project directory
cd /home/user/BISTML

# Run with proper Python path
python -c "import sys; sys.path.insert(0, 'src'); from data.collectors.fundamental_collector import FundamentalCollector; print('Import successful')"
```

### Rate Limiting

If you encounter rate limiting errors:
- Increase sleep intervals in the code
- Process stocks in smaller batches
- Use caching to avoid repeated API calls

## Integration with Trading System

The fundamental collector integrates with other system components:

```python
# Example: Filter stocks by fundamental criteria
from src.data.collectors.fundamental_collector import FundamentalCollector, Period

collector = FundamentalCollector()

bist_stocks = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'TUPRS']
metrics_df = collector.get_metrics_for_multiple_stocks(bist_stocks, Period.QUARTERLY)

# Filter undervalued stocks (low P/E, low P/B, high ROE)
undervalued = metrics_df[
    (metrics_df['pe_ratio'] < 10) &
    (metrics_df['pb_ratio'] < 2) &
    (metrics_df['roe'] > 0.15)  # ROE > 15%
]

print("Potentially undervalued stocks:")
print(undervalued[['ticker', 'pe_ratio', 'pb_ratio', 'roe']])
```

## Next Steps

1. **Test the collector**: Run the example script to verify functionality
2. **Customize metrics**: Add custom fundamental metrics as needed
3. **Integrate with ML models**: Use metrics as features for ML models
4. **Build screening tools**: Create stock screening based on fundamental criteria
5. **Implement caching**: Add caching layer for better performance

## Support Files

- **Main Module**: `/home/user/BISTML/src/data/collectors/fundamental_collector.py`
- **Examples**: `/home/user/BISTML/examples_fundamental_collector.py`
- **Documentation**: `/home/user/BISTML/src/data/collectors/README.md`
- **Requirements**: `/home/user/BISTML/requirements.txt`

## Additional Resources

- BIST Official Website: https://www.borsaistanbul.com
- Yahoo Finance: https://finance.yahoo.com
- KAP (Public Disclosure): https://www.kap.org.tr

---

**Created**: 2025-11-16
**Module**: BIST AI Trading System - Data Collection Layer
**Status**: Production Ready
