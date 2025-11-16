# BIST AI Trading System - Dashboard Guide

## Overview

The Streamlit dashboard provides a comprehensive, real-time interface for monitoring trading signals, portfolio performance, backtesting results, and performance analytics for the BIST AI Trading System.

## Features

### 1. Live Signals Tab 📊
- **Real-time signal monitoring** with automatic refresh
- **Interactive filtering** by stock, signal type, and confidence threshold
- **Signal distribution** visualizations (pie charts, bar charts)
- **Sortable signal table** with all key metrics:
  - Stock symbol
  - Signal type (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
  - Confidence score
  - Current price & target price
  - Expected return
  - Position size recommendation
  - Risk score
  - Stop loss & take profit levels

### 2. Portfolio Tab 💼
- **Portfolio overview** with key metrics:
  - Total portfolio value
  - Cash position
  - Positions value
  - Total return (%)
  - Realized & unrealized P&L
- **Portfolio allocation** pie chart
- **Equity curve** for last 30 days
- **Position details table** with:
  - Current holdings
  - Cost basis vs current price
  - Market value
  - Unrealized P&L per position
- **P&L breakdown** visualizations
- **Position performance** comparison

### 3. Backtesting Tab 🔬
- **Comprehensive backtest results**:
  - Initial vs final capital
  - Total return & annualized return
  - Sharpe, Sortino, and Calmar ratios
  - Maximum drawdown
- **Interactive equity curve** with drawdown overlay
- **Trade statistics**:
  - Total trades
  - Win rate
  - Average win/loss
  - Profit factor
  - Expectancy
- **Risk metrics** analysis
- **Returns distribution** histogram
- **Trades timeline** scatter plot
- **Recent trades table** with full details

### 4. Performance Tab 📈
- **Key performance indicators** (KPIs)
- **Monthly returns heatmap**
- **Rolling metrics**:
  - Rolling Sharpe ratio (30-day)
  - Cumulative returns
- **Comprehensive performance summary**
- **Risk-adjusted returns analysis**

## Installation & Setup

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key packages:
- `streamlit>=1.24.0`
- `plotly>=5.14.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

### Running the Dashboard

#### Method 1: Using the Launcher Script (Recommended)

```bash
python run_dashboard.py
```

#### Method 2: Direct Streamlit Command

```bash
streamlit run src/ui/dashboard.py
```

#### Method 3: With Custom Port

```bash
streamlit run src/ui/dashboard.py --server.port=8502
```

### Accessing the Dashboard

Once running, open your browser to:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

## Configuration

### Sidebar Settings

1. **Auto-refresh**
   - Enable/disable automatic data refresh
   - Configurable refresh interval (10-300 seconds)

2. **Filters**
   - Stock selection (all or specific stocks)
   - Signal type filter
   - Minimum confidence threshold

3. **Date Range**
   - From/to dates for backtesting analysis

4. **Export Options**
   - Export signals to CSV
   - Export portfolio to JSON
   - Export backtest report

## Data Integration

### Loading Real Data

The dashboard is designed to work with both sample data and real production data. To integrate with your data sources:

#### 1. Database Integration

Modify the data loading functions in `dashboard.py`:

```python
@st.cache_data(ttl=300)
def load_signals_data(filters=None):
    # Connect to your database
    from src.data.storage.database import get_signals

    # Query signals
    signals = get_signals(filters)

    return pd.DataFrame(signals)
```

#### 2. Portfolio Manager Integration

```python
@st.cache_data(ttl=300)
def load_portfolio_data():
    # Load from portfolio manager
    portfolio = PortfolioManager.load_from_json('portfolio.json')

    # Get current prices from market data
    current_prices = get_current_prices()

    return portfolio.get_portfolio_summary(current_prices)
```

#### 3. Backtest Results Integration

```python
@st.cache_data(ttl=600)
def load_backtest_results():
    # Load from saved backtest results
    from src.backtesting.engine import BacktestResults

    results = load_backtest_from_file('backtest_results.pkl')

    return results
```

### Running the Example

Generate sample data for the dashboard:

```bash
python examples/dashboard_example.py
```

This will:
1. Generate sample trading signals
2. Create a sample portfolio with positions
3. Run a sample backtest
4. Save all data to `data/dashboard/`

## Customization

### Theme Customization

The dashboard uses a custom theme defined in the launcher script. To modify:

Edit `run_dashboard.py` and change:

```python
"--theme.primaryColor=#1f77b4",      # Primary color
"--theme.backgroundColor=#FFFFFF",    # Background
"--theme.secondaryBackgroundColor=#f0f2f6",  # Secondary bg
"--theme.textColor=#262730",         # Text color
"--theme.font=sans serif"            # Font family
```

### Adding Custom Metrics

To add custom metrics to the dashboard:

1. Define the metric calculation in the appropriate module
2. Add data loading in dashboard helper functions
3. Create visualization using Plotly
4. Add to relevant tab

Example:

```python
def create_custom_metric_chart(data):
    fig = go.Figure(data=[go.Bar(
        x=data['labels'],
        y=data['values']
    )])

    fig.update_layout(
        title='Custom Metric',
        template='plotly_white'
    )

    return fig

# In the main dashboard
with tab4:  # Performance tab
    st.subheader("Custom Metric")
    custom_data = calculate_custom_metric()
    fig = create_custom_metric_chart(custom_data)
    st.plotly_chart(fig, use_container_width=True)
```

## Performance Optimization

### Caching

The dashboard uses Streamlit's caching to improve performance:

```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_signals_data(filters=None):
    # Expensive data loading...
    return data
```

- `ttl`: Time-to-live in seconds
- Adjust based on data update frequency

### Auto-refresh

Auto-refresh is configurable:
- Default: 60 seconds
- Range: 10-300 seconds
- Disable for static analysis

### Large Datasets

For large datasets:
1. Use pagination in tables
2. Apply filters server-side
3. Aggregate data before visualization
4. Use efficient data structures (pandas)

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run src/ui/dashboard.py --server.port=8502
   ```

2. **Module not found**
   - Ensure you're running from project root
   - Check PYTHONPATH
   - Verify all dependencies installed

3. **Data not loading**
   - Check data file paths
   - Verify database connection
   - Review cache settings

4. **Slow performance**
   - Increase cache TTL
   - Reduce auto-refresh frequency
   - Filter data before visualization

### Debug Mode

Enable debug mode:

```bash
streamlit run src/ui/dashboard.py --logger.level=debug
```

## Best Practices

1. **Real-time Updates**
   - Keep refresh interval reasonable (30-60 seconds)
   - Cache expensive computations
   - Use database queries efficiently

2. **Data Visualization**
   - Limit number of data points in charts
   - Use appropriate chart types
   - Ensure responsive design

3. **User Experience**
   - Provide clear metric explanations
   - Include helpful tooltips
   - Show loading states
   - Handle errors gracefully

4. **Security**
   - Don't commit sensitive credentials
   - Use environment variables
   - Implement authentication if deploying

## Deployment

### Local Network Deployment

```bash
streamlit run src/ui/dashboard.py --server.address=0.0.0.0
```

### Cloud Deployment

#### Streamlit Cloud
1. Push to GitHub
2. Connect at share.streamlit.io
3. Configure secrets

#### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/dashboard.py"]
```

Build and run:
```bash
docker build -t bist-dashboard .
docker run -p 8501:8501 bist-dashboard
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/BISTML/issues
- Documentation: See project README.md

## License

Part of BIST AI Trading System
Copyright © 2025

---

**Version**: 1.0
**Last Updated**: 2025-11-16
**Author**: BIST AI Trading System Team
