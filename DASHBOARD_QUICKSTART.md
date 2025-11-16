# BIST AI Trading Dashboard - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

Make sure you have all required packages:

```bash
cd /home/user/BISTML
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (Optional)

If you want to see the dashboard with sample data:

```bash
python examples/dashboard_example.py
```

This will generate:
- Sample trading signals
- Sample portfolio with positions
- Sample backtest results

### Step 3: Launch the Dashboard

```bash
python run_dashboard.py
```

Or use Streamlit directly:

```bash
streamlit run src/ui/dashboard.py
```

**Access the dashboard at:** http://localhost:8501

---

## 📊 Dashboard Overview

### Four Main Tabs:

#### 1️⃣ Live Signals
Monitor real-time trading signals with:
- Signal table (BUY/SELL/HOLD)
- Confidence scores
- Price targets
- Position sizing
- Risk metrics

#### 2️⃣ Portfolio
Track your positions:
- Total portfolio value
- Individual positions
- P&L breakdown
- Allocation charts
- Performance metrics

#### 3️⃣ Backtesting
Analyze historical performance:
- Equity curves
- Drawdown analysis
- Win rate & profit factor
- Trade timeline
- Risk-adjusted returns

#### 4️⃣ Performance
Comprehensive analytics:
- Monthly returns heatmap
- Rolling Sharpe ratio
- KPI dashboard
- Cumulative returns

---

## ⚙️ Key Features

### Sidebar Controls

- **Auto-refresh**: Enable/disable automatic updates
- **Filters**: Filter by stock, signal type, confidence
- **Date Range**: Select backtest period
- **Export**: Download data in CSV/JSON format

### Real-time Updates

The dashboard automatically refreshes data every 60 seconds (configurable).

### Interactive Charts

All charts are interactive:
- Hover for details
- Zoom in/out
- Pan across time
- Download as PNG

---

## 💡 Tips

1. **Start with Sample Data**
   ```bash
   python examples/dashboard_example.py
   ```

2. **Adjust Refresh Rate**
   - Use sidebar slider (10-300 seconds)
   - Disable for static analysis

3. **Filter Signals**
   - Set minimum confidence threshold
   - Focus on specific stocks
   - Filter by signal type

4. **Export Data**
   - CSV: Signals and positions
   - JSON: Portfolio state
   - PDF: Backtest reports (coming soon)

---

## 🔧 Configuration

### Change Port

```bash
streamlit run src/ui/dashboard.py --server.port=8502
```

### Run on Network

```bash
streamlit run src/ui/dashboard.py --server.address=0.0.0.0
```

### Debug Mode

```bash
streamlit run src/ui/dashboard.py --logger.level=debug
```

---

## 📁 File Structure

```
BISTML/
├── src/
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py          # Main dashboard file (2000+ lines)
├── examples/
│   └── dashboard_example.py      # Sample data generator
├── run_dashboard.py              # Dashboard launcher
├── DASHBOARD_README.md           # Full documentation
└── DASHBOARD_QUICKSTART.md       # This file
```

---

## 🔌 Integration with Your Data

### Connect to Database

Edit `dashboard.py` line ~150:

```python
@st.cache_data(ttl=300)
def load_signals_data(filters=None):
    # Replace with your database query
    from src.data.storage.database import get_signals
    signals = get_signals(filters)
    return pd.DataFrame(signals)
```

### Load Portfolio

Edit `dashboard.py` line ~220:

```python
@st.cache_data(ttl=300)
def load_portfolio_data():
    # Load from your portfolio manager
    portfolio = PortfolioManager.load_from_json('portfolio.json')
    current_prices = get_current_prices()
    return portfolio.get_portfolio_summary(current_prices)
```

### Load Backtest Results

Edit `dashboard.py` line ~310:

```python
@st.cache_data(ttl=600)
def load_backtest_results():
    # Load from saved backtest
    from src.backtesting.engine import BacktestResults
    results = load_backtest_from_file('results.pkl')
    return results
```

---

## 📊 Dashboard Metrics Explained

### Live Signals Tab
- **Confidence Score**: Model confidence (0-100%)
- **Expected Return**: Predicted price movement (%)
- **Position Size**: Recommended portfolio allocation (%)
- **Risk Score**: Position risk level (0-1)

### Portfolio Tab
- **Total Value**: Cash + positions market value
- **Realized P&L**: Closed position profits/losses
- **Unrealized P&L**: Open position profits/losses
- **Total Return %**: Overall portfolio performance

### Backtesting Tab
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline (%)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio

### Performance Tab
- **Calmar Ratio**: Return / Max drawdown
- **Volatility**: Annualized standard deviation
- **Alpha**: Excess return vs benchmark
- **Beta**: Correlation with market

---

## 🐛 Troubleshooting

### Dashboard Won't Start

**Error**: Port 8501 already in use
```bash
# Use different port
streamlit run src/ui/dashboard.py --server.port=8502
```

**Error**: Module not found
```bash
# Ensure you're in project root
cd /home/user/BISTML
python run_dashboard.py
```

### No Data Showing

1. Generate sample data:
   ```bash
   python examples/dashboard_example.py
   ```

2. Check data files exist:
   ```bash
   ls data/dashboard/
   ```

3. Clear cache:
   - Press 'C' in the dashboard
   - Or restart dashboard

### Slow Performance

1. Reduce refresh interval (sidebar)
2. Apply filters to limit data
3. Increase cache TTL in code
4. Close other browser tabs

---

## 📚 Next Steps

1. **Explore the Dashboard**
   - Navigate through all four tabs
   - Try different filters
   - Hover over charts for details

2. **Integrate Real Data**
   - Connect to your database
   - Load actual portfolio
   - Run real backtests

3. **Customize**
   - Add custom metrics
   - Modify chart styles
   - Create new visualizations

4. **Deploy**
   - Share on local network
   - Deploy to Streamlit Cloud
   - Containerize with Docker

---

## 📖 Full Documentation

For complete documentation, see:
- **DASHBOARD_README.md**: Comprehensive guide
- **src/ui/dashboard.py**: Inline code documentation
- **examples/dashboard_example.py**: Integration examples

---

## 💬 Support

Need help?
- Check documentation: `DASHBOARD_README.md`
- Review examples: `examples/dashboard_example.py`
- Open an issue on GitHub

---

**Happy Trading! 📈**
