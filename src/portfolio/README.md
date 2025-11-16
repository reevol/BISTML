# Portfolio Module

Comprehensive portfolio management and optimization system for BIST AI Trading System.

## Overview

The Portfolio module provides complete functionality for:
- **Portfolio Management**: Position tracking, P&L calculation, transaction history
- **Portfolio Optimization**: Position sizing strategies (Kelly, Risk Parity, etc.)
- **Alert System**: Real-time monitoring and notifications

## Module Components

### 1. Portfolio Manager (`manager.py`)
Tracks positions, calculates profit/loss, and manages cost basis.

### 2. Portfolio Optimizer (`optimization.py`)
Advanced position sizing and portfolio allocation strategies.

### 3. Alert System (`alerts.py`)
Real-time monitoring and alerting for portfolio events.

## Features

### Core Functionality
- **Position Tracking**: Real-time holdings and share quantities
- **Cost Basis Calculation**: FIFO, LIFO, and Average Cost methods
- **P&L Calculation**: Both realized and unrealized profit/loss
- **Transaction History**: Complete audit trail of all trades
- **Portfolio Analytics**: Performance metrics, allocation, diversification

### Advanced Features
- **Multiple Cost Basis Methods**: Choose FIFO, LIFO, or Average Cost
- **Commission Tracking**: Track and analyze trading costs
- **Portfolio I/O**: Save/load via JSON, pickle, or CSV
- **Performance Metrics**: ROI, annualized returns, win rates
- **Allocation Analysis**: Position sizing and diversification metrics

## Quick Start

```python
from src.portfolio.manager import PortfolioManager, CostBasisMethod

# Create a portfolio
portfolio = PortfolioManager(
    name="My BIST Portfolio",
    initial_cash=100000.0,
    cost_basis_method=CostBasisMethod.AVERAGE,
    currency="TRY"
)

# Buy some stocks
portfolio.buy(symbol="THYAO", shares=100, price=250.0, commission=10.0)
portfolio.buy(symbol="GARAN", shares=200, price=85.0, commission=15.0)

# Sell stocks
portfolio.sell(symbol="THYAO", shares=50, price=260.0, commission=5.0)

# Get portfolio summary
current_prices = {"THYAO": 265.0, "GARAN": 90.0}
summary = portfolio.get_portfolio_summary(current_prices)

print(f"Total Value: {summary['total_value']:,.2f} TRY")
print(f"Total P&L: {summary['total_pnl']:,.2f} TRY")
print(f"Return: {summary['total_return_pct']:.2f}%")
```

## File Structure

```
src/portfolio/
├── __init__.py           # Module exports
├── manager.py            # Portfolio Manager
├── optimization.py       # Portfolio Optimizer
├── alerts.py             # Alert System
└── README.md            # This file

examples/
├── portfolio_manager_example.py       # Manager examples
├── portfolio_optimization_example.py  # Optimizer examples
└── portfolio_alerts_example.py        # Alert examples

docs/
├── portfolio_optimization.md          # Detailed optimizer docs
└── portfolio_alerts_system.md         # Alert system docs
```

## Classes

### PortfolioManager
Main class for portfolio management.

**Key Methods:**
- `buy(symbol, shares, price, commission, notes)` - Execute buy order
- `sell(symbol, shares, price, commission, notes)` - Execute sell order
- `get_portfolio_summary(current_prices)` - Get comprehensive summary
- `get_allocation(current_prices)` - Get allocation breakdown
- `get_transaction_history()` - Get transaction history
- `calculate_performance_metrics(current_prices)` - Calculate performance
- `save_to_json(filepath)` / `load_from_json(filepath)` - Persistence

### Transaction
Represents a single buy/sell transaction.

**Attributes:**
- `symbol` - Stock symbol
- `transaction_type` - BUY, SELL, DIVIDEND, etc.
- `shares` - Number of shares
- `price` - Price per share
- `commission` - Trading fees
- `timestamp` - Transaction time

### Position
Represents a position in a single security.

**Attributes:**
- `symbol` - Stock symbol
- `shares` - Current shares held
- `cost_basis` - Average cost per share
- `total_cost` - Total cost of position
- `transactions` - List of all transactions

### PortfolioOptimizer
Advanced position sizing and portfolio allocation.

**Position Sizing Methods:**
- Kelly Criterion (Full and Fractional)
- Risk Parity
- Equal Weight
- Volatility-Weighted
- Maximum Sharpe Ratio
- Minimum Variance

**Key Methods:**
- `kelly_portfolio(assets, fractional)` - Kelly-optimal weights
- `risk_parity(assets, correlation_matrix)` - Risk parity allocation
- `equal_weight(assets)` - Equal weight allocation
- `mean_variance_optimization(assets, correlation_matrix)` - Markowitz optimization
- `apply_risk_constraints(portfolio, assets)` - Apply risk management
- `calculate_position_sizes(weights, portfolio_value, prices)` - Convert to shares
- `calculate_rebalancing_trades(target, current, prices)` - Rebalancing logic

**Quick Example:**
```python
from src.portfolio import PortfolioOptimizer, AssetMetrics, create_optimizer

# Create optimizer
optimizer = create_optimizer(risk_level="MODERATE", max_position_size=0.20)

# Define assets
assets = [
    AssetMetrics(
        symbol="THYAO",
        expected_return=0.15,
        volatility=0.25,
        win_rate=0.60,
        current_price=250.0
    )
]

# Optimize portfolio
weights = optimizer.kelly_portfolio(assets, fractional=True)

# Get position sizes
positions = optimizer.calculate_position_sizes(
    weights,
    portfolio_value=100000.0,
    current_prices={"THYAO": 250.0}
)
```

See `docs/portfolio_optimization.md` for detailed documentation.

## Cost Basis Methods

### AVERAGE (Default)
Calculates weighted average cost across all purchases.

```python
# Buy 100 @ 250, then 100 @ 260
# Average cost = (25000 + 26000) / 200 = 255
```

### FIFO (First In, First Out)
Sells oldest shares first.

```python
# Buy 100 @ 250, then 100 @ 260, sell 150
# Sells: 100 @ 250, then 50 @ 260
```

### LIFO (Last In, First Out)
Sells newest shares first.

```python
# Buy 100 @ 250, then 100 @ 260, sell 150
# Sells: 100 @ 260, then 50 @ 250
```

## Portfolio I/O

### JSON Format
```python
# Save
portfolio.save_to_json("portfolio.json")

# Load
portfolio = PortfolioManager.load_from_json("portfolio.json")
```

### CSV Export
```python
# Export positions
portfolio.export_to_csv("positions.csv", current_prices)

# Export transactions
portfolio.export_transactions_to_csv("transactions.csv")
```

### Pickle Format
```python
# Save (binary, faster but not human-readable)
portfolio.save_to_pickle("portfolio.pkl")

# Load
portfolio = PortfolioManager.load_from_pickle("portfolio.pkl")
```

## Performance Metrics

```python
metrics = portfolio.calculate_performance_metrics(
    current_prices={"THYAO": 265.0, "GARAN": 90.0},
    benchmark_return=5.0  # BIST-100 return
)

# Available metrics:
# - total_return_pct: Total return percentage
# - annualized_return_pct: Annualized return
# - win_rate: Percentage of profitable positions
# - excess_return: Alpha vs benchmark
# - total_commissions: Total trading costs
```

## Integration with BIST System

### With Data Collectors
```python
from src.data.collectors.bist_collector import BISTCollector

collector = BISTCollector()
portfolio = PortfolioManager(initial_cash=100000.0)

# Buy based on signals
portfolio.buy(symbol="THYAO", shares=100, price=250.0)

# Update with current prices
current_prices = {}
for symbol in portfolio.positions.keys():
    price_data = collector.get_latest_price(symbol)
    current_prices[symbol] = price_data['current_price']

# Get real-time portfolio value
summary = portfolio.get_portfolio_summary(current_prices)
```

### With Trading Signals
```python
from src.signals import generate_signals

# Generate trading signals
signals = generate_signals(data)

# Execute trades based on signals
for signal in signals:
    if signal['action'] == 'BUY':
        portfolio.buy(
            symbol=signal['symbol'],
            shares=signal['shares'],
            price=signal['price']
        )
```

## Error Handling

```python
from src.portfolio.manager import (
    InsufficientSharesError,
    InvalidTransactionError,
    PositionNotFoundError
)

try:
    portfolio.sell(symbol="THYAO", shares=1000, price=260.0)
except InsufficientSharesError as e:
    print(f"Cannot sell: {e}")

try:
    portfolio.buy(symbol="THYAO", shares=10000, price=1000.0)
except InvalidTransactionError as e:
    print(f"Invalid transaction: {e}")
```

## Examples

See `/home/user/BISTML/examples/portfolio_manager_example.py` for comprehensive examples covering:

1. Basic portfolio operations
2. Selling and P&L calculation
3. Allocation analysis
4. Transaction history
5. Performance metrics
6. Save/load operations
7. Cost basis method comparison

Run examples:
```bash
python examples/portfolio_manager_example.py
```

## Testing

Run unit tests:
```bash
python -m pytest tests/test_portfolio/test_manager.py -v
```

## Best Practices

1. **Choose appropriate cost basis method**
   - AVERAGE: Simplest, good for general use
   - FIFO: Tax optimization, oldest shares first
   - LIFO: Recent cost basis, newest shares first

2. **Track commissions accurately**
   - Include all trading fees
   - Commission impacts cost basis and P&L

3. **Regular portfolio backups**
   - Save portfolio state after significant trades
   - Use JSON for human-readable backups
   - Use pickle for faster save/load

4. **Performance monitoring**
   - Calculate metrics regularly
   - Compare against benchmarks (BIST-100, XU030)
   - Track win rate and profitable positions

5. **Diversification**
   - Monitor position allocation
   - Avoid over-concentration (>30% in single position)
   - Balance across sectors

## Technical Details

- **Language**: Python 3.8+
- **Dependencies**: pandas, numpy
- **Lines of Code**: ~1,135
- **Test Coverage**: Comprehensive unit tests
- **Data Structures**: Dataclasses for type safety

## Author

BIST AI Trading System
Date: 2025-11-16

## License

Part of the BISTML project.
