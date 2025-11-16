# Portfolio Optimization Module

## Overview

The Portfolio Optimization module (`src/portfolio/optimization.py`) provides advanced position sizing and portfolio allocation strategies for the BIST AI Trading System. It implements multiple optimization methods with comprehensive risk management rules.

## Features

### Position Sizing Methods

1. **Kelly Criterion**
   - Optimal position sizing for maximum logarithmic wealth growth
   - Full Kelly and Fractional Kelly variants
   - Accounts for win rate, expected return, and variance

2. **Risk Parity**
   - Equal risk contribution from each asset
   - Volatility-weighted allocation
   - Correlation-aware diversification

3. **Equal Weight**
   - Simple equal allocation across assets
   - Optional filtering by minimum return threshold
   - Baseline strategy for comparison

4. **Volatility-Weighted**
   - Position sizing based on risk-adjusted returns
   - Target volatility portfolio construction
   - Sharpe ratio optimization

5. **Mean-Variance Optimization (Markowitz)**
   - Maximum Sharpe Ratio portfolio
   - Minimum Variance portfolio
   - Custom risk-return trade-offs

### Risk Management Rules

- **Position Size Limits**: Maximum allocation per asset
- **Portfolio Concentration**: Limit on top positions
- **Volatility Constraints**: Maximum portfolio volatility
- **Correlation Limits**: Reduce highly correlated positions
- **Drawdown Protection**: Account for historical drawdowns
- **Dynamic Risk Levels**: Conservative, Moderate, Aggressive, Very Aggressive

## Quick Start

### Basic Usage

```python
from src.portfolio import (
    PortfolioOptimizer,
    AssetMetrics,
    RiskLevel,
    create_optimizer,
    optimize_portfolio
)

# Create asset metrics from your predictions
assets = [
    AssetMetrics(
        symbol="THYAO",
        expected_return=0.15,    # 15% expected return
        volatility=0.25,          # 25% volatility
        win_rate=0.60,            # 60% win rate
        current_price=250.0
    ),
    # ... more assets
]

# Create optimizer
optimizer = create_optimizer(
    risk_level="MODERATE",
    max_position_size=0.20
)

# Optimize portfolio
portfolio = optimizer.kelly_portfolio(assets, fractional=True)

# Get weights
for symbol, weight in portfolio.weights.items():
    print(f"{symbol}: {weight:.2%}")
```

### Advanced Usage with Risk Constraints

```python
from src.portfolio import RiskConstraints

# Define custom risk constraints
constraints = RiskConstraints(
    max_position_size=0.15,           # Max 15% per position
    max_portfolio_volatility=0.20,    # Max 20% portfolio vol
    max_drawdown=0.25,                 # Max 25% drawdown
    max_correlation=0.70,              # Max 70% correlation
    min_positions=5,                   # At least 5 positions
    max_positions=15                   # At most 15 positions
)

optimizer = PortfolioOptimizer(
    risk_level=RiskLevel.MODERATE,
    constraints=constraints
)

# Apply constraints to portfolio
constrained_portfolio = optimizer.apply_risk_constraints(
    portfolio,
    assets,
    current_portfolio_value=100000.0,
    correlation_matrix=correlation_matrix
)
```

## Position Sizing Methods in Detail

### 1. Kelly Criterion

The Kelly Criterion determines the optimal fraction of capital to allocate to maximize long-term wealth growth.

**Formula (Continuous):**
```
f = μ / σ²
```
where:
- f = fraction of capital to allocate
- μ = expected return
- σ² = variance of returns

**Fractional Kelly:**
To reduce risk, use a fraction of the full Kelly amount:
- Conservative: 25% of Kelly
- Moderate: 50% of Kelly
- Aggressive: 75% of Kelly
- Very Aggressive: 100% of Kelly

**Example:**
```python
# Full Kelly
portfolio = optimizer.kelly_portfolio(assets, fractional=False)

# Fractional Kelly (50% for moderate risk)
portfolio = optimizer.kelly_portfolio(assets, fractional=True)
```

**Pros:**
- Mathematically optimal for long-term growth
- Automatically sizes positions based on edge and risk
- Prevents over-leveraging

**Cons:**
- Can be aggressive with high volatility
- Sensitive to estimation errors
- May suggest large positions for high conviction bets

### 2. Risk Parity

Risk Parity allocates capital so each asset contributes equally to portfolio risk.

**Formula:**
```
w_i = (1/σ_i) / Σ(1/σ_j)
```
where:
- w_i = weight of asset i
- σ_i = volatility of asset i

**Example:**
```python
portfolio = optimizer.risk_parity(assets, correlation_matrix)
```

**Pros:**
- Balanced risk exposure
- Natural diversification
- Works well with negatively correlated assets

**Cons:**
- Ignores expected returns
- May overweight low-volatility, low-return assets
- Requires accurate volatility estimates

### 3. Equal Weight

Simple 1/N allocation across all assets.

**Example:**
```python
# Basic equal weight
portfolio = optimizer.equal_weight(assets)

# With minimum return filter
portfolio = optimizer.equal_weight(assets, min_threshold=0.05)
```

**Pros:**
- Simple and robust
- No estimation error
- Often outperforms complex strategies

**Cons:**
- Ignores differences in risk and return
- No risk management
- Can be suboptimal with large return differences

### 4. Maximum Sharpe Ratio

Finds the portfolio with the highest risk-adjusted return.

**Example:**
```python
portfolio = optimizer.mean_variance_optimization(
    assets,
    correlation_matrix,
    objective='max_sharpe'
)
```

**Pros:**
- Optimal risk-adjusted returns
- Accounts for correlations
- Theory-backed (Markowitz)

**Cons:**
- Sensitive to input estimates
- Can result in concentrated portfolios
- Requires correlation matrix

### 5. Minimum Variance

Finds the portfolio with the lowest risk for a given return target.

**Example:**
```python
portfolio = optimizer.mean_variance_optimization(
    assets,
    correlation_matrix,
    target_return=0.10,
    objective='min_variance'
)
```

## Risk Levels

The module supports four risk levels that automatically adjust constraints:

### Conservative
- Kelly Fraction: 25%
- Max Position Size: 10%
- Max Top-3 Concentration: 30%

### Moderate
- Kelly Fraction: 50%
- Max Position Size: 15%
- Max Top-3 Concentration: 40%

### Aggressive
- Kelly Fraction: 75%
- Max Position Size: 20%
- Max Top-3 Concentration: 50%

### Very Aggressive
- Kelly Fraction: 100%
- Max Position Size: 25%
- Max Top-3 Concentration: 60%

## Integration with Portfolio Manager

The optimization module integrates seamlessly with the Portfolio Manager:

```python
from src.portfolio import PortfolioManager, create_optimizer

# 1. Create portfolio manager
manager = PortfolioManager(
    name="My Portfolio",
    initial_cash=100000.0
)

# 2. Optimize weights
optimizer = create_optimizer(risk_level="MODERATE")
portfolio_weights = optimizer.kelly_portfolio(assets, fractional=True)

# 3. Calculate position sizes
current_prices = {"THYAO": 250.0, "GARAN": 85.0}
positions = optimizer.calculate_position_sizes(
    portfolio_weights,
    portfolio_value=manager.cash,
    current_prices=current_prices
)

# 4. Execute trades
for symbol, shares in positions.items():
    manager.buy(
        symbol=symbol,
        shares=shares,
        price=current_prices[symbol],
        commission=shares * current_prices[symbol] * 0.002
    )

# 5. View portfolio
summary = manager.get_portfolio_summary(current_prices)
print(f"Total Value: {summary['total_value']:,.2f} TRY")
```

## Rebalancing

The module supports portfolio rebalancing with threshold-based trading:

```python
# Calculate trades needed to reach target weights
trades = optimizer.calculate_rebalancing_trades(
    target_weights=optimal_portfolio,
    current_positions={"THYAO": 100, "GARAN": 200},
    current_prices={"THYAO": 250.0, "GARAN": 85.0},
    portfolio_value=50000.0,
    threshold=0.05  # Only trade if 5% deviation
)

# Execute rebalancing trades
for symbol, shares_to_trade in trades.items():
    if shares_to_trade > 0:
        manager.buy(symbol, shares_to_trade, current_prices[symbol])
    else:
        manager.sell(symbol, abs(shares_to_trade), current_prices[symbol])
```

## Best Practices

### 1. Data Quality
- Use accurate expected return estimates from backtesting
- Calculate volatility from sufficient historical data (>1 year)
- Update correlation matrices regularly
- Account for transaction costs

### 2. Risk Management
- Always use Fractional Kelly, not Full Kelly
- Set appropriate position size limits
- Diversify across low-correlation assets
- Monitor and respect maximum drawdown limits

### 3. Strategy Selection
- **Kelly Criterion**: When you have high-confidence predictions
- **Risk Parity**: For long-term, diversified portfolios
- **Equal Weight**: As a baseline or when predictions are uncertain
- **Max Sharpe**: When you have reliable correlation estimates

### 4. Rebalancing
- Rebalance periodically (monthly or quarterly)
- Use threshold-based rebalancing to reduce trading costs
- Consider tax implications for selling positions
- Account for transaction costs in optimization

### 5. Backtesting
- Test strategies on historical data
- Use walk-forward analysis
- Account for transaction costs and slippage
- Validate assumptions about returns and volatility

## Common Pitfalls

### 1. Estimation Error
- **Problem**: Garbage in, garbage out
- **Solution**: Use robust estimation methods, regularization, and conservative assumptions

### 2. Over-Optimization
- **Problem**: Strategies that work in-sample but fail out-of-sample
- **Solution**: Use simpler strategies, validate on hold-out data, use constraints

### 3. Ignoring Transaction Costs
- **Problem**: High-turnover strategies lose money to fees
- **Solution**: Include transaction costs in optimization, use rebalancing thresholds

### 4. Correlation Breakdown
- **Problem**: Correlations change in crisis periods
- **Solution**: Use conservative correlation estimates, stress test portfolios

### 5. Leverage Risk
- **Problem**: Full Kelly can suggest excessive leverage
- **Solution**: Always use Fractional Kelly, set maximum position sizes

## Examples

See `examples/portfolio_optimization_example.py` for comprehensive examples including:
- Basic position sizing with Kelly Criterion
- Risk Parity allocation
- Strategy comparison
- Integration with Portfolio Manager
- Risk management rules
- Portfolio rebalancing

## API Reference

### Classes

#### PortfolioOptimizer
Main optimization engine.

**Methods:**
- `kelly_criterion()`: Calculate Kelly position size
- `kelly_portfolio()`: Kelly-optimal portfolio weights
- `risk_parity()`: Risk parity allocation
- `equal_weight()`: Equal weight allocation
- `volatility_weighted()`: Volatility-adjusted allocation
- `mean_variance_optimization()`: Markowitz optimization
- `apply_risk_constraints()`: Apply risk management rules
- `calculate_position_sizes()`: Convert weights to shares
- `calculate_rebalancing_trades()`: Calculate rebalancing trades

#### AssetMetrics
Container for asset information.

**Attributes:**
- `symbol`: Stock symbol
- `expected_return`: Expected annual return
- `volatility`: Annual volatility (std dev)
- `sharpe_ratio`: Risk-adjusted return
- `win_rate`: Historical win rate
- `current_price`: Current market price

#### PortfolioWeights
Container for portfolio allocation.

**Attributes:**
- `weights`: Dictionary of symbol -> weight
- `method`: Position sizing method used
- `expected_return`: Expected portfolio return
- `expected_volatility`: Expected portfolio volatility
- `sharpe_ratio`: Expected Sharpe ratio

#### RiskConstraints
Risk management parameters.

**Attributes:**
- `max_position_size`: Maximum per-position allocation
- `max_portfolio_volatility`: Maximum portfolio vol
- `max_drawdown`: Maximum acceptable drawdown
- `max_correlation`: Maximum correlation between positions
- `min_positions`: Minimum number of positions
- `max_positions`: Maximum number of positions

### Enums

#### PositionSizingMethod
- `KELLY_CRITERION`: Full Kelly
- `FRACTIONAL_KELLY`: Fractional Kelly
- `RISK_PARITY`: Risk parity
- `EQUAL_WEIGHT`: Equal weight
- `VOLATILITY_WEIGHTED`: Volatility-weighted
- `MAX_SHARPE`: Maximum Sharpe ratio
- `MIN_VARIANCE`: Minimum variance

#### RiskLevel
- `CONSERVATIVE`: Low risk tolerance
- `MODERATE`: Medium risk tolerance
- `AGGRESSIVE`: High risk tolerance
- `VERY_AGGRESSIVE`: Very high risk tolerance

## Further Reading

- **Kelly Criterion**:
  - [Original Paper](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf)
  - [Thorp's Application to Gambling and Investing](https://www.edwardothorp.com/id9.html)

- **Risk Parity**:
  - [All About Risk Parity](https://www.aqr.com/Insights/Research/White-Papers/Understanding-Risk-Parity)

- **Mean-Variance Optimization**:
  - [Markowitz Portfolio Theory](https://www.jstor.org/stable/2975974)
  - [Black-Litterman Model](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf)

- **Position Sizing**:
  - [Van Tharp's Definitive Guide to Position Sizing](https://www.vantharp.com/)
  - [Ralph Vince's Portfolio Mathematics](https://www.wiley.com/en-us/The+Handbook+of+Portfolio+Mathematics%3A+Formulas+for+Optimal+Allocation+%26+Leverage-p-9780471757689)
