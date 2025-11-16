# Backtesting Simulator - Technical Overview

## Created Files

### 1. simulator.py (1,503 lines)
The core backtesting engine with comprehensive functionality.

**Key Components:**

#### Enumerations
- `BacktestMode`: HISTORICAL, WALK_FORWARD, MONTE_CARLO, COMBINED
- `OrderType`: MARKET, LIMIT, STOP, STOP_LIMIT
- `ExecutionModel`: IMMEDIATE, REALISTIC, PESSIMISTIC

#### Configuration Classes
- `BacktestConfig`: Main backtesting configuration
  - Date range, capital, commission, slippage
  - Position sizing, risk management
  - Execution model, timeframe

- `WalkForwardConfig`: Walk-forward analysis settings
  - Training/testing period lengths
  - Step size, reoptimization options
  - Anchored vs rolling window modes

- `MonteCarloConfig`: Monte Carlo simulation settings
  - Number of simulations
  - Randomization methods (shuffle, bootstrap, block bootstrap)
  - Confidence levels and random seed

#### Data Classes
- `Trade`: Individual trade record with full metrics
  - Entry/exit times and prices
  - P&L, commission, slippage
  - MAE/MFE tracking
  - Holding period analysis

- `BacktestResult`: Complete backtest results container
  - List of all trades
  - Equity curve DataFrame
  - Comprehensive metrics dictionary
  - Returns series and drawdown analysis

#### Main Simulator Class: `BacktestSimulator`

**Core Methods:**

1. **Historical Backtesting** (Line 342)
   ```python
   run_historical_backtest(signals=None, progress_callback=None)
   ```
   - Full historical simulation
   - Realistic order execution
   - Transaction cost modeling
   - Position management
   - Equity curve tracking

2. **Walk-Forward Analysis** (Line 410)
   ```python
   run_walk_forward_analysis(wf_config, strategy_optimizer=None, progress_callback=None)
   ```
   - Multiple train/test periods
   - Rolling or anchored windows
   - Optional parameter optimization
   - Out-of-sample validation
   - Aggregated performance metrics

3. **Monte Carlo Simulation** (Line 516)
   ```python
   run_monte_carlo_simulation(mc_config, base_result=None, progress_callback=None)
   ```
   - Trade sequence randomization
   - Bootstrap resampling
   - Block bootstrap
   - Confidence interval calculation
   - Risk distribution analysis

**Supporting Methods:**

- `load_price_data()`: Load and validate historical price data
- `_process_signals()`: Convert signals to orders
- `_execute_orders()`: Execute pending orders with slippage
- `_update_positions()`: Update position values and MAE/MFE
- `_calculate_results()`: Generate comprehensive metrics
- `_calculate_performance_metrics()`: Calculate 20+ performance metrics
- `_calculate_drawdown_series()`: Compute drawdown over time
- `plot_equity_curve()`: Visualize equity and drawdown
- `plot_monte_carlo_distribution()`: Plot MC results
- `export_results()`: Export to CSV, JSON, pickle

**Convenience Functions:**

- `run_quick_backtest()`: Simplified backtest interface
- `compare_strategies()`: Compare multiple strategies

### 2. example_usage.py (578 lines)
Comprehensive examples demonstrating all features.

**Examples Included:**

1. **Example 1: Historical Backtest**
   - Basic setup with sample data
   - Simple signal generation
   - Results display and export
   - Equity curve plotting

2. **Example 2: Walk-Forward Analysis**
   - Long-term data setup
   - Rolling window configuration
   - Strategy optimization
   - Period-by-period analysis

3. **Example 3: Monte Carlo Simulation**
   - Base backtest execution
   - MC configuration options
   - Statistical analysis
   - Distribution plotting

4. **Example 4: Strategy Comparison**
   - Multiple strategy definitions
   - Parallel backtesting
   - Comparative metrics
   - Performance ranking

5. **Example 5: Quick Backtest**
   - Minimal setup demonstration
   - Convenience function usage

### 3. test_simulator.py (379 lines)
Comprehensive unit tests for all components.

**Test Classes:**

1. `TestBacktestConfig`: Configuration validation
2. `TestTrade`: Trade calculation logic
3. `TestBacktestSimulator`: Core simulation functionality
4. `TestWalkForward`: Walk-forward period generation
5. `TestMonteCarlo`: Randomization methods
6. `TestMetrics`: Performance metric calculations

**Coverage:**
- Configuration validation
- Trade P&L calculations
- Backtest execution
- Walk-forward period generation
- Monte Carlo randomization
- Metric computation

### 4. README.md (12KB)
Complete documentation with:

- Feature overview
- Installation instructions
- Quick start guide
- Configuration reference
- API documentation
- Best practices
- Troubleshooting guide
- Integration examples

## Architecture

### Data Flow

```
Price Data + Signal Generator
           ↓
    BacktestSimulator
           ↓
    ┌──────┴──────┐
    ↓             ↓
Historical    Walk-Forward    Monte Carlo
Backtest        Analysis      Simulation
    ↓             ↓             ↓
    └──────┬──────┘             │
           ↓                    ↓
    BacktestResult      MC Statistics
           ↓                    ↓
    Performance Metrics  Confidence Intervals
           ↓                    ↓
    Visualization & Export
```

### Order Execution Pipeline

```
Signal → Order Creation → Order Queue → Execution with Slippage → Position Update
                                              ↓
                                         Commission
                                              ↓
                                      Transaction Record
                                              ↓
                                         Trade Record
                                              ↓
                                      Equity Curve Update
```

## Performance Metrics

The simulator calculates 25+ comprehensive metrics:

### Returns
- Total Return (absolute and percentage)
- Annualized Return
- Expected Return per trade

### Risk-Adjusted Returns
- Sharpe Ratio (risk-free rate adjusted)
- Sortino Ratio (downside deviation)
- Calmar Ratio (return/max drawdown)

### Trade Statistics
- Number of trades
- Win rate
- Profit factor
- Average win/loss
- Largest win/loss
- Expectancy

### Risk Metrics
- Maximum Drawdown
- Average holding period
- MAE (Maximum Adverse Excursion)
- MFE (Maximum Favorable Excursion)

### Distribution Metrics (Monte Carlo)
- Mean/median returns
- Standard deviation
- Probability of profit
- Confidence intervals (5th, 25th, 50th, 75th, 95th percentiles)
- Best/worst case scenarios

## Key Features

### 1. Realistic Transaction Modeling

**Commission:**
- Configurable commission rate
- Applied to both entry and exit
- Reduces final P&L

**Slippage:**
- Basis points configuration
- Different models (IMMEDIATE, REALISTIC, PESSIMISTIC)
- Simulates market impact

**Execution Models:**
- IMMEDIATE: No slippage (optimistic)
- REALISTIC: Configured slippage (recommended)
- PESSIMISTIC: 2x slippage (conservative)

### 2. Position Management

- Multiple concurrent positions
- Position size limits
- Risk per trade limits
- Automatic position tracking
- MAE/MFE monitoring
- Market value updates

### 3. Walk-Forward Analysis

**Benefits:**
- Out-of-sample validation
- Prevents overfitting
- Tests strategy robustness
- Simulates real-world deployment

**Modes:**
- Rolling window: Fixed-size moving window
- Anchored: Always start from beginning

**Process:**
1. Split data into train/test periods
2. Train/optimize on training data
3. Test on out-of-sample data
4. Roll forward and repeat
5. Aggregate results

### 4. Monte Carlo Simulation

**Randomization Methods:**

1. **Shuffle**: Simple random permutation
   - Preserves trade distribution
   - Tests sequence dependency

2. **Bootstrap**: Resample with replacement
   - Creates new trade sequences
   - Tests statistical significance

3. **Block Bootstrap**: Preserve correlations
   - Maintains temporal structure
   - Better for correlated trades

**Applications:**
- Risk assessment
- Worst-case analysis
- Strategy robustness
- Confidence intervals
- Position sizing optimization

## Integration with BIST System

### Signal Generator Integration

```python
from src.signals.generator import SignalGenerator

signal_gen = SignalGenerator(
    models=models,
    weights=weights,
    thresholds=thresholds
)

simulator = BacktestSimulator(
    config=config,
    signal_generator=signal_gen.generate_signals,
    price_data=price_data
)
```

### Portfolio Manager Integration

The simulator tracks positions similarly to PortfolioManager:
- Position tracking with cost basis
- Transaction history
- P&L calculation (realized and unrealized)
- Multiple cost basis methods compatible

### Data Format Compatibility

**Price Data:**
- Compatible with BIST data collectors
- Supports multi-symbol DataFrames
- Handles multiple timeframes
- Works with adjusted prices

**Signals:**
- Compatible with SignalGenerator output
- Supports TradingSignal format
- Handles confidence scores
- Integrates risk management

## Usage Patterns

### Pattern 1: Strategy Development

```python
# 1. Develop strategy
def my_strategy(timestamp, market_data):
    # Strategy logic
    return signals

# 2. Quick backtest
result = run_quick_backtest(
    start_date='2023-01-01',
    end_date='2023-12-31',
    signal_generator=my_strategy,
    price_data=price_data
)

# 3. Analyze
print(result.metrics)
```

### Pattern 2: Strategy Validation

```python
# 1. Historical backtest
hist_result = simulator.run_historical_backtest()

# 2. Walk-forward validation
wf_result = simulator.run_walk_forward_analysis(wf_config)

# 3. Monte Carlo robustness test
mc_result = simulator.run_monte_carlo_simulation(mc_config)

# 4. Make deployment decision
if (wf_result['combined_metrics']['consistency_score'] > 0.7 and
    mc_result['statistics']['prob_profit'] > 0.65):
    deploy_strategy()
```

### Pattern 3: Parameter Optimization

```python
# Test multiple parameter combinations
results = {}

for lookback in [10, 20, 30, 50]:
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        signal_gen = create_strategy(lookback, threshold)
        result = run_quick_backtest(...)
        results[(lookback, threshold)] = result.metrics

# Find best parameters
best_params = max(results.items(),
                  key=lambda x: x[1]['sharpe_ratio'])
```

## Performance Considerations

### Memory Usage
- Equity curve stored in memory
- Trade history accumulates
- For large backtests (years of minute data):
  - Process in chunks
  - Use sampling
  - Limit MC simulations

### Computation Time
- Historical backtest: Fast (seconds to minutes)
- Walk-forward: Moderate (minutes to hours)
- Monte Carlo: Depends on n_simulations (1000 sims ~ 1-5 min)

### Optimization Tips
1. Use appropriate timeframe (hourly vs minute)
2. Limit concurrent positions
3. Pre-filter symbols
4. Use progress callbacks
5. Parallelize Monte Carlo (future enhancement)

## Future Enhancements

Potential improvements:
1. Multi-threaded Monte Carlo
2. GPU acceleration for large simulations
3. Advanced order types (OCO, trailing stops)
4. Partial position sizing
5. Multiple portfolio support
6. Real-time backtest replay
7. Interactive visualization dashboard
8. Genetic algorithm optimization
9. Machine learning integration
10. Risk parity portfolio construction

## File Locations

```
/home/user/BISTML/src/backtesting/
├── simulator.py          # Main simulator (1,503 lines)
├── example_usage.py      # Examples (578 lines)
├── test_simulator.py     # Unit tests (379 lines)
├── README.md            # Documentation (12 KB)
└── SIMULATOR_OVERVIEW.md # This file
```

## Testing

Run unit tests:
```bash
cd /home/user/BISTML/src/backtesting
python test_simulator.py
```

Run examples:
```bash
python example_usage.py
```

## Dependencies

Required packages:
- pandas
- numpy
- scipy
- matplotlib
- seaborn

All standard packages in scientific Python stack.

## Summary

The backtesting simulator provides:

1. **Comprehensive Testing**: Historical, walk-forward, and Monte Carlo
2. **Realistic Modeling**: Transaction costs, slippage, position management
3. **Rich Metrics**: 25+ performance and risk metrics
4. **Flexibility**: Configurable execution models, timeframes, parameters
5. **Integration**: Works seamlessly with existing BIST system
6. **Visualization**: Equity curves, distributions, drawdown analysis
7. **Export**: Multiple formats (CSV, JSON, pickle)
8. **Documentation**: Complete examples and API reference
9. **Testing**: Comprehensive unit test coverage

Total code: ~2,500 lines of production-quality Python code with full documentation and examples.
