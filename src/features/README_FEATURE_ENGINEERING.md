# Feature Engineering Module

## Overview

The Feature Engineering module provides comprehensive tools for transforming raw features (technical, fundamental, and whale) into a unified, engineered feature matrix suitable for machine learning models in the BIST AI Trading System.

## Key Features

### 1. Feature Combination
- Combine technical indicators, fundamental metrics, and whale activity features
- Automatic prefix handling to avoid column name conflicts
- Support for multiple data sources

### 2. Time Series Features
- **Lag Features**: Create lagged versions of features for time series modeling
- **Rolling Windows**: Calculate rolling statistics (mean, std, min, max, median)
- **Difference Features**: Create first and higher-order differences
- **Percentage Changes**: Track relative changes over time

### 3. Feature Scaling
Multiple scaling methods supported:
- **Standard Scaling**: Zero mean, unit variance
- **MinMax Scaling**: Scale to [0, 1] range
- **Robust Scaling**: Robust to outliers using median and IQR
- **Quantile Transformation**: Transform to uniform distribution
- **Power Transformation**: Yeo-Johnson or Box-Cox transformation

### 4. Missing Value Handling
Various imputation strategies:
- **Forward Fill**: Propagate last valid observation forward
- **Backward Fill**: Use next valid observation to fill gap
- **Mean/Median/Mode**: Statistical imputation
- **KNN Imputation**: Use K-nearest neighbors
- **Constant Value**: Fill with specified constant
- **Interpolation**: Linear interpolation for time series

### 5. Categorical Encoding
- **Label Encoding**: Convert categories to integers
- **One-Hot Encoding**: Create binary columns for each category
- **Target Encoding**: Encode based on target variable statistics

### 6. Feature Interactions
Create new features from combinations:
- Multiplication
- Division (with zero protection)
- Addition
- Subtraction

### 7. Feature Selection
- **SelectKBest**: Select top K features by statistical tests
- **SelectPercentile**: Select top percentile of features
- Support for both classification and regression tasks

### 8. Time-Based Features
Extract temporal information:
- Year, month, day, day of week
- Quarter, week of year
- Month/quarter start/end indicators
- Cyclical encoding (sin/cos) for periodic features

## Quick Start

### Basic Usage

```python
from src.features import FeatureEngineer, FeatureConfig, ScalingMethod

# Create configuration
config = FeatureConfig(
    scaling_method=ScalingMethod.STANDARD,
    create_lags=True,
    lag_periods=[1, 5, 10, 20],
    create_rolling=True,
    rolling_windows=[5, 10, 20]
)

# Initialize engineer
engineer = FeatureEngineer(config=config)

# Transform features
X_transformed = engineer.fit_transform(technical_features)
```

### Combining Multiple Feature Types

```python
# Combine technical, fundamental, and whale features
X_combined = engineer.fit_transform(
    X=None,
    technical_features=tech_df,
    fundamental_features=fund_df,
    whale_features=whale_df
)
```

### Train-Test Split

```python
# Fit on training data
X_train = engineer.fit_transform(train_features)

# Transform test data using fitted transformers
X_test = engineer.transform(test_features)
```

## Configuration Options

### FeatureConfig Parameters

```python
@dataclass
class FeatureConfig:
    # Scaling
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scale_target: bool = False

    # Imputation
    imputation_method: ImputationMethod = ImputationMethod.FORWARD_FILL
    imputation_constant: float = 0.0
    knn_neighbors: int = 5

    # Lag features
    create_lags: bool = True
    lag_periods: List[int] = [1, 2, 3, 5, 10, 20]
    lag_features: Optional[List[str]] = None  # None = all numeric

    # Rolling features
    create_rolling: bool = True
    rolling_windows: List[int] = [5, 10, 20, 60]
    rolling_features: Optional[List[str]] = None

    # Difference features
    create_differences: bool = True
    diff_periods: List[int] = [1, 5, 20]

    # Feature interactions
    create_interactions: bool = False
    interaction_pairs: Optional[List[Tuple[str, str]]] = None

    # Feature selection
    feature_selection: bool = False
    selection_method: str = "kbest"
    n_features: Optional[int] = None
    selection_percentile: float = 75.0

    # Categorical encoding
    encoding_method: EncodingMethod = EncodingMethod.LABEL
    categorical_features: List[str] = []

    # Other options
    drop_na: bool = False
    fill_na_after_lag: bool = True
    preserve_timestamps: bool = True
```

## Examples

### Example 1: Minimal Configuration

```python
from src.features import FeatureEngineer, FeatureConfig

# Only create lag features with standard scaling
config = FeatureConfig(
    create_lags=True,
    lag_periods=[1, 5],
    create_rolling=False,
    create_differences=False
)

engineer = FeatureEngineer(config)
X = engineer.fit_transform(data)
```

### Example 2: Advanced Time Series Features

```python
config = FeatureConfig(
    scaling_method=ScalingMethod.ROBUST,
    create_lags=True,
    lag_periods=[1, 5, 10, 20, 60],
    lag_features=['close', 'volume', 'RSI_14'],
    create_rolling=True,
    rolling_windows=[5, 10, 20, 60, 120],
    rolling_features=['close', 'volume'],
    create_differences=True,
    diff_periods=[1, 5, 20],
    imputation_method=ImputationMethod.FORWARD_FILL
)

engineer = FeatureEngineer(config)
X = engineer.fit_transform(price_data)
```

### Example 3: Feature Interactions

```python
# Define feature pairs for interactions
interaction_pairs = [
    ('close', 'volume'),
    ('RSI_14', 'MACD'),
    ('PE_ratio', 'PB_ratio'),
    ('whale_buy_volume', 'institutional_ownership_pct')
]

config = FeatureConfig(
    create_interactions=True,
    interaction_pairs=interaction_pairs
)

engineer = FeatureEngineer(config)
X = engineer.fit_transform(data)

# This creates features like:
# - close_x_volume
# - close_div_volume
# - RSI_14_x_MACD
# - RSI_14_div_MACD
# etc.
```

### Example 4: With Feature Selection

```python
config = FeatureConfig(
    create_lags=True,
    lag_periods=[1, 5, 10, 20],
    create_rolling=True,
    rolling_windows=[5, 20],
    feature_selection=True,
    selection_method='kbest',
    n_features=50  # Select top 50 features
)

engineer = FeatureEngineer(config)
X = engineer.fit_transform(features, y=target)

# Get feature importance
importance_df = engineer.get_feature_importance()
print(importance_df.head(10))
```

### Example 5: Custom Step-by-Step Pipeline

```python
from src.features import create_time_features, remove_correlated_features

engineer = FeatureEngineer(config=FeatureConfig(
    create_lags=False,
    create_rolling=False,
    scaling_method=ScalingMethod.NONE
))

# Step 1: Add time features
df = create_time_features(price_data)

# Step 2: Create specific lags
engineer.config.lag_periods = [1, 5, 10]
engineer.config.lag_features = ['close', 'volume']
df = engineer.create_lag_features(df)

# Step 3: Create rolling features
engineer.config.rolling_windows = [5, 20]
engineer.config.rolling_features = ['close']
df = engineer.create_rolling_features(df)

# Step 4: Remove correlated features
df = remove_correlated_features(df, threshold=0.95)

# Step 5: Scale
engineer.config.scaling_method = ScalingMethod.STANDARD
df = engineer.scale_features(df, fit=True)
```

## Utility Functions

### create_time_features
Extract time-based features from datetime index or column:

```python
from src.features import create_time_features

df = create_time_features(data, datetime_col='date')
# Adds: year, month, day, dayofweek, quarter, weekofyear,
# is_month_start, is_month_end, is_quarter_start, is_quarter_end,
# month_sin, month_cos, dayofweek_sin, dayofweek_cos
```

### remove_correlated_features
Remove highly correlated features to reduce multicollinearity:

```python
from src.features import remove_correlated_features

df_reduced = remove_correlated_features(
    df,
    threshold=0.95,
    method='pearson'  # or 'spearman', 'kendall'
)
```

### get_feature_statistics
Get comprehensive statistics for all features:

```python
from src.features import get_feature_statistics

stats = get_feature_statistics(df)
print(stats)
# Shows: dtype, missing_count, missing_percentage, unique_count,
# mean, std, min, max, median
```

## Custom Transformers

### LagFeatureTransformer

```python
from src.features import LagFeatureTransformer

transformer = LagFeatureTransformer(
    lag_periods=[1, 5, 10],
    feature_names=['close', 'volume'],
    fill_na=True
)

df_lagged = transformer.fit_transform(data)
```

### RollingFeatureTransformer

```python
from src.features import RollingFeatureTransformer

transformer = RollingFeatureTransformer(
    windows=[5, 10, 20],
    feature_names=['close', 'volume'],
    statistics=['mean', 'std', 'min', 'max']
)

df_rolling = transformer.fit_transform(data)
```

### DifferenceFeatureTransformer

```python
from src.features import DifferenceFeatureTransformer

transformer = DifferenceFeatureTransformer(
    periods=[1, 5],
    feature_names=['close', 'volume']
)

df_diff = transformer.fit_transform(data)
# Creates both diff and pct_change features
```

### FeatureInteractionTransformer

```python
from src.features import FeatureInteractionTransformer

transformer = FeatureInteractionTransformer(
    interaction_pairs=[('close', 'volume'), ('RSI_14', 'MACD')],
    interaction_types=['multiply', 'divide']
)

df_interact = transformer.fit_transform(data)
```

## Best Practices

### 1. Feature Engineering for Time Series

```python
# Good: Preserve temporal order and use time-aware methods
config = FeatureConfig(
    imputation_method=ImputationMethod.FORWARD_FILL,  # Not KNN
    create_lags=True,
    create_rolling=True,
    drop_na=False  # Don't lose temporal information
)
```

### 2. Train-Test Split

```python
# Always fit on training data only
engineer = FeatureEngineer(config)
X_train = engineer.fit_transform(train_data)
X_test = engineer.transform(test_data)  # Use fitted transformers
```

### 3. Handling Large Feature Sets

```python
# Use feature selection to reduce dimensionality
config = FeatureConfig(
    create_lags=True,
    lag_periods=[1, 5, 10, 20, 60],
    create_rolling=True,
    rolling_windows=[5, 10, 20, 60],
    feature_selection=True,
    selection_percentile=50.0  # Keep top 50%
)
```

### 4. Dealing with Outliers

```python
# Use robust scaling for data with outliers
config = FeatureConfig(
    scaling_method=ScalingMethod.ROBUST,  # Or QUANTILE
    imputation_method=ImputationMethod.MEDIAN  # More robust than mean
)
```

### 5. Memory Efficiency

```python
# Be selective about which features to transform
config = FeatureConfig(
    lag_features=['close', 'volume', 'RSI_14'],  # Not all features
    rolling_features=['close', 'volume'],
    create_differences=False  # Disable if not needed
)
```

## Performance Considerations

### Memory Usage
- Creating many lag and rolling features can significantly increase memory usage
- Use `lag_features` and `rolling_features` parameters to limit feature creation
- Consider using `drop_na=True` to reduce dataset size

### Computation Time
- Lag and rolling features are fast (O(n))
- Feature selection with many features can be slow
- KNN imputation is slower than simpler methods

### Recommendations
1. Start with a small subset of features
2. Monitor memory usage with large datasets
3. Use feature selection to reduce final feature count
4. Consider using Dask or similar for very large datasets

## Integration with Other Modules

### With Technical Indicators

```python
from src.features.technical import MomentumIndicators

# Calculate technical indicators
tech_calc = MomentumIndicators(price_data)
tech_features = tech_calc.calculate_all_momentum_indicators()

# Engineer features
engineer = FeatureEngineer(config)
X = engineer.fit_transform(technical_features=tech_features)
```

### With Fundamental Features

```python
from src.features.fundamental import ValuationCalculator

# Calculate fundamental metrics
calc = ValuationCalculator('THYAO')
metrics = calc.calculate_from_statements(income_stmt, balance_sheet)
fund_features = pd.DataFrame([metrics.to_dict()])

# Combine and engineer
X = engineer.fit_transform(
    technical_features=tech_features,
    fundamental_features=fund_features
)
```

### With Models

```python
from sklearn.ensemble import RandomForestClassifier

# Engineer features
engineer = FeatureEngineer(config)
X_train = engineer.fit_transform(train_data)
X_test = engineer.transform(test_data)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Troubleshooting

### Issue: Too many NaN values after creating lag features

```python
# Solution: Reduce lag periods or use backward fill
config = FeatureConfig(
    lag_periods=[1, 5, 10],  # Fewer lags
    fill_na_after_lag=True,
    imputation_method=ImputationMethod.BACKWARD_FILL
)
```

### Issue: Features on different scales causing poor model performance

```python
# Solution: Always use scaling
config = FeatureConfig(
    scaling_method=ScalingMethod.STANDARD  # or ROBUST
)
```

### Issue: Model overfitting due to too many features

```python
# Solution: Enable feature selection
config = FeatureConfig(
    feature_selection=True,
    selection_percentile=50.0  # Keep top 50%
)
```

### Issue: Data leakage in time series

```python
# Solution: Always fit on training data, transform test data
engineer.fit_transform(train_data)  # Fit here
engineer.transform(test_data)  # Only transform here
```

## API Reference

See inline documentation in `feature_engineering.py` for detailed API reference.

## Examples

Complete working examples are available in:
- `/examples/feature_engineering_example.py`

Run examples:
```bash
cd /home/user/BISTML
python examples/feature_engineering_example.py
```

## Testing

Unit tests are located in:
- `/tests/test_features/test_feature_engineering.py`

Run tests:
```bash
pytest tests/test_features/test_feature_engineering.py
```

## Version History

- **v1.0.0** (2025-11-16): Initial implementation
  - Feature combination from multiple sources
  - Lag, rolling, and difference features
  - Multiple scaling and imputation methods
  - Feature selection capabilities
  - Categorical encoding
  - Feature interactions
  - Time-based features

## Author

BIST AI Trading System Development Team

## License

MIT License
