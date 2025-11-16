# Model Training Guide - BIST AI Trading System

This guide provides comprehensive documentation for training models using the `ModelTrainer` class.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Data Splitting Strategies](#data-splitting-strategies)
5. [Cross-Validation](#cross-validation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Persistence](#model-persistence)
8. [Automated Retraining](#automated-retraining)
9. [Model Comparison](#model-comparison)
10. [Best Practices](#best-practices)

## Overview

The `ModelTrainer` class provides a unified interface for training, evaluating, and managing all machine learning models in the BIST AI Trading System. It supports:

- **Multiple Model Types**: LSTM, GRU, XGBoost, LightGBM, Random Forest, ANN
- **Task Types**: Regression, Classification, Time Series Forecasting
- **Data Splitting**: Time series, random, stratified, walk-forward
- **Cross-Validation**: Time series split, stratified k-fold, k-fold
- **Hyperparameter Tuning**: Grid search, random search, Optuna
- **Model Versioning**: Automatic versioning and artifact management
- **Automated Retraining**: Scheduled retraining with performance monitoring

## Quick Start

### Basic Training

```python
from src.models.trainer import ModelTrainer, TrainingConfig, ModelType, TaskType
import pandas as pd

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('targets.csv')['target']

# Configure training
config = TrainingConfig(
    model_type=ModelType.XGBOOST_REGRESSOR,
    task_type=TaskType.REGRESSION,
    save_model=True,
    model_dir='models/my_model'
)

# Train
trainer = ModelTrainer(config)
result = trainer.train(X, y)

# View results
print(f"Test RMSE: {result.test_metrics['rmse']:.4f}")
print(f"Test R2: {result.test_metrics['r2']:.4f}")
```

### Quick Train Function

For simple use cases, use the `quick_train` function:

```python
from src.models.trainer import quick_train

result = quick_train(
    model_type='xgboost_regressor',
    X=X,
    y=y,
    task_type='regression',
    test_size=0.2
)
```

## Configuration

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | ModelType | XGBOOST_REGRESSOR | Type of model to train |
| `task_type` | TaskType | REGRESSION | ML task type |
| `train_size` | float | 0.7 | Training set proportion |
| `val_size` | float | 0.15 | Validation set proportion |
| `test_size` | float | 0.15 | Test set proportion |
| `split_strategy` | SplitStrategy | TIME_SERIES | Data splitting strategy |
| `use_cross_validation` | bool | True | Enable cross-validation |
| `cv_folds` | int | 5 | Number of CV folds |
| `tune_hyperparameters` | bool | True | Enable hyperparameter tuning |
| `tuning_method` | TuningMethod | GRID_SEARCH | Tuning method |
| `save_model` | bool | True | Save trained model |
| `version_models` | bool | True | Enable model versioning |

### Model Types

```python
from src.models.trainer import ModelType

# Forecasting Models
ModelType.LSTM
ModelType.GRU
ModelType.XGBOOST_REGRESSOR
ModelType.LIGHTGBM_REGRESSOR

# Classification Models
ModelType.RANDOM_FOREST_CLASSIFIER
ModelType.ANN_CLASSIFIER
ModelType.XGBOOST_CLASSIFIER
```

### Task Types

```python
from src.models.trainer import TaskType

TaskType.REGRESSION
TaskType.CLASSIFICATION
TaskType.TIME_SERIES_FORECASTING
```

## Data Splitting Strategies

### Time Series Split

Maintains temporal order - essential for time series data:

```python
config = TrainingConfig(
    split_strategy=SplitStrategy.TIME_SERIES,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)
```

**Data Flow:**
```
[Train................................][Val......][Test.....]
   70%                                    15%        15%
```

### Random Split

Random shuffling - suitable for i.i.d. data:

```python
config = TrainingConfig(
    split_strategy=SplitStrategy.RANDOM,
    shuffle_data=True
)
```

### Stratified Split

Maintains class distribution - ideal for imbalanced classification:

```python
config = TrainingConfig(
    split_strategy=SplitStrategy.STRATIFIED,
    task_type=TaskType.CLASSIFICATION
)
```

## Cross-Validation

### Time Series Cross-Validation

```python
config = TrainingConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy='time_series'
)
```

**Visualization:**
```
Fold 1: [Train][Test]
Fold 2: [Train......][Test]
Fold 3: [Train............][Test]
Fold 4: [Train..................][Test]
Fold 5: [Train......................][Test]
```

### Stratified K-Fold

For classification tasks:

```python
config = TrainingConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy='stratified'
)
```

### Standard K-Fold

```python
config = TrainingConfig(
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy='kfold'
)
```

## Hyperparameter Tuning

### Grid Search

Exhaustive search over parameter grid:

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5]
}

config = TrainingConfig(
    model_type=ModelType.XGBOOST_REGRESSOR,
    tune_hyperparameters=True,
    tuning_method=TuningMethod.GRID_SEARCH,
    param_grid=param_grid,
    cv_folds=5
)

trainer = ModelTrainer(config)
result = trainer.train(X, y)

print(f"Best parameters: {result.best_params}")
```

### Random Search

More efficient for large parameter spaces:

```python
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9, 11],
    'n_estimators': [50, 100, 200, 300, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}

config = TrainingConfig(
    tune_hyperparameters=True,
    tuning_method=TuningMethod.RANDOM_SEARCH,
    n_iter_random_search=50,
    param_grid=param_distributions
)
```

### Optuna (Advanced)

Bayesian optimization for efficient hyperparameter search:

```python
# Requires: pip install optuna

param_ranges = {
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'min_child_weight': {'type': 'int', 'low': 1, 'high': 10}
}

config = TrainingConfig(
    tune_hyperparameters=True,
    tuning_method=TuningMethod.OPTUNA,
    optuna_n_trials=100,
    param_grid=param_ranges
)
```

## Model Persistence

### Saving Models

Models are automatically saved when `save_model=True`:

```python
config = TrainingConfig(
    save_model=True,
    model_dir='models/production',
    model_name='xgboost_stock_predictor',
    version_models=True  # Adds timestamp to model name
)

trainer = ModelTrainer(config)
result = trainer.train(X, y)

print(f"Model saved to: {result.model_path}")
```

**Saved Artifacts:**
```
models/production/xgboost_stock_predictor_v20251116_143022/
├── model.pkl                    # Trained model
├── training_result.json         # Training metrics and metadata
├── feature_importance.csv       # Feature importance scores
└── config.json                  # Training configuration
```

### Loading Models

```python
# Load a saved model
model_path = "models/production/xgboost_stock_predictor_v20251116_143022/model.pkl"
trainer = ModelTrainer()
loaded_model = trainer.load_model(model_path)

# Make predictions
predictions = loaded_model.predict(X_new)
```

## Automated Retraining

### Setup Retraining Scheduler

```python
from src.models.trainer import RetrainingScheduler

# Data loader function
def load_fresh_data():
    """Load fresh training data"""
    X = fetch_latest_features()  # Your data loading logic
    y = fetch_latest_targets()
    return X, y

# Configure trainer
config = TrainingConfig(
    model_type=ModelType.XGBOOST_REGRESSOR,
    task_type=TaskType.REGRESSION,
    save_model=True,
    model_dir='models/production'
)
trainer = ModelTrainer(config)

# Create scheduler
scheduler = RetrainingScheduler(
    trainer=trainer,
    data_loader=load_fresh_data,
    schedule_type='interval',
    interval_days=7,  # Retrain every 7 days
    performance_threshold=0.85  # Only retrain if R2 < 0.85
)

# Start automated retraining
scheduler.start()
```

### Interval-based Scheduling

```python
scheduler = RetrainingScheduler(
    trainer=trainer,
    data_loader=load_fresh_data,
    schedule_type='interval',
    interval_days=7
)
scheduler.start()
```

### Cron-based Scheduling

```python
scheduler = RetrainingScheduler(
    trainer=trainer,
    data_loader=load_fresh_data,
    schedule_type='cron',
    cron_expression='0 2 * * 1'  # Every Monday at 2 AM
)
scheduler.start()
```

### Manual Trigger

```python
# Trigger immediate retraining
scheduler.trigger_immediate_retrain()
```

## Model Comparison

Compare multiple models on the same dataset:

```python
models_config = [
    {
        'model_type': ModelType.XGBOOST_REGRESSOR,
        'model_name': 'xgboost'
    },
    {
        'model_type': ModelType.LIGHTGBM_REGRESSOR,
        'model_name': 'lightgbm'
    },
    {
        'model_type': ModelType.RANDOM_FOREST_CLASSIFIER,
        'model_name': 'random_forest'
    }
]

config = TrainingConfig(
    task_type=TaskType.REGRESSION,
    use_cross_validation=True,
    cv_folds=5
)

trainer = ModelTrainer(config)
comparison_df = trainer.compare_models(models_config, X, y)

print(comparison_df)
```

**Output:**
```
      model_type  training_time  test_rmse  test_r2  cv_mean_score  cv_std_score
0  xgboost          12.34         0.123      0.89       0.87          0.03
1  lightgbm          8.76         0.118      0.91       0.89          0.02
2  random_forest    15.23         0.145      0.85       0.83          0.04
```

## Best Practices

### 1. Time Series Data

Always use time series split and cross-validation:

```python
config = TrainingConfig(
    split_strategy=SplitStrategy.TIME_SERIES,
    cv_strategy='time_series',
    shuffle_data=False  # Never shuffle time series
)
```

### 2. Imbalanced Classification

Use stratified splitting and appropriate class weights:

```python
config = TrainingConfig(
    split_strategy=SplitStrategy.STRATIFIED,
    cv_strategy='stratified',
    task_type=TaskType.CLASSIFICATION
)
```

### 3. Feature Engineering Integration

```python
from src.features.feature_engineering import FeatureEngineer, FeatureConfig

# Configure feature engineering
feature_config = FeatureConfig(
    create_lags=True,
    lag_periods=[1, 5, 10, 20],
    create_rolling=True,
    rolling_windows=[5, 10, 20]
)

# Apply feature engineering
engineer = FeatureEngineer(feature_config)
X_engineered = engineer.fit_transform(X)

# Train model
result = trainer.train(X_engineered, y)
```

### 4. Model Versioning

Always enable versioning in production:

```python
config = TrainingConfig(
    save_model=True,
    version_models=True,
    model_name='production_model'
)
```

### 5. Early Stopping

Enable early stopping to prevent overfitting:

```python
config = TrainingConfig(
    early_stopping=True,
    early_stopping_rounds=20
)
```

### 6. Performance Monitoring

Track training history:

```python
trainer = ModelTrainer(config)

# Train multiple models
for model_type in [ModelType.XGBOOST_REGRESSOR, ModelType.LIGHTGBM_REGRESSOR]:
    config.model_type = model_type
    result = trainer.train(X, y)

# View all training history
for i, result in enumerate(trainer.training_history):
    print(f"{i+1}. {result.model_type}: R2={result.test_metrics['r2']:.4f}")
```

### 7. Hyperparameter Tuning Strategy

- **Development**: Use quick grid search with fewer parameters
- **Staging**: Use random search with more iterations
- **Production**: Use Optuna for optimal parameters

```python
# Development
param_grid = {
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'n_estimators': [100]
}

# Production
param_ranges = {
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500}
}
```

## Complete Example

Here's a complete example putting it all together:

```python
from src.models.trainer import (
    ModelTrainer, TrainingConfig, ModelType, TaskType, SplitStrategy
)
from src.features.feature_engineering import FeatureEngineer, FeatureConfig
import pandas as pd

# 1. Load and prepare data
X = pd.read_csv('features.csv')
y = pd.read_csv('targets.csv')['price']

# 2. Feature engineering
feature_config = FeatureConfig(
    create_lags=True,
    lag_periods=[1, 5, 10, 20],
    create_rolling=True,
    rolling_windows=[5, 10, 20],
    scaling_method='standard'
)

engineer = FeatureEngineer(feature_config)
X_engineered = engineer.fit_transform(X, y)

# 3. Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

# 4. Configure training
config = TrainingConfig(
    model_type=ModelType.XGBOOST_REGRESSOR,
    task_type=TaskType.REGRESSION,
    split_strategy=SplitStrategy.TIME_SERIES,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    use_cross_validation=True,
    cv_folds=5,
    cv_strategy='time_series',
    tune_hyperparameters=True,
    param_grid=param_grid,
    early_stopping=True,
    early_stopping_rounds=20,
    save_model=True,
    model_dir='models/production',
    model_name='stock_price_predictor',
    version_models=True
)

# 5. Train
trainer = ModelTrainer(config)
result = trainer.train(X_engineered, y)

# 6. Evaluate
print("\n" + "="*60)
print("Training Results")
print("="*60)
print(f"Model: {result.model_type}")
print(f"Training Time: {result.training_time:.2f}s")
print(f"\nTest Metrics:")
print(f"  RMSE: {result.test_metrics['rmse']:.6f}")
print(f"  MAE: {result.test_metrics['mae']:.6f}")
print(f"  R2: {result.test_metrics['r2']:.6f}")
print(f"  Directional Accuracy: {result.test_metrics['directional_accuracy']:.4f}")

if result.best_params:
    print(f"\nBest Parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")

if result.cv_scores:
    print(f"\nCross-Validation:")
    print(f"  Mean Score: {result.cv_scores['mean_score']:.6f}")
    print(f"  Std Score: {result.cv_scores['std_score']:.6f}")

print(f"\nModel saved to: {result.model_path}")
print("="*60)

# 7. Setup automated retraining (optional)
from src.models.trainer import RetrainingScheduler

def load_data():
    X = fetch_latest_data()
    y = fetch_latest_targets()
    return engineer.transform(X), y

scheduler = RetrainingScheduler(
    trainer=trainer,
    data_loader=load_data,
    interval_days=7,
    performance_threshold=0.85
)
scheduler.start()
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or use data streaming

```python
config = TrainingConfig(
    # For tree-based models, limit data size
    # For neural networks, reduce batch size in model params
)
```

### Issue: Overfitting

**Solution**: Enable early stopping and regularization

```python
config = TrainingConfig(
    early_stopping=True,
    early_stopping_rounds=20
)

param_grid = {
    'reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization
    'reg_lambda': [1.0, 1.5, 2.0]  # L2 regularization
}
```

### Issue: Long Training Time

**Solution**: Use random search or reduce parameter grid

```python
config = TrainingConfig(
    tuning_method=TuningMethod.RANDOM_SEARCH,
    n_iter_random_search=30  # Instead of exhaustive grid search
)
```

## Further Reading

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)

## Support

For issues or questions:
1. Check the examples in `examples/train_models_example.py`
2. Review this documentation
3. Consult the source code with inline documentation
