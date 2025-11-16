"""
Example: Model Training with the ModelTrainer

This example demonstrates how to use the ModelTrainer class to orchestrate
training of different models with various configurations.

Usage:
    python examples/train_models_example.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainer import (
    ModelTrainer,
    RetrainingScheduler,
    TrainingConfig,
    ModelType,
    TaskType,
    SplitStrategy,
    TuningMethod,
    quick_train
)


def example_1_basic_xgboost_training():
    """Example 1: Basic XGBoost model training"""
    print("\n" + "=" * 80)
    print("Example 1: Basic XGBoost Regression")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = X.iloc[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5

    # Configure training
    config = TrainingConfig(
        model_type=ModelType.XGBOOST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        split_strategy=SplitStrategy.TIME_SERIES,
        use_cross_validation=True,
        cv_folds=5,
        tune_hyperparameters=False,
        save_model=True,
        model_dir="models/example1",
        model_name="xgboost_basic"
    )

    # Train
    trainer = ModelTrainer(config)
    result = trainer.train(X, y)

    # Display results
    print(f"\nTest RMSE: {result.test_metrics['rmse']:.6f}")
    print(f"Test R2: {result.test_metrics['r2']:.6f}")
    print(f"Training Time: {result.training_time:.2f}s")

    return result


def example_2_hyperparameter_tuning():
    """Example 2: XGBoost with hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("Example 2: XGBoost with Grid Search Hyperparameter Tuning")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(800, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y = X.iloc[:, :3].sum(axis=1) + np.random.randn(800) * 0.3

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100],
        'min_child_weight': [1, 3]
    }

    # Configure training with hyperparameter tuning
    config = TrainingConfig(
        model_type=ModelType.XGBOOST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        tune_hyperparameters=True,
        tuning_method=TuningMethod.GRID_SEARCH,
        param_grid=param_grid,
        cv_folds=3,
        save_model=True,
        model_dir="models/example2",
        model_name="xgboost_tuned"
    )

    # Train
    trainer = ModelTrainer(config)
    result = trainer.train(X, y)

    # Display results
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Test RMSE: {result.test_metrics['rmse']:.6f}")
    print(f"CV Mean Score: {result.cv_scores['mean_score']:.6f}")

    return result


def example_3_classification_random_forest():
    """Example 3: Random Forest Classification for trading signals"""
    print("\n" + "=" * 80)
    print("Example 3: Random Forest Classification")
    print("=" * 80)

    # Create sample classification data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame(
        np.random.randn(n_samples, 20),
        columns=[f'feature_{i}' for i in range(20)]
    )

    # Create trading signal labels (0-4: STRONG_SELL to STRONG_BUY)
    signal_scores = X.iloc[:, :5].sum(axis=1)
    y = pd.cut(signal_scores, bins=5, labels=[0, 1, 2, 3, 4]).astype(int)

    # Configure training
    config = TrainingConfig(
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
        task_type=TaskType.CLASSIFICATION,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        split_strategy=SplitStrategy.STRATIFIED,
        use_cross_validation=True,
        cv_folds=5,
        cv_strategy='stratified',
        eval_metrics=['accuracy', 'precision', 'recall', 'f1'],
        save_model=True,
        model_dir="models/example3",
        model_name="rf_classifier"
    )

    # Train
    trainer = ModelTrainer(config)
    result = trainer.train(X, y)

    # Display results
    print(f"\nTest Accuracy: {result.test_metrics['accuracy']:.6f}")
    print(f"Test F1 Score: {result.test_metrics['f1']:.6f}")
    print(f"Test Precision: {result.test_metrics['precision']:.6f}")

    return result


def example_4_compare_models():
    """Example 4: Compare multiple models"""
    print("\n" + "=" * 80)
    print("Example 4: Compare Multiple Models")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(1000, 15),
        columns=[f'feature_{i}' for i in range(15)]
    )
    y = X.iloc[:, :5].sum(axis=1) + np.random.randn(1000) * 0.5

    # Define models to compare
    models_config = [
        {
            'model_type': ModelType.XGBOOST_REGRESSOR,
            'model_name': 'xgboost_comparison'
        },
        {
            'model_type': ModelType.LIGHTGBM_REGRESSOR,
            'model_name': 'lightgbm_comparison'
        }
    ]

    # Base configuration
    config = TrainingConfig(
        task_type=TaskType.REGRESSION,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        use_cross_validation=True,
        cv_folds=3,
        save_model=False,  # Don't save for comparison
        verbose=0
    )

    # Compare models
    trainer = ModelTrainer(config)
    comparison_df = trainer.compare_models(models_config, X, y)

    print("\nModel Comparison Results:")
    print(comparison_df.to_string())

    return comparison_df


def example_5_quick_train():
    """Example 5: Quick training for simple use cases"""
    print("\n" + "=" * 80)
    print("Example 5: Quick Train Function")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(500, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y = X.iloc[:, :3].sum(axis=1) + np.random.randn(500) * 0.3

    # Quick train with minimal configuration
    result = quick_train(
        model_type='xgboost_regressor',
        X=X,
        y=y,
        task_type='regression',
        test_size=0.2,
        save_model=False
    )

    print(f"\nTest RMSE: {result.test_metrics['rmse']:.6f}")
    print(f"Test R2: {result.test_metrics['r2']:.6f}")

    return result


def example_6_model_persistence():
    """Example 6: Save and load models"""
    print("\n" + "=" * 80)
    print("Example 6: Model Persistence (Save/Load)")
    print("=" * 80)

    # Create and train a model
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 8), columns=[f'feature_{i}' for i in range(8)])
    y = X.sum(axis=1) + np.random.randn(500) * 0.2

    config = TrainingConfig(
        model_type=ModelType.XGBOOST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        save_model=True,
        model_dir="models/example6",
        model_name="persistence_test",
        version_models=True
    )

    trainer = ModelTrainer(config)
    result = trainer.train(X, y)

    print(f"\nModel saved to: {result.model_path}")

    # Load the model
    loaded_model = trainer.load_model(result.model_path)
    print(f"Model loaded successfully!")

    # Make predictions with loaded model
    X_test = X.iloc[:10]
    predictions = loaded_model.predict(X_test)
    print(f"\nSample predictions: {predictions[:5]}")

    return loaded_model


def example_7_retraining_scheduler():
    """Example 7: Automated retraining scheduler"""
    print("\n" + "=" * 80)
    print("Example 7: Automated Retraining Scheduler")
    print("=" * 80)

    # Note: This example demonstrates the setup but doesn't run the scheduler
    # in production mode to avoid blocking

    # Data loader function
    def load_fresh_data():
        """Simulates loading fresh data"""
        print("Loading fresh data...")
        X = pd.DataFrame(np.random.randn(800, 10), columns=[f'feature_{i}' for i in range(10)])
        y = X.sum(axis=1) + np.random.randn(800) * 0.2
        return X, y

    # Configure trainer
    config = TrainingConfig(
        model_type=ModelType.XGBOOST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        save_model=True,
        model_dir="models/example7",
        model_name="scheduled_model",
        verbose=0
    )

    trainer = ModelTrainer(config)

    # Create scheduler
    try:
        scheduler = RetrainingScheduler(
            trainer=trainer,
            data_loader=load_fresh_data,
            schedule_type='interval',
            interval_days=7,  # Retrain every 7 days
            performance_threshold=0.8  # Only retrain if R2 < 0.8
        )

        print("\nRetraining scheduler created successfully!")
        print("Schedule: Retrain every 7 days")
        print("Performance threshold: R2 < 0.8")
        print("\nNote: In production, call scheduler.start() to begin automated retraining")

        # Example of manual trigger
        print("\nTriggering manual retraining...")
        scheduler.trigger_immediate_retrain()

        print("Manual retraining completed!")

    except ImportError as e:
        print(f"\nScheduler not available: {e}")
        print("Install APScheduler with: pip install apscheduler")

    return None


def example_8_time_series_lstm():
    """Example 8: LSTM for time series forecasting"""
    print("\n" + "=" * 80)
    print("Example 8: LSTM Time Series Forecasting")
    print("=" * 80)

    try:
        # Create time series data
        np.random.seed(42)
        n_samples = 1000
        time_steps = 60

        # Generate price-like time series
        prices = 100 + np.cumsum(np.random.randn(n_samples)) * 2

        # Create sequences
        X_sequences = []
        y_targets = []

        for i in range(n_samples - time_steps - 1):
            X_sequences.append(prices[i:i+time_steps])
            y_targets.append(prices[i+time_steps])

        X = np.array(X_sequences).reshape(-1, time_steps, 1)
        y = np.array(y_targets)

        # Convert to DataFrame for consistency
        X_df = pd.DataFrame(X.reshape(len(X), -1))

        # Configure training
        config = TrainingConfig(
            model_type=ModelType.LSTM,
            task_type=TaskType.TIME_SERIES_FORECASTING,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            split_strategy=SplitStrategy.TIME_SERIES,
            save_model=True,
            model_dir="models/example8",
            model_name="lstm_forecaster",
            eval_metrics=['mse', 'rmse', 'mae']
        )

        # Train
        trainer = ModelTrainer(config)
        result = trainer.train(X_df, y)

        print(f"\nTest RMSE: {result.test_metrics['rmse']:.6f}")
        print(f"Test MAE: {result.test_metrics['mae']:.6f}")

        return result

    except ImportError as e:
        print(f"\nLSTM model not available: {e}")
        print("Make sure TensorFlow is installed: pip install tensorflow>=2.13.0")
        return None


def main():
    """Run all examples"""
    print("\n")
    print("=" * 80)
    print("MODEL TRAINING EXAMPLES - BIST AI TRADING SYSTEM")
    print("=" * 80)

    examples = [
        ("Basic XGBoost Training", example_1_basic_xgboost_training),
        ("Hyperparameter Tuning", example_2_hyperparameter_tuning),
        ("Random Forest Classification", example_3_classification_random_forest),
        ("Model Comparison", example_4_compare_models),
        ("Quick Train Function", example_5_quick_train),
        ("Model Persistence", example_6_model_persistence),
        ("Retraining Scheduler", example_7_retraining_scheduler),
        # ("LSTM Time Series", example_8_time_series_lstm),  # Uncomment if TensorFlow available
    ]

    results = {}

    for name, example_func in examples:
        try:
            print(f"\n\nRunning: {name}")
            result = example_func()
            results[name] = result
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
