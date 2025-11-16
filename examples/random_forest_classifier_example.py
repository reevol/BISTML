"""
Example: Using Random Forest Classifier for Trading Signal Generation

This example demonstrates how to use the TradingSignalClassifier to generate
trading signals (Strong BUY, BUY, HOLD, SELL, Strong SELL) based on technical
and fundamental features.

The example covers:
1. Creating synthetic training data
2. Training the classifier with class balancing
3. Making predictions
4. Evaluating model performance
5. Analyzing feature importance
6. Saving and loading the model
7. Cross-validation
8. Grid search for hyperparameter tuning

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classification import TradingSignalClassifier, create_signal_labels


def create_sample_features(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create sample technical and fundamental features for demonstration.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        Feature DataFrame
    """
    np.random.seed(random_state)

    # Technical indicators (momentum, trend, volatility)
    features = pd.DataFrame({
        # Momentum indicators
        'RSI_14': np.random.uniform(20, 80, n_samples),
        'MACD': np.random.normal(0, 2, n_samples),
        'MACD_Signal': np.random.normal(0, 1.5, n_samples),
        'Stoch_K': np.random.uniform(10, 90, n_samples),
        'Stoch_D': np.random.uniform(10, 90, n_samples),
        'Williams_R': np.random.uniform(-100, -10, n_samples),
        'ROC_12': np.random.normal(0, 5, n_samples),

        # Trend indicators
        'SMA_50': np.random.uniform(80, 120, n_samples),
        'SMA_200': np.random.uniform(80, 120, n_samples),
        'EMA_12': np.random.uniform(80, 120, n_samples),
        'ADX_14': np.random.uniform(10, 60, n_samples),
        'Plus_DI_14': np.random.uniform(10, 40, n_samples),
        'Minus_DI_14': np.random.uniform(10, 40, n_samples),

        # Volatility indicators
        'ATR_14': np.random.uniform(1, 10, n_samples),
        'Bollinger_Upper': np.random.uniform(105, 125, n_samples),
        'Bollinger_Lower': np.random.uniform(75, 95, n_samples),
        'Bollinger_Width': np.random.uniform(5, 30, n_samples),

        # Volume indicators
        'Volume_SMA_Ratio': np.random.uniform(0.5, 2.0, n_samples),
        'OBV': np.random.normal(0, 1000000, n_samples),

        # Fundamental indicators (example)
        'PE_Ratio': np.random.uniform(5, 30, n_samples),
        'PB_Ratio': np.random.uniform(0.5, 5, n_samples),
        'ROE': np.random.uniform(-10, 30, n_samples),
        'Debt_to_Equity': np.random.uniform(0, 2, n_samples),

        # Market indicators
        'Market_Sentiment': np.random.uniform(-1, 1, n_samples),
        'Relative_Strength': np.random.uniform(0.5, 1.5, n_samples),
    })

    return features


def create_sample_labels(features: pd.DataFrame) -> np.ndarray:
    """
    Create sample labels based on features (simplified logic for demonstration).

    In practice, labels would be created from actual forward returns.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature DataFrame

    Returns:
    --------
    np.ndarray
        Labels (0-4)
    """
    # Simple logic: combine multiple indicators to create signals
    scores = (
        (features['RSI_14'] - 50) / 30 +  # RSI contribution
        features['MACD'] / 5 +  # MACD contribution
        (features['ADX_14'] - 25) / 20 +  # ADX contribution
        features['ROC_12'] / 10 +  # ROC contribution
        features['Market_Sentiment']  # Sentiment contribution
    )

    # Convert scores to labels
    labels = np.zeros(len(scores), dtype=int)
    labels[scores >= 1.5] = 4  # STRONG_BUY
    labels[(scores >= 0.5) & (scores < 1.5)] = 3  # BUY
    labels[(scores > -0.5) & (scores < 0.5)] = 2  # HOLD
    labels[(scores <= -0.5) & (scores > -1.5)] = 1  # SELL
    labels[scores <= -1.5] = 0  # STRONG_SELL

    return labels


def main():
    """Main example function."""
    print("=" * 70)
    print("Random Forest Trading Signal Classifier - Example")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Generate Sample Data
    # -------------------------------------------------------------------------
    print("\n1. Generating sample data...")
    n_samples = 2000
    features = create_sample_features(n_samples=n_samples)
    labels = create_sample_labels(features)

    print(f"   Generated {n_samples} samples with {len(features.columns)} features")
    print(f"   Features: {list(features.columns[:5])} ... (and {len(features.columns)-5} more)")

    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\n   Class distribution:")
    signal_names = ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']
    for label, count in zip(unique, counts):
        print(f"     {signal_names[label]}: {count} ({count/len(labels)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 2. Split Data
    # -------------------------------------------------------------------------
    print("\n2. Splitting data into train/test sets...")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 3. Train Classifier with Class Balancing
    # -------------------------------------------------------------------------
    print("\n3. Training Random Forest classifier...")
    print("   Configuration:")
    print("     - Trees: 100")
    print("     - Class balancing: balanced weights")
    print("     - Out-of-bag scoring: enabled")
    print("     - SMOTE: disabled (can be enabled with use_smote=True)")

    classifier = TradingSignalClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        oob_score=True,
        class_balance_method='balanced',
        use_smote=False,  # Set to True to use SMOTE (requires imbalanced-learn)
        random_state=42,
        verbose=1
    )

    classifier.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 4. Make Predictions
    # -------------------------------------------------------------------------
    print("\n4. Making predictions on test set...")

    # Predict numeric labels
    predictions = classifier.predict(X_test)

    # Predict signal names
    signal_predictions = classifier.predict_signals(X_test, as_string=True)

    # Get prediction probabilities
    probabilities = classifier.predict_proba(X_test)

    print(f"   Sample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        true_signal = signal_names[y_test.iloc[i]] if hasattr(y_test, 'iloc') else signal_names[y_test[i]]
        pred_signal = signal_predictions[i]
        confidence = probabilities[i].max() * 100
        print(f"     True: {true_signal:12s} | Predicted: {pred_signal:12s} | Confidence: {confidence:.1f}%")

    # -------------------------------------------------------------------------
    # 5. Evaluate Model Performance
    # -------------------------------------------------------------------------
    print("\n5. Evaluating model performance...")

    metrics = classifier.evaluate(X_test, y_test, detailed=True)

    print(f"\n   Overall Metrics:")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
    print(f"     Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"     Weighted F1-Score: {metrics['f1_weighted']:.4f}")

    if classifier.oob_score_:
        print(f"     Out-of-Bag Score: {metrics['oob_score']:.4f}")

    print(f"\n   Per-Class Metrics:")
    print(f"     {'Signal':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("     " + "-" * 60)
    for signal in signal_names:
        if f'{signal}_precision' in metrics:
            precision = metrics[f'{signal}_precision']
            recall = metrics[f'{signal}_recall']
            f1 = metrics[f'{signal}_f1']
            support = int(metrics[f'{signal}_support'])
            print(f"     {signal:<15} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")

    print("\n   Classification Report:")
    print(metrics['classification_report'])

    # -------------------------------------------------------------------------
    # 6. Analyze Feature Importance
    # -------------------------------------------------------------------------
    print("\n6. Analyzing feature importance...")

    top_features = classifier.get_feature_importance(top_n=10)

    print("   Top 10 Most Important Features:")
    print(f"     {'Feature':<25} {'Importance':>12}")
    print("     " + "-" * 40)
    for feature, importance in top_features.items():
        print(f"     {feature:<25} {importance:12.4f}")

    # -------------------------------------------------------------------------
    # 7. Cross-Validation
    # -------------------------------------------------------------------------
    print("\n7. Performing 5-fold cross-validation...")

    cv_results = classifier.cross_validate(X_train, y_train, cv=5, scoring='accuracy')

    print(f"   Cross-validation scores: {cv_results['scores']}")
    print(f"   Mean CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    print(f"   Min CV Score: {cv_results['min_score']:.4f}")
    print(f"   Max CV Score: {cv_results['max_score']:.4f}")

    # -------------------------------------------------------------------------
    # 8. Save and Load Model
    # -------------------------------------------------------------------------
    print("\n8. Saving and loading model...")

    model_path = Path(__file__).parent.parent / "models" / "saved_models" / "trading_signal_rf.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    classifier.save(model_path)
    print(f"   Model saved to: {model_path}")

    # Load model
    loaded_classifier = TradingSignalClassifier.load(model_path)
    print(f"   Model loaded successfully")

    # Verify loaded model
    loaded_predictions = loaded_classifier.predict(X_test[:5])
    original_predictions = predictions[:5]
    matches = np.array_equal(loaded_predictions, original_predictions)
    print(f"   Predictions match: {matches}")

    # -------------------------------------------------------------------------
    # 9. Model Information
    # -------------------------------------------------------------------------
    print("\n9. Model information...")

    model_info = classifier.get_model_info()

    print(f"   Model Status: {'Fitted' if model_info['is_fitted'] else 'Not Fitted'}")
    print(f"   Training Date: {model_info['training_date']}")
    print(f"   Training Samples: {model_info['training_samples']}")
    print(f"   Number of Features: {model_info['n_features']}")
    print(f"   Number of Trees: {model_info['n_estimators']}")
    print(f"   Max Depth: {model_info['max_depth']}")
    print(f"   Class Balance Method: {model_info['class_balance_method']}")
    print(f"   OOB Score: {model_info['oob_score']:.4f if model_info['oob_score'] else 'N/A'}")

    # -------------------------------------------------------------------------
    # 10. Grid Search (Optional - commented out as it takes time)
    # -------------------------------------------------------------------------
    print("\n10. Grid search for hyperparameter tuning (skipped in this example)")
    print("    Uncomment the code below to perform grid search:")
    print("""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid_results = classifier.grid_search(
        X_train, y_train,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy'
    )

    print("Best parameters:", grid_results['best_params'])
    print("Best score:", grid_results['best_score'])
    """)

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
