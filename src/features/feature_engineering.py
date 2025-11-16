"""
Feature Engineering Module - BIST AI Trading System

This module provides comprehensive feature engineering capabilities to combine
technical, fundamental, and whale features into a unified feature matrix.
It includes:

- Feature scaling (Standard, MinMax, Robust)
- Categorical encoding (One-hot, Label, Target)
- Lag features for time series
- Feature interactions
- Missing value handling
- Feature selection utilities
- Pipeline construction

Author: BIST AI Trading System
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings

# Scikit-learn imports
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, QuantileTransformer,
    PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """Enumeration of available scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    NONE = "none"


class EncodingMethod(Enum):
    """Enumeration of available encoding methods"""
    ONEHOT = "onehot"
    LABEL = "label"
    TARGET = "target"
    ORDINAL = "ordinal"


class ImputationMethod(Enum):
    """Enumeration of available imputation methods"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    KNN = "knn"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline"""

    # Scaling configuration
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scale_target: bool = False

    # Imputation configuration
    imputation_method: ImputationMethod = ImputationMethod.FORWARD_FILL
    imputation_constant: float = 0.0
    knn_neighbors: int = 5

    # Lag features configuration
    create_lags: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    lag_features: Optional[List[str]] = None

    # Rolling features configuration
    create_rolling: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    rolling_features: Optional[List[str]] = None

    # Difference features
    create_differences: bool = True
    diff_periods: List[int] = field(default_factory=lambda: [1, 5, 20])

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
    categorical_features: List[str] = field(default_factory=list)

    # Other options
    drop_na: bool = False
    fill_na_after_lag: bool = True
    preserve_timestamps: bool = True


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series data.

    Parameters:
    -----------
    lag_periods : List[int]
        List of lag periods to create
    feature_names : Optional[List[str]]
        Specific features to lag (if None, lag all numeric features)
    fill_na : bool
        Whether to fill NaN values after creating lags
    """

    def __init__(
        self,
        lag_periods: List[int] = [1, 2, 3, 5],
        feature_names: Optional[List[str]] = None,
        fill_na: bool = True
    ):
        self.lag_periods = lag_periods
        self.feature_names = feature_names
        self.fill_na = fill_na
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (learns feature names)"""
        if self.feature_names is None:
            # Select numeric columns only
            self.feature_names_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_names_ = self.feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        X_lagged = X.copy()

        for feature in self.feature_names_:
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in DataFrame")
                continue

            for lag in self.lag_periods:
                lag_col_name = f"{feature}_lag_{lag}"
                X_lagged[lag_col_name] = X[feature].shift(lag)

        if self.fill_na:
            # Forward fill first, then backward fill any remaining
            X_lagged = X_lagged.fillna(method='ffill').fillna(method='bfill')

        return X_lagged

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names"""
        if input_features is None:
            input_features = self.feature_names_

        output_names = list(input_features)
        for feature in self.feature_names_:
            for lag in self.lag_periods:
                output_names.append(f"{feature}_lag_{lag}")
        return output_names


class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create rolling window features (mean, std, min, max).

    Parameters:
    -----------
    windows : List[int]
        List of window sizes
    feature_names : Optional[List[str]]
        Specific features to apply rolling windows to
    statistics : List[str]
        Statistics to compute ('mean', 'std', 'min', 'max', 'median')
    """

    def __init__(
        self,
        windows: List[int] = [5, 10, 20],
        feature_names: Optional[List[str]] = None,
        statistics: List[str] = ['mean', 'std']
    ):
        self.windows = windows
        self.feature_names = feature_names
        self.statistics = statistics
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer"""
        if self.feature_names is None:
            self.feature_names_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_names_ = self.feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        X_rolling = X.copy()

        for feature in self.feature_names_:
            if feature not in X.columns:
                logger.warning(f"Feature {feature} not found in DataFrame")
                continue

            for window in self.windows:
                for stat in self.statistics:
                    col_name = f"{feature}_rolling_{stat}_{window}"

                    if stat == 'mean':
                        X_rolling[col_name] = X[feature].rolling(window=window).mean()
                    elif stat == 'std':
                        X_rolling[col_name] = X[feature].rolling(window=window).std()
                    elif stat == 'min':
                        X_rolling[col_name] = X[feature].rolling(window=window).min()
                    elif stat == 'max':
                        X_rolling[col_name] = X[feature].rolling(window=window).max()
                    elif stat == 'median':
                        X_rolling[col_name] = X[feature].rolling(window=window).median()
                    elif stat == 'sum':
                        X_rolling[col_name] = X[feature].rolling(window=window).sum()

        # Forward fill NaN values created by rolling windows
        X_rolling = X_rolling.fillna(method='ffill').fillna(method='bfill')

        return X_rolling


class DifferenceFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create difference features for time series data.

    Parameters:
    -----------
    periods : List[int]
        List of periods for differencing
    feature_names : Optional[List[str]]
        Specific features to difference
    """

    def __init__(
        self,
        periods: List[int] = [1],
        feature_names: Optional[List[str]] = None
    ):
        self.periods = periods
        self.feature_names = feature_names
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer"""
        if self.feature_names is None:
            self.feature_names_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_names_ = self.feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create difference features"""
        X_diff = X.copy()

        for feature in self.feature_names_:
            if feature not in X.columns:
                continue

            for period in self.periods:
                diff_col_name = f"{feature}_diff_{period}"
                X_diff[diff_col_name] = X[feature].diff(periods=period)

                # Also create percentage change
                pct_col_name = f"{feature}_pct_change_{period}"
                X_diff[pct_col_name] = X[feature].pct_change(periods=period)

        # Fill NaN values
        X_diff = X_diff.fillna(method='bfill').fillna(0)

        return X_diff


class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Create feature interactions (products, ratios, etc.).

    Parameters:
    -----------
    interaction_pairs : List[Tuple[str, str]]
        List of feature pairs to create interactions for
    interaction_types : List[str]
        Types of interactions ('multiply', 'divide', 'add', 'subtract')
    """

    def __init__(
        self,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        interaction_types: List[str] = ['multiply', 'divide']
    ):
        self.interaction_pairs = interaction_pairs
        self.interaction_types = interaction_types

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        if self.interaction_pairs is None:
            return X

        X_interact = X.copy()

        for feat1, feat2 in self.interaction_pairs:
            if feat1 not in X.columns or feat2 not in X.columns:
                logger.warning(f"Features {feat1} or {feat2} not found")
                continue

            if 'multiply' in self.interaction_types:
                X_interact[f"{feat1}_x_{feat2}"] = X[feat1] * X[feat2]

            if 'divide' in self.interaction_types:
                # Avoid division by zero
                X_interact[f"{feat1}_div_{feat2}"] = X[feat1] / (X[feat2].replace(0, np.nan))

            if 'add' in self.interaction_types:
                X_interact[f"{feat1}_plus_{feat2}"] = X[feat1] + X[feat2]

            if 'subtract' in self.interaction_types:
                X_interact[f"{feat1}_minus_{feat2}"] = X[feat1] - X[feat2]

        # Fill any NaN created by interactions
        X_interact = X_interact.fillna(method='ffill').fillna(0)

        return X_interact


class FeatureEngineer:
    """
    Main feature engineering class that combines all transformations.

    This class orchestrates the entire feature engineering pipeline including:
    - Loading features from different sources
    - Combining technical, fundamental, and whale features
    - Applying transformations (scaling, encoding, lag features)
    - Feature selection
    - Creating unified feature matrix
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the FeatureEngineer.

        Parameters:
        -----------
        config : Optional[FeatureConfig]
            Configuration object for feature engineering
        """
        self.config = config or FeatureConfig()
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.feature_selector = None
        self.feature_names_ = None
        self.categorical_encoders_ = {}
        self.is_fitted_ = False

        logger.info("FeatureEngineer initialized with config")

    def _get_scaler(self):
        """Get the appropriate scaler based on configuration"""
        if self.config.scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif self.config.scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler()
        elif self.config.scaling_method == ScalingMethod.ROBUST:
            return RobustScaler()
        elif self.config.scaling_method == ScalingMethod.QUANTILE:
            return QuantileTransformer()
        elif self.config.scaling_method == ScalingMethod.POWER:
            return PowerTransformer()
        else:
            return None

    def _get_imputer(self):
        """Get the appropriate imputer based on configuration"""
        method = self.config.imputation_method

        if method == ImputationMethod.KNN:
            return KNNImputer(n_neighbors=self.config.knn_neighbors)
        elif method == ImputationMethod.MEAN:
            return SimpleImputer(strategy='mean')
        elif method == ImputationMethod.MEDIAN:
            return SimpleImputer(strategy='median')
        elif method == ImputationMethod.MODE:
            return SimpleImputer(strategy='most_frequent')
        elif method == ImputationMethod.CONSTANT:
            return SimpleImputer(strategy='constant', fill_value=self.config.imputation_constant)
        else:
            return None

    def create_lag_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with time series data
        feature_cols : Optional[List[str]]
            Columns to create lags for (if None, use all numeric columns)

        Returns:
        --------
        pd.DataFrame
            DataFrame with added lag features
        """
        if not self.config.create_lags:
            return df

        logger.info("Creating lag features...")

        feature_cols = feature_cols or self.config.lag_features
        transformer = LagFeatureTransformer(
            lag_periods=self.config.lag_periods,
            feature_names=feature_cols,
            fill_na=self.config.fill_na_after_lag
        )

        transformer.fit(df)
        df_lagged = transformer.transform(df)

        logger.info(f"Created {len(df_lagged.columns) - len(df.columns)} lag features")
        return df_lagged

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : Optional[List[str]]
            Columns to create rolling features for

        Returns:
        --------
        pd.DataFrame
            DataFrame with added rolling features
        """
        if not self.config.create_rolling:
            return df

        logger.info("Creating rolling window features...")

        feature_cols = feature_cols or self.config.rolling_features
        transformer = RollingFeatureTransformer(
            windows=self.config.rolling_windows,
            feature_names=feature_cols,
            statistics=['mean', 'std', 'min', 'max']
        )

        transformer.fit(df)
        df_rolling = transformer.transform(df)

        logger.info(f"Created {len(df_rolling.columns) - len(df.columns)} rolling features")
        return df_rolling

    def create_difference_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create difference and percentage change features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : Optional[List[str]]
            Columns to create differences for

        Returns:
        --------
        pd.DataFrame
            DataFrame with added difference features
        """
        if not self.config.create_differences:
            return df

        logger.info("Creating difference features...")

        transformer = DifferenceFeatureTransformer(
            periods=self.config.diff_periods,
            feature_names=feature_cols
        )

        transformer.fit(df)
        df_diff = transformer.transform(df)

        logger.info(f"Created {len(df_diff.columns) - len(df.columns)} difference features")
        return df_diff

    def create_interaction_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create feature interactions.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            DataFrame with added interaction features
        """
        if not self.config.create_interactions:
            return df

        logger.info("Creating interaction features...")

        transformer = FeatureInteractionTransformer(
            interaction_pairs=self.config.interaction_pairs
        )

        df_interact = transformer.transform(df)

        logger.info(f"Created {len(df_interact.columns) - len(df.columns)} interaction features")
        return df_interact

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the encoder or use existing one

        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded categorical features
        """
        if not self.config.categorical_features:
            return df

        logger.info("Encoding categorical features...")
        df_encoded = df.copy()

        for cat_feature in self.config.categorical_features:
            if cat_feature not in df.columns:
                logger.warning(f"Categorical feature {cat_feature} not found")
                continue

            if self.config.encoding_method == EncodingMethod.LABEL:
                if fit or cat_feature not in self.categorical_encoders_:
                    encoder = LabelEncoder()
                    df_encoded[cat_feature] = encoder.fit_transform(df[cat_feature].astype(str))
                    self.categorical_encoders_[cat_feature] = encoder
                else:
                    encoder = self.categorical_encoders_[cat_feature]
                    df_encoded[cat_feature] = encoder.transform(df[cat_feature].astype(str))

            elif self.config.encoding_method == EncodingMethod.ONEHOT:
                if fit or cat_feature not in self.categorical_encoders_:
                    dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, drop_first=True)
                    self.categorical_encoders_[cat_feature] = dummies.columns.tolist()
                else:
                    dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, drop_first=True)

                df_encoded = pd.concat([df_encoded.drop(columns=[cat_feature]), dummies], axis=1)

        return df_encoded

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the imputer

        Returns:
        --------
        pd.DataFrame
            DataFrame with imputed values
        """
        logger.info("Handling missing values...")

        method = self.config.imputation_method

        # Time series specific methods
        if method == ImputationMethod.FORWARD_FILL:
            return df.fillna(method='ffill').fillna(method='bfill')
        elif method == ImputationMethod.BACKWARD_FILL:
            return df.fillna(method='bfill').fillna(method='ffill')
        elif method == ImputationMethod.INTERPOLATE:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
            return df

        # Sklearn-based imputation
        if fit:
            self.imputer = self._get_imputer()

        if self.imputer is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_imputed = df.copy()

            if fit:
                df_imputed[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            else:
                df_imputed[numeric_cols] = self.imputer.transform(df[numeric_cols])

            return df_imputed

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the scaler
        exclude_cols : Optional[List[str]]
            Columns to exclude from scaling

        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        if self.config.scaling_method == ScalingMethod.NONE:
            return df

        logger.info(f"Scaling features using {self.config.scaling_method.value}...")

        if fit:
            self.scaler = self._get_scaler()

        if self.scaler is None:
            return df

        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude specified columns
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if fit:
            df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df_scaled[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df_scaled

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Perform feature selection.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        fit : bool
            Whether to fit the selector

        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features
        """
        if not self.config.feature_selection:
            return X

        logger.info("Performing feature selection...")

        if fit:
            if self.config.selection_method == 'kbest':
                n_features = self.config.n_features or int(len(X.columns) * 0.5)
                self.feature_selector = SelectKBest(f_regression, k=min(n_features, len(X.columns)))
            elif self.config.selection_method == 'percentile':
                self.feature_selector = SelectPercentile(f_regression, percentile=self.config.selection_percentile)

            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            self.feature_names_ = selected_features
        else:
            X_selected = self.feature_selector.transform(X)
            selected_features = self.feature_names_

        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def combine_features(
        self,
        technical_features: Optional[pd.DataFrame] = None,
        fundamental_features: Optional[pd.DataFrame] = None,
        whale_features: Optional[pd.DataFrame] = None,
        other_features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Combine features from different sources into a unified feature matrix.

        Parameters:
        -----------
        technical_features : Optional[pd.DataFrame]
            Technical indicator features
        fundamental_features : Optional[pd.DataFrame]
            Fundamental analysis features
        whale_features : Optional[pd.DataFrame]
            Whale activity features
        other_features : Optional[Dict[str, pd.DataFrame]]
            Additional feature dataframes

        Returns:
        --------
        pd.DataFrame
            Combined feature matrix
        """
        logger.info("Combining features from multiple sources...")

        dataframes = []

        if technical_features is not None:
            logger.info(f"Adding {len(technical_features.columns)} technical features")
            # Add prefix to avoid column name conflicts
            tech_df = technical_features.copy()
            tech_df.columns = [f"tech_{col}" if not col.startswith('tech_') else col
                              for col in tech_df.columns]
            dataframes.append(tech_df)

        if fundamental_features is not None:
            logger.info(f"Adding {len(fundamental_features.columns)} fundamental features")
            fund_df = fundamental_features.copy()
            fund_df.columns = [f"fund_{col}" if not col.startswith('fund_') else col
                              for col in fund_df.columns]
            dataframes.append(fund_df)

        if whale_features is not None:
            logger.info(f"Adding {len(whale_features.columns)} whale features")
            whale_df = whale_features.copy()
            whale_df.columns = [f"whale_{col}" if not col.startswith('whale_') else col
                               for col in whale_df.columns]
            dataframes.append(whale_df)

        if other_features:
            for name, df in other_features.items():
                logger.info(f"Adding {len(df.columns)} features from {name}")
                other_df = df.copy()
                other_df.columns = [f"{name}_{col}" if not col.startswith(f"{name}_") else col
                                   for col in other_df.columns]
                dataframes.append(other_df)

        if not dataframes:
            raise ValueError("No feature dataframes provided")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=1, join='inner')

        logger.info(f"Combined feature matrix shape: {combined_df.shape}")

        return combined_df

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        technical_features: Optional[pd.DataFrame] = None,
        fundamental_features: Optional[pd.DataFrame] = None,
        whale_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the complete feature engineering pipeline.

        Parameters:
        -----------
        X : pd.DataFrame
            Base feature dataframe
        y : Optional[pd.Series]
            Target variable (for feature selection)
        technical_features : Optional[pd.DataFrame]
            Technical features to combine
        fundamental_features : Optional[pd.DataFrame]
            Fundamental features to combine
        whale_features : Optional[pd.DataFrame]
            Whale features to combine

        Returns:
        --------
        pd.DataFrame
            Fully engineered feature matrix
        """
        logger.info("Starting complete feature engineering pipeline (fit_transform)...")

        # Step 1: Combine features from different sources
        if any([technical_features is not None, fundamental_features is not None, whale_features is not None]):
            X = self.combine_features(technical_features, fundamental_features, whale_features)

        # Step 2: Encode categorical features
        X = self.encode_categorical_features(X, fit=True)

        # Step 3: Handle missing values (initial)
        X = self.handle_missing_values(X, fit=True)

        # Step 4: Create lag features
        X = self.create_lag_features(X)

        # Step 5: Create rolling features
        X = self.create_rolling_features(X)

        # Step 6: Create difference features
        X = self.create_difference_features(X)

        # Step 7: Create interaction features
        X = self.create_interaction_features(X)

        # Step 8: Handle missing values again (after feature creation)
        X = self.handle_missing_values(X, fit=False)

        # Step 9: Scale features
        X = self.scale_features(X, fit=True)

        # Step 10: Feature selection (if enabled and target provided)
        if self.config.feature_selection and y is not None:
            X = self.select_features(X, y, fit=True)

        # Step 11: Drop NaN if configured
        if self.config.drop_na:
            X = X.dropna()

        self.is_fitted_ = True
        logger.info(f"Feature engineering complete. Final shape: {X.shape}")

        return X

    def transform(
        self,
        X: pd.DataFrame,
        technical_features: Optional[pd.DataFrame] = None,
        fundamental_features: Optional[pd.DataFrame] = None,
        whale_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.

        Parameters:
        -----------
        X : pd.DataFrame
            Base feature dataframe
        technical_features : Optional[pd.DataFrame]
            Technical features to combine
        fundamental_features : Optional[pd.DataFrame]
            Fundamental features to combine
        whale_features : Optional[pd.DataFrame]
            Whale features to combine

        Returns:
        --------
        pd.DataFrame
            Transformed feature matrix
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted before transform")

        logger.info("Transforming new data using fitted pipeline...")

        # Step 1: Combine features
        if any([technical_features is not None, fundamental_features is not None, whale_features is not None]):
            X = self.combine_features(technical_features, fundamental_features, whale_features)

        # Step 2: Encode categorical features
        X = self.encode_categorical_features(X, fit=False)

        # Step 3: Handle missing values
        X = self.handle_missing_values(X, fit=False)

        # Step 4-7: Create features (these are stateless transformations)
        X = self.create_lag_features(X)
        X = self.create_rolling_features(X)
        X = self.create_difference_features(X)
        X = self.create_interaction_features(X)

        # Step 8: Handle missing values again
        if self.imputer is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self.imputer.transform(X[numeric_cols])

        # Step 9: Scale features
        X = self.scale_features(X, fit=False)

        # Step 10: Feature selection
        if self.config.feature_selection and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X)
            X = pd.DataFrame(X_selected, columns=self.feature_names_, index=X.index)

        # Step 11: Drop NaN if configured
        if self.config.drop_na:
            X = X.dropna()

        logger.info(f"Transform complete. Final shape: {X.shape}")

        return X

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get feature importance from feature selector.

        Parameters:
        -----------
        feature_names : Optional[List[str]]
            Feature names (if None, use stored names)

        Returns:
        --------
        Optional[pd.DataFrame]
            DataFrame with feature importance scores
        """
        if self.feature_selector is None:
            logger.warning("No feature selector fitted")
            return None

        if not hasattr(self.feature_selector, 'scores_'):
            logger.warning("Feature selector does not have scores_")
            return None

        feature_names = feature_names or self.feature_names_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'score': self.feature_selector.scores_[self.feature_selector.get_support()]
        }).sort_values('score', ascending=False)

        return importance_df


# Utility functions

def create_time_features(df: pd.DataFrame, datetime_col: str = 'date') -> pd.DataFrame:
    """
    Create time-based features from datetime column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
    """
    df = df.copy()

    if datetime_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df[datetime_col] = df.index
        else:
            logger.warning(f"Datetime column {datetime_col} not found")
            return df

    # Ensure datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract time features
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['dayofweek'] = df[datetime_col].dt.dayofweek
    df['quarter'] = df[datetime_col].dt.quarter
    df['weekofyear'] = df[datetime_col].dt.isocalendar().week
    df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[datetime_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[datetime_col].dt.is_quarter_end.astype(int)

    # Cyclical encoding for month and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Remove highly correlated features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Correlation threshold for removal
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
    --------
    pd.DataFrame
        DataFrame with correlated features removed
    """
    logger.info(f"Removing features with correlation > {threshold}...")

    # Calculate correlation matrix
    corr_matrix = df.corr(method=method).abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    logger.info(f"Removing {len(to_drop)} highly correlated features")

    return df.drop(columns=to_drop)


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive statistics for all features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Statistics dataframe
    """
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100,
        'unique_count': df.nunique(),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
        'median': df.median(numeric_only=True)
    })

    return stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Feature Engineering Module - BIST AI Trading System")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    sample_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(100)),
        'volume': np.random.randint(1000000, 5000000, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'macd': np.random.randn(100),
        'pe_ratio': np.random.uniform(10, 30, 100),
        'whale_activity': np.random.uniform(0, 1, 100)
    })

    sample_data = sample_data.set_index('date')

    print("\nSample data created:")
    print(sample_data.head())

    # Configure feature engineering
    config = FeatureConfig(
        scaling_method=ScalingMethod.STANDARD,
        create_lags=True,
        lag_periods=[1, 5, 10],
        create_rolling=True,
        rolling_windows=[5, 10],
        create_differences=True,
        diff_periods=[1],
        imputation_method=ImputationMethod.FORWARD_FILL
    )

    # Initialize feature engineer
    engineer = FeatureEngineer(config=config)

    # Create target variable
    y = (sample_data['close'].shift(-1) > sample_data['close']).astype(int)
    y = y[:-1]  # Remove last NaN

    # Fit and transform
    X_transformed = engineer.fit_transform(sample_data[:-1])

    print(f"\nOriginal features: {sample_data.shape[1]}")
    print(f"Engineered features: {X_transformed.shape[1]}")
    print(f"\nFirst few engineered features:")
    print(X_transformed.iloc[:5, :10])

    print("\n" + "=" * 80)
    print("Feature engineering example completed successfully!")
    print("=" * 80)
