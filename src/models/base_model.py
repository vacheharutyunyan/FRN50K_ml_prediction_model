# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
import joblib
import logging
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
logger = logging.getLogger(__name__)

class BaseForecastingModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    Teaching purpose: This demonstrates good software engineering practices
    - inheritance, abstraction, and consistent interfaces across different models.
    
    All models in our framework will inherit from this class, ensuring
    consistent behavior and making it easy to swap between different algorithms.
    """
    
    def __init__(self, model_name: str, config: Dict = None):
        """
        Initialize the base model with common attributes.
        
        Args:
            model_name: Human-readable name for this model
            config: Dictionary of model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_fitted = False
        self.feature_names = None
        self.model = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseForecastingModel':
        """
        Fit the model to training data.
        
        This method must be implemented by all subclasses.
        It should set self.is_fitted = True upon successful completion.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target variable (pandas Series)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining (allows model.fit().predict() syntax)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions for given features.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Validate input data for common issues.
        
        Teaching point: Always validate your inputs! This prevents mysterious
        errors later and makes debugging much easier.
        """
        # Check for completely empty data
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Check for infinite values
        # if np.isinf(X.select_dtypes(include=[np.number]).values).any():
        #     logger.warning("Infinite values detected in features")
            
        # Check for extremely high missing value rates
        missing_rates = X.isnull().mean()
        high_missing = missing_rates[missing_rates > 0.5]
        if not high_missing.empty:
            logger.warning(f"Features with >50% missing values: {high_missing.index.tolist()}")
        
        # Validate target if provided
        if y is not None:
            if len(X) != len(y):
                raise ValueError(f"Feature matrix ({len(X)} rows) and target ({len(y)} rows) have different lengths")
            
            if y.isnull().sum() > 0:
                logger.warning(f"Target variable has {y.isnull().sum()} missing values")
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk for later use."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire model object
        joblib.dump(self, filepath)
        logger.info(f"Model {self.model_name} saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseForecastingModel':
        """Load a previously saved model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if the underlying model supports it.
        
        Teaching insight: Not all models provide feature importance,
        but when they do, it's incredibly valuable for understanding
        what drives your predictions.
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None
            
        # Check for sklearn-style feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        # Check for linear model coefficients
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            logger.info(f"Feature importance not available for {self.model_name}")
            return None
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return dict(zip([f"feature_{i}" for i in range(len(importances))], importances))
    
    def get_model_summary(self) -> Dict:
        """Get a summary of the model's key characteristics."""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names) if self.feature_names else None,
            'config': self.config,
            'training_history': self.training_history
        }

class SimpleNaiveModel(BaseForecastingModel):

    def __init__(self):
        super().__init__(model_name="SimpleNaive")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        if y.dropna().empty:
            self.last_value = 0.0
        else:
            self.last_value = y.dropna().iloc[-1]

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        horizon = len(X)
        return np.full(horizon, self.last_value)



class SeasonalNaiveModel(BaseForecastingModel):

    SEASONAL_LAG = 168

    def __init__(self):
        super().__init__(model_name="SeasonalNaive")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.series = y.reset_index(drop=True)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        horizon = len(X)
        preds = []
        n = len(self.series)

        for i in range(horizon):
            idx = n - self.SEASONAL_LAG + i
            if idx < 0 or idx >= n or pd.isna(self.series.iloc[idx]):
                preds.append(self.series.dropna().iloc[-1] if not self.series.dropna().empty else 0.0)
            else:
                preds.append(self.series.iloc[idx])

        return np.array(preds)



class WeightedMovingAverageModel(BaseForecastingModel):

    def __init__(self, window: int):
        super().__init__(
            model_name=f"WeightedMovingAverage_{window}",
            config={"window": window}
        )
        self.window = window

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.series = y.dropna().values
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        horizon = len(X)
        preds = []

        for _ in range(horizon):
            if len(self.series) == 0:
                preds.append(0.0)
            elif len(self.series) < self.window:
                preds.append(np.mean(self.series))
            else:
                window_data = self.series[-self.window:]
                weights = np.arange(1, self.window + 1)
                preds.append(np.average(window_data, weights=weights))

        return np.array(preds)



class LinearRegressionBaseline(BaseForecastingModel):
    def __init__(self, config: Dict = None):
        super().__init__("LinearRegressionBaseline", config)

        self.lag_hours = self.config.get("lag_hours", [1, 7, 168])
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.pipeline = None

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "dt" not in df.columns:
            raise ValueError("Column 'dt' is required for temporal features")

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        df["month"] = df["dt"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _create_lag_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()

        for lag in self.lag_hours:
            df[f"lag_{lag}"] = y.shift(lag)

        return df

    def _prepare_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        X_feat = self._create_temporal_features(X)

        if y is not None:
            X_feat = self._create_lag_features(X_feat, y)

        drop_cols = ["dt", "hour", "day_of_week", "month"]
        X_feat = X_feat.drop(columns=[c for c in drop_cols if c in X_feat.columns])

        return X_feat

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.validate_input(X, y)

        if y.dropna().empty:
            raise ValueError("Target variable contains only NaNs")

        X_feat = self._prepare_features(X, y)

        valid_idx = X_feat.dropna().index
        X_feat = X_feat.loc[valid_idx]
        y = y.loc[valid_idx]

        if X_feat.empty:
            raise ValueError("No valid rows after feature engineering")

        self.feature_names = X_feat.columns.tolist()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("model", self.model),
            ]
        )

        self.pipeline.fit(X_feat, y)

        self.is_fitted = True
        self.training_history["n_samples"] = len(X_feat)

        logger.info("LinearRegressionBaseline fitted successfully")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.validate_input(X)

        X_feat = self._prepare_features(X)

        for col in self.feature_names:
            if col not in X_feat.columns:
                X_feat[col] = 0.0

        X_feat = X_feat[self.feature_names].fillna(0.0)

        preds = self.pipeline.predict(X_feat)

        preds = np.maximum(preds, 0.0)

        return preds


class RandomForestRegressionBaseline(BaseForecastingModel):
    """
    Random Forest regression baseline with temporal feature engineering.
    """

    def __init__(self, config: Dict = None):
        super().__init__("RandomForestRegressionBaseline", config)

        self.lag_hours = self.config.get("lag_hours", [1, 7, 168])
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", None)
        self.random_state = self.config.get("random_state", 42)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.pipeline = None



    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "dt" not in df.columns:
            raise ValueError("Column 'dt' is required")

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        df["month"] = df["dt"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _create_lag_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lag_hours:
            df[f"lag_{lag}"] = y.shift(lag)
        return df

    def _prepare_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        X_feat = self._create_temporal_features(X)

        if y is not None:
            X_feat = self._create_lag_features(X_feat, y)

        drop_cols = ["dt", "hour", "day_of_week", "month"]
        X_feat = X_feat.drop(columns=[c for c in drop_cols if c in X_feat.columns])

        return X_feat

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.validate_input(X, y)

        if y.dropna().empty:
            raise ValueError("Target variable contains only NaNs")

        X_feat = self._prepare_features(X, y)

        valid_idx = X_feat.dropna().index
        X_feat = X_feat.loc[valid_idx]
        y = y.loc[valid_idx]

        if X_feat.empty:
            raise ValueError("No valid rows after feature engineering")

        self.feature_names = X_feat.columns.tolist()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", self.model),
            ]
        )

        self.pipeline.fit(X_feat, y)

        self.is_fitted = True
        self.training_history["n_samples"] = len(X_feat)

        logger.info("RandomForestRegressionBaseline fitted successfully")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.validate_input(X)

        X_feat = self._prepare_features(X)

        for lag in self.lag_hours:
            col = f"lag_{lag}"
            if col not in X_feat.columns:
                X_feat[col] = 0.0

        X_feat = X_feat[self.feature_names].fillna(0.0)

        preds = self.pipeline.predict(X_feat)

        return np.maximum(preds, 0.0)



class XGBoostRegressionBaseline(BaseForecastingModel):
    """
    XGBoost regression baseline with temporal feature engineering.
    """

    def __init__(self, config: Dict = None):
        super().__init__("XGBoostRegressionBaseline", config)

        self.lag_hours = self.config.get("lag_hours", [1, 7, 168])

        self.model = XGBRegressor(
            n_estimators=self.config.get("n_estimators", 300),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.05),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            objective="reg:squarederror",
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
        )

        self.pipeline = None

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "dt" not in df.columns:
            raise ValueError("Column 'dt' is required")

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        df["month"] = df["dt"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _create_lag_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lag_hours:
            df[f"lag_{lag}"] = y.shift(lag)
        return df

    def _prepare_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        X_feat = self._create_temporal_features(X)

        if y is not None:
            X_feat = self._create_lag_features(X_feat, y)

        drop_cols = ["dt", "hour", "day_of_week", "month"]
        X_feat = X_feat.drop(columns=[c for c in drop_cols if c in X_feat.columns])

        return X_feat

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.validate_input(X, y)

        if y.dropna().empty:
            raise ValueError("Target variable contains only NaNs")

        X_feat = self._prepare_features(X, y)

        valid_idx = X_feat.dropna().index
        X_feat = X_feat.loc[valid_idx]
        y = y.loc[valid_idx]

        if X_feat.empty:
            raise ValueError("No valid rows after feature engineering")

        self.feature_names = X_feat.columns.tolist()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", self.model),
            ]
        )

        self.pipeline.fit(X_feat, y)

        self.is_fitted = True
        self.training_history["n_samples"] = len(X_feat)

        logger.info("XGBoostRegressionBaseline fitted successfully")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.validate_input(X)

        X_feat = self._prepare_features(X)

        for lag in self.lag_hours:
            col = f"lag_{lag}"
            if col not in X_feat.columns:
                X_feat[col] = 0.0

        X_feat = X_feat[self.feature_names].fillna(0.0)

        preds = self.pipeline.predict(X_feat)

        return np.maximum(preds, 0.0)


class EnsembleRegressionBaseline(BaseForecastingModel):
    """
    Ensemble regression baseline using VotingRegressor.
    Combines Linear Regression, Random Forest, and XGBoost.
    """

    def __init__(self, config: Dict = None):
        super().__init__("EnsembleRegressionBaseline", config)

        self.lag_hours = self.config.get("lag_hours", [1, 7, 168])

        self.base_models = [
            ("lr", LinearRegression()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=self.config.get("rf_n_estimators", 200),
                    max_depth=self.config.get("rf_max_depth", None),
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=self.config.get("xgb_n_estimators", 300),
                    max_depth=self.config.get("xgb_max_depth", 6),
                    learning_rate=self.config.get("xgb_learning_rate", 0.05),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]

        self.model = VotingRegressor(estimators=self.base_models, n_jobs=-1)
        self.pipeline = None

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "dt" not in df.columns:
            raise ValueError("Column 'dt' is required")

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek
        df["month"] = df["dt"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _create_lag_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lag_hours:
            df[f"lag_{lag}"] = y.shift(lag)
        return df

    def _prepare_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        X_feat = self._create_temporal_features(X)

        if y is not None:
            X_feat = self._create_lag_features(X_feat, y)

        drop_cols = ["dt", "hour", "day_of_week", "month"]
        X_feat = X_feat.drop(columns=[c for c in drop_cols if c in X_feat.columns])

        return X_feat

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.validate_input(X, y)

        if y.dropna().empty:
            raise ValueError("Target variable contains only NaNs")

        X_feat = self._prepare_features(X, y)

        valid_idx = X_feat.dropna().index
        X_feat = X_feat.loc[valid_idx]
        y = y.loc[valid_idx]

        if X_feat.empty:
            raise ValueError("No valid rows after feature engineering")

        self.feature_names = X_feat.columns.tolist()

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", self.model),
            ]
        )

        self.pipeline.fit(X_feat, y)

        self.is_fitted = True
        self.training_history["n_samples"] = len(X_feat)

        logger.info("EnsembleRegressionBaseline fitted successfully")

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.validate_input(X)

        X_feat = self._prepare_features(X)

        for lag in self.lag_hours:
            col = f"lag_{lag}"
            if col not in X_feat.columns:
                X_feat[col] = 0.0

        X_feat = X_feat[self.feature_names].fillna(0.0)

        preds = self.pipeline.predict(X_feat)

        return np.maximum(preds, 0.0)