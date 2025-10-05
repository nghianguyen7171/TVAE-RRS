"""
XGBoost Baseline Model for TVAE-RRS
Implements XGBoost classifier for time series classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
import xgboost as xgb
import joblib
import os


class XGBoostBaseline:
    """
    XGBoost baseline model for time series classification
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize XGBoost baseline model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            random_state: Random state
            n_jobs: Number of parallel jobs
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def _reshape_data(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape time series data for XGBoost
        
        Args:
            X: Input data with shape (n_samples, sequence_length, n_features)
            
        Returns:
            Reshaped data with shape (n_samples, sequence_length * n_features)
        """
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: int = 10,
            verbose: bool = True) -> 'XGBoostBaseline':
        """
        Fit XGBoost model
        
        Args:
            X: Training data
            y: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            early_stopping_rounds: Early stopping rounds
            verbose: Verbose output
            
        Returns:
            Self
        """
        # Reshape data
        X_reshaped = self._reshape_data(X)
        
        # Store feature names
        if len(X.shape) == 3:
            seq_len, n_features = X.shape[1], X.shape[2]
            self.feature_names = [f'feature_{i}' for i in range(seq_len * n_features)]
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_reshaped.shape[1])]
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Fit model
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_reshaped = self._reshape_data(X_val)
            
            self.model.fit(
                X_reshaped, y,
                eval_set=[(X_val_reshaped, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
        else:
            self.model.fit(X_reshaped, y, verbose=verbose)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_reshaped = self._reshape_data(X)
        return self.model.predict(X_reshaped)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_reshaped = self._reshape_data(X)
        return self.model.predict_proba(X_reshaped)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predict probabilities
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'auroc': roc_auc_score(y, y_pred_proba[:, 1]),
            'auprc': average_precision_score(y, y_pred_proba[:, 1]),
            'f1': f1_score(y, y_pred),
            'kappa': cohen_kappa_score(y, y_pred),
        }
        
        return metrics
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray,
                             param_grid: Optional[Dict[str, List]] = None,
                             cv: int = 5,
                             scoring: str = 'roc_auc',
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Training data
            y: Training labels
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and results
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
            }
        
        # Reshape data
        X_reshaped = self._reshape_data(X)
        
        # Create base model
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_reshaped, y)
        
        # Update model with best parameters
        self.n_estimators = grid_search.best_params_['n_estimators']
        self.max_depth = grid_search.best_params_['max_depth']
        self.learning_rate = grid_search.best_params_['learning_rate']
        self.subsample = grid_search.best_params_['subsample']
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> 'XGBoostBaseline':
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        return self
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dictionary of feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


def build_xgbm_baseline(**kwargs) -> XGBoostBaseline:
    """
    Build XGBoost baseline model
    
    Args:
        **kwargs: Additional arguments for XGBoostBaseline
        
    Returns:
        XGBoost baseline model
    """
    return XGBoostBaseline(**kwargs)
