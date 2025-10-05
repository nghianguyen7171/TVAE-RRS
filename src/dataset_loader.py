"""
Dataset loader for TVAE-RRS
Handles data loading and preparation for training and evaluation
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings

from src.data_preprocessing import DataPreprocessor, load_data, split_data
from src.utils.window_processing import WindowProcessor


class DatasetLoader:
    """
    Dataset loader class for TVAE-RRS
    """
    
    def __init__(self,
                 data_dir: str = "data",
                 processed_data_dir: str = "data/processed",
                 window_size: int = 16,
                 stride: int = 1,
                 prediction_horizon: int = 1,
                 normalize: bool = True,
                 scaler_type: str = "standard"):
        """
        Initialize DatasetLoader
        
        Args:
            data_dir: Directory containing raw data
            processed_data_dir: Directory for processed data
            window_size: Size of sliding window
            stride: Step size for sliding window
            prediction_horizon: Number of time steps ahead to predict
            normalize: Whether to normalize features
            scaler_type: Type of scaler
        """
        self.data_dir = data_dir
        self.processed_data_dir = processed_data_dir
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        # Initialize components
        self.preprocessor = DataPreprocessor(
            normalize_features=normalize,
            scaler_type=scaler_type
        )
        self.window_processor = WindowProcessor(
            window_size=window_size,
            stride=stride,
            prediction_horizon=prediction_horizon,
            normalize=normalize,
            scaler_type=scaler_type
        )
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_cnuh_data(self, 
                       train_path: str,
                       test_path: Optional[str] = None,
                       validation_path: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load CNUH dataset
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            validation_path: Path to validation data (optional)
            
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Load training data
        train_df = load_data(train_path, "cnuh")
        train_df = self.preprocessor.preprocess_cnuh_data(train_df)
        X_train, y_train, y_train_onehot = self.window_processor.process_cnuh_data(train_df)
        datasets['train'] = (X_train, y_train, y_train_onehot)
        
        # Load test data if provided
        if test_path:
            test_df = load_data(test_path, "cnuh")
            test_df = self.preprocessor.transform_data(test_df)
            X_test, y_test, y_test_onehot = self.window_processor.process_cnuh_data(test_df)
            datasets['test'] = (X_test, y_test, y_test_onehot)
        
        # Load validation data if provided
        if validation_path:
            val_df = load_data(validation_path, "cnuh")
            val_df = self.preprocessor.transform_data(val_df)
            X_val, y_val, y_val_onehot = self.window_processor.process_cnuh_data(val_df)
            datasets['validation'] = (X_val, y_val, y_val_onehot)
        
        # Store feature names
        self.feature_names = self.preprocessor.get_feature_names()
        self.is_loaded = True
        
        return datasets
    
    def load_uv_data(self, 
                      train_path: str,
                      test_path: Optional[str] = None,
                      validation_path: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load UV dataset
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            validation_path: Path to validation data (optional)
            
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Load training data
        train_df = load_data(train_path, "uv")
        train_df = self.preprocessor.preprocess_uv_data(train_df)
        X_train, y_train, y_train_onehot = self.window_processor.process_uv_data(train_df)
        datasets['train'] = (X_train, y_train, y_train_onehot)
        
        # Load test data if provided
        if test_path:
            test_df = load_data(test_path, "uv")
            test_df = self.preprocessor.transform_data(test_df)
            X_test, y_test, y_test_onehot = self.window_processor.process_uv_data(test_df)
            datasets['test'] = (X_test, y_test, y_test_onehot)
        
        # Load validation data if provided
        if validation_path:
            val_df = load_data(validation_path, "uv")
            val_df = self.preprocessor.transform_data(val_df)
            X_val, y_val, y_val_onehot = self.window_processor.process_uv_data(val_df)
            datasets['validation'] = (X_val, y_val, y_val_onehot)
        
        # Store feature names
        self.feature_names = self.preprocessor.get_feature_names()
        self.is_loaded = True
        
        return datasets
    
    def load_data(self, 
                  train_path: str,
                  dataset_type: str = "cnuh",
                  test_path: Optional[str] = None,
                  validation_path: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load data based on dataset type
        
        Args:
            train_path: Path to training data
            dataset_type: Type of dataset ('cnuh' or 'uv')
            test_path: Path to test data (optional)
            validation_path: Path to validation data (optional)
            
        Returns:
            Dictionary containing loaded datasets
        """
        if dataset_type.lower() == "cnuh":
            return self.load_cnuh_data(train_path, test_path, validation_path)
        elif dataset_type.lower() == "uv":
            return self.load_uv_data(train_path, test_path, validation_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def create_train_test_split(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train-test split
        
        Args:
            X: Input features
            y: Target labels
            test_size: Proportion of test set
            random_state: Random state
            stratify: Whether to stratify split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return split_data(X, y, test_size, random_state, stratify)
    
    def create_cross_validation_splits(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray,
                                      cv_folds: int = 5,
                                      cv_strategy: str = "stratified_kfold",
                                      random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits
        
        Args:
            X: Input features
            y: Target labels
            cv_folds: Number of CV folds
            cv_strategy: CV strategy ('stratified_kfold', 'kfold', 'loocv')
            random_state: Random state
            
        Returns:
            List of CV splits
        """
        cv_splits = []
        
        if cv_strategy == "stratified_kfold":
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                cv_splits.append((X_train, X_val, y_train, y_val))
        
        elif cv_strategy == "kfold":
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                cv_splits.append((X_train, X_val, y_train, y_val))
        
        elif cv_strategy == "loocv":
            loo = LeaveOneOut()
            for train_idx, val_idx in loo.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                cv_splits.append((X_train, X_val, y_train, y_val))
        
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        return cv_splits
    
    def save_processed_data(self, 
                           datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                           output_dir: str) -> None:
        """
        Save processed datasets
        
        Args:
            datasets: Dictionary containing datasets
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, (X, y, y_onehot) in datasets.items():
            # Save numpy arrays
            np.savez(
                os.path.join(output_dir, f"{split_name}_data.npz"),
                X=X, y=y, y_onehot=y_onehot
            )
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'stride': self.stride,
            'prediction_horizon': self.prediction_horizon,
            'normalize': self.normalize,
            'scaler_type': self.scaler_type
        }
        
        with open(os.path.join(output_dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_processed_data(self, data_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load processed datasets
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.pickle'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Update attributes
        self.feature_names = metadata['feature_names']
        self.window_size = metadata['window_size']
        self.stride = metadata['stride']
        self.prediction_horizon = metadata['prediction_horizon']
        self.normalize = metadata['normalize']
        self.scaler_type = metadata['scaler_type']
        
        # Load datasets
        for split_name in ['train', 'test', 'validation']:
            data_path = os.path.join(data_dir, f"{split_name}_data.npz")
            if os.path.exists(data_path):
                data = np.load(data_path)
                datasets[split_name] = (data['X'], data['y'], data['y_onehot'])
        
        self.is_loaded = True
        return datasets
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names
    
    def get_data_info(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get data information
        
        Returns:
            Dictionary containing data information
        """
        if not self.is_loaded:
            return {}
        
        info = {
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'stride': self.stride,
            'prediction_horizon': self.prediction_horizon,
            'normalize': self.normalize,
            'scaler_type': self.scaler_type
        }
        
        return info


def create_data_generator(X: np.ndarray, 
                         y: np.ndarray,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create data generator for training
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        random_state: Random state
        
    Returns:
        Generator yielding batches of data
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]


def balance_dataset(X: np.ndarray, 
                    y: np.ndarray,
                    method: str = "undersample",
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset using various methods
    
    Args:
        X: Input features
        y: Target labels
        method: Balancing method ('undersample', 'oversample', 'smote')
        random_state: Random state
        
    Returns:
        Balanced dataset
    """
    from collections import Counter
    
    # Get class distribution
    class_counts = Counter(y)
    print(f"Original class distribution: {class_counts}")
    
    if method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X.reshape(X.shape[0], -1), y)
        X_balanced = X_balanced.reshape(-1, X.shape[1], X.shape[2])
    
    elif method == "oversample":
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X.reshape(X.shape[0], -1), y)
        X_balanced = X_balanced.reshape(-1, X.shape[1], X.shape[2])
    
    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X.reshape(X.shape[0], -1), y)
        X_balanced = X_balanced.reshape(-1, X.shape[1], X.shape[2])
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Print new class distribution
    new_class_counts = Counter(y_balanced)
    print(f"Balanced class distribution: {new_class_counts}")
    
    return X_balanced, y_balanced
