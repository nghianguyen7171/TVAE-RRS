"""
Data preprocessing module for TVAE-RRS
Handles data loading, preprocessing, and feature engineering
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class DataPreprocessor:
    """
    Data preprocessing class for TVAE-RRS
    """
    
    def __init__(self,
                 normalize_features: bool = True,
                 scaler_type: str = "standard",
                 handle_missing: str = "forward_fill",
                 outlier_threshold: float = 3.0):
        """
        Initialize DataPreprocessor
        
        Args:
            normalize_features: Whether to normalize features
            scaler_type: Type of scaler ('standard', 'minmax')
            handle_missing: Method to handle missing values
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.normalize_features = normalize_features
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.outlier_threshold = outlier_threshold
        
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
    
    def _get_scaler(self):
        """Get appropriate scaler based on type"""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "none":
            return None
        else:
            # Default to StandardScaler for unknown types
            return StandardScaler()
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str = None) -> pd.DataFrame:
        """
        Handle missing values in dataframe
        
        Args:
            df: Input dataframe
            method: Method to handle missing values
            
        Returns:
            Dataframe with handled missing values
        """
        if method is None:
            method = self.handle_missing
        
        df_processed = df.copy()
        
        if method == "forward_fill":
            df_processed = df_processed.fillna(method='ffill')
        elif method == "backward_fill":
            df_processed = df_processed.fillna(method='bfill')
        elif method == "interpolate":
            df_processed = df_processed.interpolate()
        elif method == "mean":
            df_processed = df_processed.fillna(df_processed.mean())
        elif method == "median":
            df_processed = df_processed.fillna(df_processed.median())
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
        
        return df_processed
    
    def _detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect and handle outliers using Z-score
        
        Args:
            df: Input dataframe
            columns: Columns to check for outliers
            
        Returns:
            Dataframe with outliers handled
        """
        df_processed = df.copy()
        
        for col in columns:
            if col in df_processed.columns:
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                outlier_mask = z_scores > self.outlier_threshold
                
                if outlier_mask.any():
                    warnings.warn(f"Found {outlier_mask.sum()} outliers in column {col}")
                    # Cap outliers at threshold
                    df_processed.loc[outlier_mask, col] = df_processed[col].mean() + \
                        self.outlier_threshold * df_processed[col].std()
        
        return df_processed
    
    def _normalize_features(self, df: pd.DataFrame, features: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Normalize features in the dataframe
        
        Args:
            df: Input dataframe
            features: List of feature names to normalize
            fit: Whether to fit the scaler
            
        Returns:
            Dataframe with normalized features
        """
        if not self.normalize_features:
            return df
        
        if fit:
            self.scaler = self._get_scaler()
            normalized_features = self.scaler.fit_transform(df[features])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized_features = self.scaler.transform(df[features])
        
        # Update dataframe with normalized features
        df_normalized = df.copy()
        for idx, feature_name in enumerate(features):
            df_normalized[feature_name] = normalized_features[:, idx]
        
        return df_normalized
    
    def preprocess_cnuh_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess CNUH dataset
        
        Args:
            df: Input dataframe with CNUH data
            
        Returns:
            Preprocessed dataframe
        """
        # CNUH feature list
        features_list = [
            'Albumin', 'Hgb', 'BUN', 'Alkaline phosphatase', 'WBC Count',
            'SBP', 'Gender', 'Total calcium', 'RR', 'Age', 'Total bilirubin',
            'Creatinin', 'ALT', 'Lactate', 'SaO2', 'AST', 'Glucose', 'Sodium', 'BT',
            'HR', 'CRP', 'Chloride', 'Potassium', 'platelet', 'Total protein'
        ]
        
        # Check if all features exist
        missing_features = [f for f in features_list if f not in df.columns]
        if missing_features:
            warnings.warn(f"Missing features: {missing_features}")
            features_list = [f for f in features_list if f in df.columns]
        
        self.feature_names = features_list
        
        # Handle missing values
        df_processed = self._handle_missing_values(df)
        
        # Detect and handle outliers
        numeric_features = [f for f in features_list if df_processed[f].dtype in ['int64', 'float64']]
        df_processed = self._detect_outliers(df_processed, numeric_features)
        
        # Normalize features
        df_processed = self._normalize_features(df_processed, features_list, fit=True)
        
        self.is_fitted = True
        return df_processed
    
    def preprocess_uv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess UV dataset
        
        Args:
            df: Input dataframe with UV data
            
        Returns:
            Preprocessed dataframe
        """
        # UV features (exclude id and y columns)
        features_list = [col for col in df.columns if col not in ["id", "y"]]
        self.feature_names = features_list
        
        # Handle missing values
        df_processed = self._handle_missing_values(df)
        
        # Detect and handle outliers
        numeric_features = [f for f in features_list if df_processed[f].dtype in ['int64', 'float64']]
        df_processed = self._detect_outliers(df_processed, numeric_features)
        
        # Normalize features
        df_processed = self._normalize_features(df_processed, features_list, fit=True)
        
        self.is_fitted = True
        return df_processed
    
    def preprocess_data(self, df: pd.DataFrame, dataset_type: str = "cnuh") -> pd.DataFrame:
        """
        Preprocess data based on dataset type
        
        Args:
            df: Input dataframe
            dataset_type: Type of dataset ('cnuh' or 'uv')
            
        Returns:
            Preprocessed dataframe
        """
        if dataset_type.lower() == "cnuh":
            return self.preprocess_cnuh_data(df)
        elif dataset_type.lower() == "uv":
            return self.preprocess_uv_data(df)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation")
        
        # Handle missing values
        df_processed = self._handle_missing_values(df)
        
        # Detect and handle outliers
        numeric_features = [f for f in self.feature_names if df_processed[f].dtype in ['int64', 'float64']]
        df_processed = self._detect_outliers(df_processed, numeric_features)
        
        # Normalize features
        df_processed = self._normalize_features(df_processed, self.feature_names, fit=False)
        
        return df_processed
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in preprocessing"""
        return self.feature_names
    
    def get_scaler(self):
        """Get the fitted scaler"""
        return self.scaler
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save preprocessor to file
        
        Args:
            filepath: Path to save preprocessor
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'normalize_features': self.normalize_features,
            'scaler_type': self.scaler_type,
            'handle_missing': self.handle_missing,
            'outlier_threshold': self.outlier_threshold,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
    
    def load_preprocessor(self, filepath: str) -> 'DataPreprocessor':
        """
        Load preprocessor from file
        
        Args:
            filepath: Path to load preprocessor from
            
        Returns:
            Self
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        self.normalize_features = preprocessor_data['normalize_features']
        self.scaler_type = preprocessor_data['scaler_type']
        self.handle_missing = preprocessor_data['handle_missing']
        self.outlier_threshold = preprocessor_data['outlier_threshold']
        self.is_fitted = preprocessor_data['is_fitted']
        
        return self


def load_data(data_path: str, dataset_type: str = "cnuh") -> pd.DataFrame:
    """
    Load data from file
    
    Args:
        data_path: Path to data file
        dataset_type: Type of dataset ('cnuh' or 'uv')
        
    Returns:
        Loaded dataframe
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.pickle'):
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return df


def split_data(X: np.ndarray, y: np.ndarray, 
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets
    
    Args:
        X: Input features
        y: Target labels
        test_size: Proportion of test set
        random_state: Random state
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def create_sequence_data(df: pd.DataFrame, 
                         window_len: int,
                         var_list: List[str],
                         index: str = "patient_id",
                         is_abn: str = "is_abn",
                         target_list: str = "target") -> pd.DataFrame:
    """
    Create sequence data from time series
    
    Args:
        df: Input dataframe
        window_len: Length of sequence window
        var_list: List of variables to include
        index: Patient ID column name
        is_abn: Abnormal flag column name
        target_list: Target column name
        
    Returns:
        Dataframe with sequence data
    """
    return_df = pd.DataFrame(columns=[
        index, is_abn, "sequence", target_list,
        "measurement_time", "detection_time", "event_time"
    ])
    
    patient_list = df[index].unique()
    
    for patient_id in patient_list:
        patient_data = df[df[index] == patient_id].reset_index(drop=True)
        
        for j in range(len(patient_data) - window_len):
            sequence_data = {
                "Patient": patient_data[index].iloc[j],
                "is_abn": patient_data[is_abn].iloc[j],
                "target": patient_data[target_list].iloc[j + window_len - 1],
                "sequence": patient_data[var_list].iloc[j:j + window_len].values,
                "measurement_time": patient_data["measurement_time"].iloc[j],
                "detection_time": patient_data["detection_time"].iloc[j],
                "event_time": patient_data["event_time"].iloc[j]
            }
            
            return_df = pd.concat([return_df, pd.DataFrame([sequence_data])], ignore_index=True)
    
    return return_df
