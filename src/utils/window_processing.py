"""
Window Interval Processing (WIP) module for TVAE-RRS
Handles temporal windowing and sequence generation for time series data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class WindowProcessor:
    """
    Window Interval Processing class for time series data
    Implements sliding window approach for temporal feature extraction
    """
    
    def __init__(self, 
                 window_size: int = 16,
                 stride: int = 1,
                 prediction_horizon: int = 1,
                 normalize: bool = True,
                 scaler_type: str = "standard"):
        """
        Initialize WindowProcessor
        
        Args:
            window_size: Size of the sliding window
            stride: Step size for sliding window
            prediction_horizon: Number of time steps ahead to predict
            normalize: Whether to normalize features
            scaler_type: Type of scaler ('standard', 'minmax')
        """
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
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
        if not self.normalize:
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
    
    def process_cnuh_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process CNUH dataset with windowing
        
        Args:
            df: Input dataframe with CNUH data
            
        Returns:
            Tuple of (X, y, y_onehot) arrays
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
        
        # Normalize features
        df_normalized = self._normalize_features(df, features_list, fit=True)
        
        # Get patient information
        patient_ids = df_normalized["Patient"].unique()
        
        # Process each patient
        data_info = {"x": [], "y": [], "seq_y": []}
        for patient_id in patient_ids:
            df_patient = df_normalized[df_normalized["Patient"] == patient_id]
            
            # Generate windows for this patient
            for idx in range(len(df_patient) - self.window_size + 1 - self.stride):
                from_idx = idx
                to_idx = idx + self.window_size
                target_idx = to_idx + self.stride - 1
                
                # Check if target index is within bounds
                if target_idx >= len(df_patient):
                    continue
                
                # Extract window data
                window_x = df_patient[features_list].iloc[from_idx:to_idx].values
                window_y = df_patient["target"].iloc[target_idx]
                
                data_info["x"].append(window_x)
                data_info["y"].append(window_y)
                data_info["seq_y"].append(window_y)
        
        # Convert to numpy arrays
        X = np.array(data_info["x"])
        y = np.array(data_info["y"])
        
        # Convert to one-hot encoding
        y_onehot = np.zeros((len(y), 2), dtype=np.float32)
        for idx in range(len(y)):
            y_onehot[idx, int(y[idx])] = 1.0
        
        return X, y, y_onehot
    
    def process_uv_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process UV dataset with windowing
        
        Args:
            df: Input dataframe with UV data
            
        Returns:
            Tuple of (X, y, y_onehot) arrays
        """
        # UV features (exclude id and y columns)
        features_list = [col for col in df.columns if col not in ["id", "y"]]
        self.feature_names = features_list
        
        # Normalize features
        df_normalized = self._normalize_features(df, features_list, fit=True)
        
        # Get patient information
        patient_ids = df_normalized["id"].unique()
        
        # Process each patient
        data_info = {}
        for patient_id in patient_ids:
            df_patient = df_normalized[df_normalized["id"] == patient_id]
            
            # Generate windows for this patient
            for idx in range(len(df_patient) - self.window_size + 1 - self.stride):
                from_idx = idx
                to_idx = idx + self.window_size - 1
                to_target = to_idx + self.stride
                
                # Extract window data
                window_data = {
                    "pid": df_patient["id"].iloc[from_idx:to_idx + 1].values,
                    "x": df_patient[features_list].iloc[from_idx:to_idx + 1].values,
                    "y": df_patient["y"].iloc[from_idx:to_target].values,
                    "seq_y": df_patient["y"].iloc[to_target]
                }
                
                # Append to data_info
                for key in window_data:
                    if key not in data_info:
                        data_info[key] = []
                    data_info[key].append(window_data[key])
        
        # Convert to numpy arrays
        for key in data_info:
            data_info[key] = np.array(data_info[key])
        
        # Extract X, y, y_onehot
        X = data_info["x"]
        y = data_info["seq_y"]
        
        # Convert to one-hot encoding
        y_onehot = np.zeros((len(y), 2), dtype=np.float32)
        for idx in range(len(y)):
            y_onehot[idx, y[idx]] = 1.0
        
        return X, y, y_onehot
    
    def process_data(self, df: pd.DataFrame, dataset_type: str = "cnuh") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process data based on dataset type
        
        Args:
            df: Input dataframe
            dataset_type: Type of dataset ('cnuh' or 'uv')
            
        Returns:
            Tuple of (X, y, y_onehot) arrays
        """
        if dataset_type.lower() == "cnuh":
            return self.process_cnuh_data(df)
        elif dataset_type.lower() == "uv":
            return self.process_uv_data(df)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in processing"""
        return self.feature_names
    
    def get_scaler(self):
        """Get the fitted scaler"""
        return self.scaler


class TemporalFeatureExtractor:
    """
    Extract temporal features from time series data
    """
    
    def __init__(self, 
                 window_size: int = 8,
                 include_statistics: bool = True,
                 include_rate_of_change: bool = True):
        """
        Initialize TemporalFeatureExtractor
        
        Args:
            window_size: Size of temporal window
            include_statistics: Whether to include statistical features
            include_rate_of_change: Whether to include rate of change features
        """
        self.window_size = window_size
        self.include_statistics = include_statistics
        self.include_rate_of_change = include_rate_of_change
    
    def _create_multi_column_names(self, col: str, window: int) -> List[str]:
        """Create multi-column names for temporal features"""
        return [f"{col}_{i - (window - 1)}" for i in range(window)]
    
    def _compute_statistics(self, df: pd.DataFrame, col_list: List[str], col_name: str) -> pd.DataFrame:
        """Compute statistical features"""
        df[f"mean_{col_name}"] = np.mean(df[col_list], axis=1)
        df[f"std_{col_name}"] = np.std(df[col_list], axis=1)
        df[f"max_{col_name}"] = np.max(df[col_list], axis=1)
        df[f"min_{col_name}"] = np.min(df[col_list], axis=1)
        return df
    
    def _compute_rate_of_change(self, df: pd.DataFrame, col_list: List[str], col_zero: str) -> pd.DataFrame:
        """Compute rate of change features"""
        for col in col_list:
            try:
                df[f"roc_{col}"] = (df[col_zero] - df[col]) / df[col]
            except:
                df[f"roc_{col}"] = df[col_zero]
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame, time_vars: List[str]) -> pd.DataFrame:
        """
        Extract temporal features from time series variables
        
        Args:
            df: Input dataframe
            time_vars: List of time series variable names
            
        Returns:
            Dataframe with additional temporal features
        """
        df_enhanced = df.copy()
        
        for var in time_vars:
            # Create multi-column names
            col_list = self._create_multi_column_names(var, self.window_size)
            
            # Compute statistics if requested
            if self.include_statistics:
                df_enhanced = self._compute_statistics(df_enhanced, col_list, var)
            
            # Compute rate of change if requested
            if self.include_rate_of_change:
                col_list_roc = self._create_multi_column_names(var, self.window_size - 1)
                df_enhanced = self._compute_rate_of_change(df_enhanced, col_list_roc, f"{var}_0")
        
        return df_enhanced


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
