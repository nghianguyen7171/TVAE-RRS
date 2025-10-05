"""
Training utilities for TVAE-RRS
Includes training functions, callbacks, and utilities
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import warnings


class TrainingCallback(callbacks.Callback):
    """
    Custom training callback for TVAE-RRS
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 save_best_only: bool = True,
                 monitor: str = "val_loss",
                 mode: str = "min",
                 patience: int = 20,
                 verbose: bool = True):
        """
        Initialize TrainingCallback
        
        Args:
            log_dir: Directory for logs
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor
            mode: Mode for monitoring ('min' or 'max')
            patience: Patience for early stopping
            verbose: Verbose output
        """
        super().__init__()
        self.log_dir = log_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.training_start_time = time.time()
        if self.verbose:
            print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            warnings.warn(f"Metric {self.monitor} not available")
            return
        
        # Check if current score is better
        if self.mode == 'min':
            is_better = current_score < self.best_score
        else:
            is_better = current_score > self.best_score
        
        if is_better:
            self.best_score = current_score
            self.wait = 0
            if self.save_best_only:
                self.model.save_weights(os.path.join(self.log_dir, 'best_model.h5'))
        else:
            self.wait += 1
        
        # Early stopping
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        training_time = time.time() - self.training_start_time
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best {self.monitor}: {self.best_score:.4f}")


class MetricsTracker:
    """
    Track and store training metrics
    """
    
    def __init__(self):
        """Initialize MetricsTracker"""
        self.metrics_history = {}
        self.best_metrics = {}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(metric_value)
            
            # Update best metrics
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = metric_value
            else:
                if metric_name in ['loss', 'val_loss']:
                    if metric_value < self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                else:
                    if metric_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics"""
        return self.best_metrics
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics
        
        Args:
            save_path: Path to save plot
        """
        n_metrics = len(self.metrics_history)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(self.metrics_history.items()):
            if i >= 4:
                break
            
            ax = axes[i]
            ax.plot(values, label=metric_name)
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(len(self.metrics_history), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                callbacks_list: Optional[List[callbacks.Callback]] = None,
                verbose: bool = True) -> keras.callbacks.History:
    """
    Train a model
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        callbacks_list: List of callbacks
        verbose: Verbose output
        
    Returns:
        Training history
    """
    # Compile model if not already compiled
    if not model.optimizer:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', keras.metrics.AUC(name='auroc')]
        )
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=verbose,
        shuffle=True
    )
    
    return history


def train_tvae_model(model: keras.Model,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None,
                     epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     log_dir: str = "logs",
                     save_best_only: bool = True,
                     early_stopping_patience: int = 20,
                     verbose: bool = True) -> keras.callbacks.History:
    """
    Train TVAE model with custom callbacks
    
    Args:
        model: TVAE model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_dir: Directory for logs
        save_best_only: Whether to save only the best model
        early_stopping_patience: Patience for early stopping
        verbose: Verbose output
        
    Returns:
        Training history
    """
    # Create callbacks
    callbacks_list = [
        TrainingCallback(
            log_dir=log_dir,
            save_best_only=save_best_only,
            monitor='val_loss',
            mode='min',
            patience=early_stopping_patience,
            verbose=verbose
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=verbose
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        callbacks_list=callbacks_list,
        verbose=verbose
    )
    
    return history


def train_baseline_model(model: keras.Model,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None,
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         log_dir: str = "logs",
                         model_name: str = "baseline",
                         verbose: bool = True) -> keras.callbacks.History:
    """
    Train baseline model
    
    Args:
        model: Baseline model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        log_dir: Directory for logs
        model_name: Name of the model
        verbose: Verbose output
        
    Returns:
        Training history
    """
    # Create callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=verbose
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=verbose
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=verbose
        )
    ]
    
    # Train model
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        callbacks_list=callbacks_list,
        verbose=verbose
    )
    
    return history


def cross_validate_model(model_builder: Callable,
                         X: np.ndarray,
                         y: np.ndarray,
                         cv_folds: int = 5,
                         cv_strategy: str = "stratified_kfold",
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         random_state: int = 42,
                         verbose: bool = True) -> Dict[str, List[float]]:
    """
    Perform cross-validation on a model
    
    Args:
        model_builder: Function that builds the model
        X: Input features
        y: Target labels
        cv_folds: Number of CV folds
        cv_strategy: CV strategy
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        random_state: Random state
        verbose: Verbose output
        
    Returns:
        Dictionary of CV results
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    # Create CV splits
    if cv_strategy == "stratified_kfold":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif cv_strategy == "kfold":
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    
    cv_results = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_auroc': [],
        'val_auroc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if verbose:
            print(f"Training fold {fold + 1}/{cv_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Convert binary labels to one-hot encoded for model training
        y_train_onehot = np.zeros((len(y_train), 2), dtype=np.float32)
        y_val_onehot = np.zeros((len(y_val), 2), dtype=np.float32)
        for idx in range(len(y_train)):
            y_train_onehot[idx, int(y_train[idx])] = 1.0
        for idx in range(len(y_val)):
            y_val_onehot[idx, int(y_val[idx])] = 1.0
        
        # Build model
        model = model_builder()
        
        # Train model
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train_onehot,
            X_val=X_val,
            y_val=y_val_onehot,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=False
        )
        
        # Store results
        cv_results['train_loss'].append(history.history['loss'][-1])
        cv_results['val_loss'].append(history.history['val_loss'][-1])
        cv_results['train_accuracy'].append(history.history['accuracy'][-1])
        cv_results['val_accuracy'].append(history.history['val_accuracy'][-1])
        cv_results['train_auroc'].append(history.history['auroc'][-1])
        cv_results['val_auroc'].append(history.history['val_auroc'][-1])
    
    return cv_results


def hyperparameter_tuning(model_builder: Callable,
                          param_grid: Dict[str, List],
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          epochs: int = 50,
                          batch_size: int = 32,
                          cv_folds: int = 3,
                          scoring: str = 'val_auroc',
                          n_jobs: int = 1,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using grid search
    
    Args:
        model_builder: Function that builds the model with parameters
        param_grid: Parameter grid for tuning
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        cv_folds: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbose output
        
    Returns:
        Best parameters and results
    """
    from sklearn.model_selection import ParameterGrid
    
    best_score = float('-inf')
    best_params = None
    results = []
    
    # Generate parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    
    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations")
    
    for i, params in enumerate(param_combinations):
        if verbose:
            print(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")
        
        # Build model with parameters
        model = model_builder(**params)
        
        # Cross-validation
        cv_results = cross_validate_model(
            model_builder=lambda: model_builder(**params),
            X=X_train,
            y=y_train,
            cv_folds=cv_folds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )
        
        # Calculate mean score
        mean_score = np.mean(cv_results[scoring])
        
        # Store results
        result = {
            'params': params,
            'mean_score': mean_score,
            'cv_results': cv_results
        }
        results.append(result)
        
        # Update best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        if verbose:
            print(f"Mean {scoring}: {mean_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'results': results
    }


def save_training_results(results: Dict[str, Any], 
                         output_path: str) -> None:
    """
    Save training results to file
    
    Args:
        results: Training results
        output_path: Output file path
    """
    import pickle
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)


def load_training_results(input_path: str) -> Dict[str, Any]:
    """
    Load training results from file
    
    Args:
        input_path: Input file path
        
    Returns:
        Training results
    """
    import pickle
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    return results
