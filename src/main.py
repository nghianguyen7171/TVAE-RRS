"""
Main entry point for TVAE-RRS
Command-line interface for training and evaluating models
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras

# Import modules
from src.utils.config import Config, get_config
from src.dataset_loader import DatasetLoader
from src.models.tvae_simple import build_tvae_model
from src.models.rnn_baseline import build_rnn_baseline
from src.models.bilstm_attention import build_bilstm_attention_baseline
from src.models.dcnn import build_dcnn_baseline
from src.models.fcnn import build_fcnn_baseline
from src.models.xgbm_baseline import build_xgbm_baseline
from src.training.utils_train import train_tvae_model, train_baseline_model, cross_validate_model
from src.evaluation.evaluate_metrics import evaluate_model, ModelEvaluator


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_tvae(config: Config, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None) -> keras.Model:
    """
    Train TVAE model
    
    Args:
        config: Configuration object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained TVAE model
    """
    # Build model
    model = build_tvae_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        latent_dim=config.model.latent_dim,
        encoder_lstm_layers=config.model.encoder_lstm_layers,
        reconstruction_lstm_layers=config.model.reconstruction_lstm_layers,
        classification_fc_layers=config.model.classification_fc_layers,
        learning_rate=config.training.learning_rate,
        reconstruction_weight=config.model.reconstruction_weight,
        classification_weight=config.model.classification_weight,
        kl_weight=config.model.kl_weight,
        clinical_weight=config.model.clinical_weight,
        imbalance_weight=config.model.imbalance_weight,
        beta=config.model.beta
    )
    
    # Train model
    history = train_tvae_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        log_dir=os.path.join(config.experiment.log_dir, "tvae"),
        save_best_only=config.training.save_best_only,
        early_stopping_patience=config.training.early_stopping_patience,
        verbose=True
    )
    
    return model


def train_baseline_models(config: Config,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> Dict[str, keras.Model]:
    """
    Train baseline models
    
    Args:
        config: Configuration object
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Dictionary of trained baseline models
    """
    models = {}
    
    # RNN Baseline
    if "rnn" in config.experiment.baseline_models:
        print("Training RNN Baseline...")
        rnn_model = build_rnn_baseline(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_layers=config.baseline.rnn_hidden_layers,
            dropout=config.baseline.rnn_dropout,
            learning_rate=config.training.learning_rate
        )
        
        history = train_baseline_model(
            model=rnn_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            log_dir=os.path.join(config.experiment.log_dir, "rnn"),
            model_name="rnn",
            verbose=True
        )
        models["rnn"] = rnn_model
    
    # BiLSTM + Attention Baseline
    if "bilstm_attention" in config.experiment.baseline_models:
        print("Training BiLSTM + Attention Baseline...")
        bilstm_model = build_bilstm_attention_baseline(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_size=config.baseline.bilstm_hidden_size,
            attention_dim=config.baseline.attention_dim,
            dropout=config.baseline.bilstm_dropout,
            learning_rate=config.training.learning_rate
        )
        
        history = train_baseline_model(
            model=bilstm_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            log_dir=os.path.join(config.experiment.log_dir, "bilstm_attention"),
            model_name="bilstm_attention",
            verbose=True
        )
        models["bilstm_attention"] = bilstm_model
    
    # DCNN Baseline
    if "dcnn" in config.experiment.baseline_models:
        print("Training DCNN Baseline...")
        dcnn_model = build_dcnn_baseline(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            filters=config.baseline.dcnn_filters,
            kernel_sizes=config.baseline.dcnn_kernel_sizes,
            dropout=config.baseline.dcnn_dropout,
            learning_rate=config.training.learning_rate
        )
        
        history = train_baseline_model(
            model=dcnn_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            log_dir=os.path.join(config.experiment.log_dir, "dcnn"),
            model_name="dcnn",
            verbose=True
        )
        models["dcnn"] = dcnn_model
    
    # FCNN Baseline
    if "fcnn" in config.experiment.baseline_models:
        print("Training FCNN Baseline...")
        fcnn_model = build_fcnn_baseline(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hidden_layers=config.baseline.fcnn_layers,
            dropout=config.baseline.fcnn_dropout,
            learning_rate=config.training.learning_rate
        )
        
        history = train_baseline_model(
            model=fcnn_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            log_dir=os.path.join(config.experiment.log_dir, "fcnn"),
            model_name="fcnn",
            verbose=True
        )
        models["fcnn"] = fcnn_model
    
    # XGBM Baseline
    if "xgbm" in config.experiment.baseline_models:
        print("Training XGBM Baseline...")
        xgbm_model = build_xgbm_baseline(
            n_estimators=config.baseline.xgb_n_estimators,
            max_depth=config.baseline.xgb_max_depth,
            learning_rate=config.baseline.xgb_learning_rate,
            subsample=config.baseline.xgb_subsample,
            random_state=config.experiment.seed
        )
        
        xgbm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            early_stopping_rounds=config.training.early_stopping_patience,
            verbose=True
        )
        models["xgbm"] = xgbm_model
    
    return models


def evaluate_models(models: Dict[str, Any],
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   config: Config) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        config: Configuration object
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Create output directory for this model
        output_dir = os.path.join(config.experiment.output_dir, model_name)
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            output_dir=output_dir,
            plot_results=True
        )
        
        results[model_name] = metrics
    
    return results


def run_cross_validation(config: Config,
                        X: np.ndarray,
                        y: np.ndarray,
                        model_type: str = "tvae") -> Dict[str, Any]:
    """
    Run cross-validation
    
    Args:
        config: Configuration object
        X: Input features
        y: Target labels
        model_type: Type of model to evaluate
        
    Returns:
        Cross-validation results
    """
    def model_builder():
        if model_type == "tvae":
            return build_tvae_model(
                input_shape=(X.shape[1], X.shape[2]),
                latent_dim=config.model.latent_dim,
                learning_rate=config.training.learning_rate
            )
        elif model_type == "rnn":
            return build_rnn_baseline(
                input_shape=(X.shape[1], X.shape[2]),
                learning_rate=config.training.learning_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Run cross-validation
    cv_results = cross_validate_model(
        model_builder=model_builder,
        X=X,
        y=y,
        cv_folds=config.evaluation.cv_folds,
        cv_strategy=config.evaluation.cv_strategy,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        random_state=config.experiment.seed,
        verbose=True
    )
    
    return cv_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TVAE-RRS: Temporal Variational Autoencoder for Rapid Response System")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="tvae", 
                       choices=["tvae", "rnn", "bilstm_attention", "dcnn", "fcnn", "xgbm", "all"],
                       help="Model to train")
    parser.add_argument("--dataset", type=str, default="CNUH", 
                       choices=["CNUH", "UV"],
                       help="Dataset to use")
    parser.add_argument("--window", type=int, default=16,
                       help="Window size for time series")
    parser.add_argument("--slide", type=int, default=1,
                       help="Stride for sliding window")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--test_path", type=str, default=None,
                       help="Path to test data")
    parser.add_argument("--validation_path", type=str, default=None,
                       help="Path to validation data")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                       help="Output directory")
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                       help="Log directory")
    
    # Evaluation arguments
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--cv_strategy", type=str, default="stratified_kfold",
                       choices=["stratified_kfold", "kfold", "loocv"],
                       help="Cross-validation strategy")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = Config()
    
    # Update configuration with command line arguments
    config.experiment.output_dir = args.output_dir
    config.experiment.log_dir = args.log_dir
    config.experiment.seed = args.seed
    config.data.window_size = args.window
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.evaluation.cv_folds = args.cv_folds
    config.evaluation.cv_strategy = args.cv_strategy
    
    # Load data
    print("Loading data...")
    data_loader = DatasetLoader(
        data_dir=config.data.data_dir,
        processed_data_dir=config.data.processed_data_dir,
        window_size=config.data.window_size,
        stride=args.slide,
        normalize=config.data.normalize_features,
        scaler_type=config.data.handle_missing
    )
    
    datasets = data_loader.load_data(
        train_path=args.train_path,
        dataset_type=args.dataset,
        test_path=args.test_path,
        validation_path=args.validation_path
    )
    
    X_train, y_train, y_train_onehot = datasets['train']
    X_test, y_test, y_test_onehot = datasets.get('test', (None, None, None))
    X_val, y_val, y_val_onehot = datasets.get('validation', (None, None, None))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    if X_test is not None:
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
    
    # Train models
    models = {}
    
    if args.model == "tvae" or args.model == "all":
        print("Training TVAE model...")
        tvae_model = train_tvae(config, X_train, y_train_onehot, X_val, y_val_onehot)
        models["tvae"] = tvae_model
    
    if args.model in ["rnn", "bilstm_attention", "dcnn", "fcnn", "xgbm"] or args.model == "all":
        print("Training baseline models...")
        baseline_models = train_baseline_models(config, X_train, y_train_onehot, X_val, y_val_onehot)
        models.update(baseline_models)
    
    # Evaluate models
    if models and X_test is not None:
        print("Evaluating models...")
        results = evaluate_models(models, X_test, y_test_onehot, config)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  AUROC: {metrics.get('auroc', 'N/A'):.4f}")
            print(f"  AUPRC: {metrics.get('auprc', 'N/A'):.4f}")
            print(f"  F1 Score: {metrics.get('f1_optimal', 'N/A'):.4f}")
            print(f"  Kappa: {metrics.get('kappa_optimal', 'N/A'):.4f}")
    
    # Cross-validation
    if args.cv_folds > 1:
        print(f"Running {args.cv_folds}-fold cross-validation...")
        # Convert one-hot encoded labels to binary for cross-validation
        y_train_binary = np.argmax(y_train_onehot, axis=1)
        cv_results = run_cross_validation(config, X_train, y_train_binary, args.model)
        
        print("\n" + "="*50)
        print("CROSS-VALIDATION RESULTS")
        print("="*50)
        
        for metric, values in cv_results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nTraining and evaluation completed!")


if __name__ == "__main__":
    main()
