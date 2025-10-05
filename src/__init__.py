"""
TVAE-RRS: Temporal Variational Autoencoder Model for Rapid Response System

A comprehensive implementation of TVAE for clinical deterioration prediction
with modular architecture and extensive evaluation capabilities.
"""

__version__ = "1.0.0"
__author__ = "Trong-Nghia Nguyen"
__email__ = "nghianguyen7171@gmail.com"

# Import main modules
from .models.tvae import build_tvae_model, TVAE
from .models.rnn_baseline import build_rnn_baseline
from .models.bilstm_attention import build_bilstm_attention_baseline
from .models.dcnn import build_dcnn_baseline
from .models.fcnn import build_fcnn_baseline
from .models.xgbm_baseline import build_xgbm_baseline

from .dataset_loader import DatasetLoader
from .data_preprocessing import DataPreprocessor

from .evaluation.evaluate_metrics import ModelEvaluator, evaluate_model
from .evaluation.visualize_results import plot_training_history, plot_model_comparison
from .evaluation.t_sne_latent import visualize_latent_space_tvae

from .training.utils_train import train_tvae_model, train_baseline_model, cross_validate_model

from .utils.config import Config, get_config
from .utils.losses import get_loss_function, get_metrics
from .utils.window_processing import WindowProcessor

__all__ = [
    # Models
    "build_tvae_model",
    "TVAE", 
    "build_rnn_baseline",
    "build_bilstm_attention_baseline",
    "build_dcnn_baseline",
    "build_fcnn_baseline",
    "build_xgbm_baseline",
    
    # Data
    "DatasetLoader",
    "DataPreprocessor",
    "WindowProcessor",
    
    # Evaluation
    "ModelEvaluator",
    "evaluate_model",
    "plot_training_history",
    "plot_model_comparison",
    "visualize_latent_space_tvae",
    
    # Training
    "train_tvae_model",
    "train_baseline_model",
    "cross_validate_model",
    
    # Utils
    "Config",
    "get_config",
    "get_loss_function",
    "get_metrics",
]
