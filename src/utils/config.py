"""
Configuration module for TVAE-RRS
Contains all hyperparameters and settings for the model
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Dataset paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    external_data_dir: str = "data/external"
    
    # Dataset names
    cnuh_dataset: str = "CNUH"
    uv_dataset: str = "UV"
    
    # Window processing parameters
    window_size: int = 16
    stride: int = 1
    prediction_horizon: int = 1
    
    # Feature lists
    cnuh_features: List[str] = field(default_factory=lambda: [
        'Albumin', 'Hgb', 'BUN', 'Alkaline phosphatase', 'WBC Count',
        'SBP', 'Gender', 'Total calcium', 'RR', 'Age', 'Total bilirubin',
        'Creatinin', 'ALT', 'Lactate', 'SaO2', 'AST', 'Glucose', 'Sodium', 'BT',
        'HR', 'CRP', 'Chloride', 'Potassium', 'platelet', 'Total protein'
    ])
    
    # Data preprocessing
    normalize_features: bool = True
    handle_missing: str = "forward_fill"  # forward_fill, backward_fill, interpolate
    outlier_threshold: float = 3.0  # Z-score threshold for outlier detection


@dataclass
class ModelConfig:
    """Configuration for TVAE model architecture"""
    # Encoder architecture
    encoder_lstm_layers: List[int] = field(default_factory=lambda: [100, 50, 25])
    encoder_dropout: float = 0.2
    encoder_recurrent_dropout: float = 0.1
    
    # VAE latent space
    latent_dim: int = 8
    beta: float = 1.0  # KL divergence weight
    
    # Decoder architecture
    reconstruction_lstm_layers: List[int] = field(default_factory=lambda: [25, 50, 100])
    classification_fc_layers: List[int] = field(default_factory=lambda: [8, 64, 32, 16])
    classification_dropout: float = 0.2
    
    # Loss weights
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    kl_weight: float = 1.0
    clinical_weight: float = 1.0
    imbalance_weight: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"  # adam, sgd, rmsprop
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    
    # Learning rate scheduling
    lr_scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, exponential
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    
    # Validation
    validation_split: float = 0.2
    shuffle: bool = True
    
    # Callbacks
    use_tensorboard: bool = True
    use_wandb: bool = False
    save_best_only: bool = True


@dataclass
class BaselineConfig:
    """Configuration for baseline models"""
    # RNN baseline
    rnn_hidden_layers: List[int] = field(default_factory=lambda: [100, 50, 25])
    rnn_dropout: float = 0.2
    
    # BiLSTM + Attention
    bilstm_hidden_size: int = 100
    attention_dim: int = 10
    bilstm_dropout: float = 0.2
    
    # DCNN
    dcnn_filters: List[int] = field(default_factory=lambda: [32, 64])
    dcnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    dcnn_dropout: float = 0.5
    
    # FCNN
    fcnn_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    fcnn_dropout: float = 0.3
    
    # XGBM
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified_kfold"  # stratified_kfold, kfold, loocv
    
    # Metrics
    primary_metrics: List[str] = field(default_factory=lambda: ["auroc", "auprc", "f1", "kappa"])
    secondary_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "specificity"])
    
    # Threshold optimization
    threshold_optimization: str = "youden"  # youden, f1, precision_recall_curve
    
    # Late alarm analysis
    late_alarm_thresholds: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.95, 0.99])
    
    # Visualization
    plot_roc: bool = True
    plot_pr: bool = True
    plot_tsne: bool = True
    plot_dews_scores: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Experiment tracking
    experiment_name: str = "tvae_rrs_experiment"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Output paths
    output_dir: str = "experiments/results"
    log_dir: str = "experiments/logs"
    model_dir: str = "experiments/models"
    
    # Hyperparameter tuning
    tune_hyperparameters: bool = False
    tune_trials: int = 50
    tune_objective: str = "val_auroc"
    
    # Model comparison
    compare_baselines: bool = True
    baseline_models: List[str] = field(default_factory=lambda: [
        "rnn", "bilstm_attention", "dcnn", "fcnn", "xgbm"
    ])


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories if they don't exist
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.raw_data_dir, exist_ok=True)
        os.makedirs(self.data.processed_data_dir, exist_ok=True)
        os.makedirs(self.data.external_data_dir, exist_ok=True)
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.model_dir, exist_ok=True)


# Default configuration instance
default_config = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)
    return default_config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    import yaml
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'baseline': config.baseline.__dict__,
        'evaluation': config.evaluation.__dict__,
        'experiment': config.experiment.__dict__,
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
