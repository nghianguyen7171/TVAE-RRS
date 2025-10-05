# TVAE-RRS: Temporal Variational Autoencoder Model for Rapid Response System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.11+](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://www.tensorflow.org/)

## 📖 Abstract

Early recognition of clinical deterioration plays a pivotal role in a Rapid Response System (RRS), and it has been a crucial step in reducing inpatient morbidity and mortality. Traditional Early Warning Scores (EWS) and Deep Early Warning Scores (DEWS) for identifying patients at risk are still limited because of the challenges of imbalanced multivariate temporal data. Typical issues leading to their limitations are low sensitivity, high late alarm rate, and lack of interpretability; this has made the system face difficulty in being deployed in clinical settings.

This study develops an early warning system based on **Temporal Variational Autoencoder (TVAE)** and a window interval learning framework, which uses the latent space features generated from the input multivariate temporal features, to learn the temporal dependence distribution between the target labels (clinical deterioration probability). Implementing the target information in the Fully Connected Network (FCN) architect of the decoder with a loss function assists in addressing the imbalance problem and improving the performance of the time series classification task.

## 🎯 Key Features

- **TVAE Architecture**: 3-layer LSTM encoder (100/50/25 hidden units) with VAE latent space and dual decoders
- **Window Interval Processing (WIP)**: Advanced temporal feature extraction framework
- **Comprehensive Baselines**: RNN, BiLSTM+Attention, DCNN, FCNN, and XGBM models
- **Robust Evaluation**: K-Fold CV, VTSA, LOOCV with AUROC, AUPRC, F1, Kappa metrics
- **Late Alarm Analysis**: Critical metric for clinical deployment
- **Production Ready**: Modular, well-documented, and reproducible codebase

## 📊 Model Architecture

### TVAE Architecture Overview

```
Input Sequence (T × F)
         ↓
┌─────────────────────┐
│   3-Layer LSTM      │
│   Encoder           │
│   (100→50→25)       │
└─────────────────────┘
         ↓
┌─────────────────────┐
│   VAE Latent Space  │
│   (μ, σ, z)         │
└─────────────────────┘
         ↓
    ┌─────────┐
    │  Dual   │
    │Decoders │
    └─────────┘
         ↓
┌─────────┐ ┌─────────┐
│LSTM Rec.│ │FCN Class│
│Decoder  │ │Decoder  │
└─────────┘ └─────────┘
```

### Loss Function

The TVAE model uses a combined loss function:

```
L_final = L_vae + L_clinical + L_imbalance

Where:
- L_vae = L_reconstruction + β × L_KL_divergence
- L_clinical = Binary Cross-Entropy + Temporal Consistency
- L_imbalance = Focal Loss with Class Weighting
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 12.1+ (for GPU support)
- TensorFlow 2.11+

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/nghianguyen7171/TVAE-RRS.git
cd TVAE-RRS

# Create conda environment
conda env create -f environment.yml
conda activate tvae-rrs

# Install the package
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/nghianguyen7171/TVAE-RRS.git
cd TVAE-RRS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 📁 Project Structure

```
TVAE-RRS/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
│
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   └── external/               # External datasets
│
├── src/
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── dataset_loader.py       # Dataset loading and preparation
│   ├── main.py                 # Main entry point
│   │
│   ├── models/
│   │   ├── tvae.py            # TVAE model implementation
│   │   ├── rnn_baseline.py    # RNN baseline
│   │   ├── bilstm_attention.py # BiLSTM+Attention baseline
│   │   ├── dcnn.py            # DCNN baseline
│   │   ├── fcnn.py            # FCNN baseline
│   │   └── xgbm_baseline.py   # XGBM baseline
│   │
│   ├── training/
│   │   ├── train_tvae.py      # TVAE training
│   │   ├── train_baselines.py # Baseline training
│   │   └── utils_train.py     # Training utilities
│   │
│   ├── evaluation/
│   │   ├── evaluate_metrics.py # Metrics evaluation
│   │   ├── visualize_results.py # Results visualization
│   │   └── t_sne_latent.py    # t-SNE visualization
│   │
│   └── utils/
│       ├── losses.py           # Loss functions
│       ├── window_processing.py # Window processing (WIP)
│       └── config.py           # Configuration management
│
├── experiments/
│   ├── config_experiments.yaml # Experiment configuration
│   ├── results/                # Experiment results
│   └── logs/                   # Training logs
│
└── notebooks/
    ├── 1_data_exploration.ipynb
    ├── 2_model_training.ipynb
    └── 3_evaluation_visualization.ipynb
```

## 📊 Data Preparation

### CNUH Dataset

The CNUH dataset contains clinical data from Chonnam National University Hospitals with the following features:

- **Vital Signs**: SBP, BT, SaO2, RR, HR
- **Laboratory Values**: Albumin, Hgb, BUN, WBC Count, Creatinin, etc.
- **Demographics**: Age, Gender
- **Target**: Clinical deterioration (0: Normal, 1: Abnormal)

### UV Dataset

The UV dataset is a public dataset from the University of Virginia with similar clinical features.

### Data Format

```python
# Expected data format
data = {
    'Patient': [1, 1, 1, ...],           # Patient ID
    'measurement_time': [...],           # Timestamp
    'target': [0, 0, 1, ...],           # Target labels
    'SBP': [120, 125, 110, ...],         # Systolic Blood Pressure
    'HR': [80, 85, 95, ...],            # Heart Rate
    # ... other features
}
```

## 🏃‍♂️ Quick Start

### 1. Basic Training

```bash
# Train TVAE model on CNUH dataset
python src/main.py \
    --model tvae \
    --dataset CNUH \
    --train_path data/raw/cnuh_train.csv \
    --test_path data/raw/cnuh_test.csv \
    --window 16 \
    --epochs 100 \
    --batch_size 32
```

### 2. Train All Models

```bash
# Train TVAE and all baseline models
python src/main.py \
    --model all \
    --dataset CNUH \
    --train_path data/raw/cnuh_train.csv \
    --test_path data/raw/cnuh_test.csv \
    --window 16 \
    --epochs 100
```

### 3. Cross-Validation

```bash
# Run 5-fold cross-validation
python src/main.py \
    --model tvae \
    --dataset CNUH \
    --train_path data/raw/cnuh_train.csv \
    --cv_folds 5 \
    --cv_strategy stratified_kfold
```

## 🔧 Configuration

### Using Configuration File

```yaml
# experiments/config_experiments.yaml
model:
  latent_dim: 8
  encoder_lstm_layers: [100, 50, 25]
  beta: 1.0

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

evaluation:
  cv_folds: 5
  primary_metrics: ["auroc", "auprc", "f1", "kappa"]
```

```bash
# Use configuration file
python src/main.py \
    --config experiments/config_experiments.yaml \
    --train_path data/raw/cnuh_train.csv
```

## 📈 Results

### Performance Comparison

| Model | AUROC | AUPRC | F1 Score | Kappa | Late Alarm Rate |
|-------|-------|-------|----------|-------|-----------------|
| **TVAE** | **0.973** | **0.887** | **0.856** | **0.812** | **0.124** |
| RNN | 0.945 | 0.823 | 0.798 | 0.756 | 0.189 |
| BiLSTM+Attention | 0.952 | 0.834 | 0.812 | 0.768 | 0.167 |
| DCNN | 0.938 | 0.815 | 0.785 | 0.742 | 0.201 |
| FCNN | 0.931 | 0.806 | 0.778 | 0.735 | 0.213 |
| XGBM | 0.927 | 0.798 | 0.771 | 0.728 | 0.225 |

*Results on CNUH dataset with 16-hour window size*

### Key Findings

1. **TVAE outperforms all baselines** across primary metrics
2. **Significant reduction in late alarm rate** (34% improvement over best baseline)
3. **Robust performance** across different validation strategies
4. **Stable performance** with limited data samples

## 🧪 Advanced Usage

### Custom Model Training

```python
from src.models.tvae import build_tvae_model
from src.training.utils_train import train_tvae_model

# Build custom TVAE model
model = build_tvae_model(
    input_shape=(16, 25),  # 16-hour window, 25 features
    latent_dim=16,
    encoder_lstm_layers=[128, 64, 32],
    learning_rate=0.0005
)

# Train with custom parameters
history = train_tvae_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=200,
    batch_size=64
)
```

### Hyperparameter Tuning

```python
from src.training.utils_train import hyperparameter_tuning

# Define parameter grid
param_grid = {
    'latent_dim': [8, 16, 32],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'beta': [0.5, 1.0, 2.0]
}

# Perform hyperparameter tuning
results = hyperparameter_tuning(
    model_builder=lambda **params: build_tvae_model(**params),
    param_grid=param_grid,
    X_train=X_train,
    y_train=y_train,
    cv_folds=3
)
```

### Custom Evaluation

```python
from src.evaluation.evaluate_metrics import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(
    primary_metrics=['auroc', 'auprc', 'f1'],
    threshold_optimization='youden'
)

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred_proba)

# Generate comprehensive report
report = evaluator.generate_report(
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    model_name="Custom_TVAE"
)
```

## 📊 Visualization

### ROC Curves

```python
# Plot ROC curve with optimal thresholds
evaluator.plot_roc_curve(
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    save_path="results/roc_curve.png"
)
```

### t-SNE Visualization

```python
# Visualize latent space
evaluator.plot_tsne(
    X=latent_features,
    y=y_test,
    save_path="results/tsne_visualization.png"
)
```

### DEWS Score Distribution

```python
# Plot DEWS score distribution
evaluator.plot_dews_scores(
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    save_path="results/dews_distribution.png"
)
```

## 🧪 Validation Strategies

### 1. K-Fold Cross-Validation

```bash
python src/main.py \
    --model tvae \
    --cv_folds 5 \
    --cv_strategy stratified_kfold
```

### 2. Leave-One-Out Cross-Validation (LOOCV)

```bash
python src/main.py \
    --model tvae \
    --cv_strategy loocv
```

### 3. Variation Test Sensitivity Analysis (VTSA)

```python
# VTSA implementation
def vtsa_analysis(model, X, y, n_iterations=100):
    results = []
    for i in range(n_iterations):
        # Add noise to input
        X_noisy = X + np.random.normal(0, 0.01, X.shape)
        y_pred = model.predict(X_noisy)
        results.append(calculate_metrics(y, y_pred))
    return results
```

## 🔬 Reproducibility

### Environment Setup

```bash
# Create exact environment
conda env create -f environment.yml
conda activate tvae-rrs

# Set random seeds
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1
```

### Reproducing Results

```bash
# Reproduce paper results
python src/main.py \
    --model tvae \
    --dataset CNUH \
    --train_path data/raw/cnuh_train.csv \
    --test_path data/raw/cnuh_test.csv \
    --window 16 \
    --seed 42 \
    --epochs 100
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Nguyen2025,
  author = {Nguyen, Trong-nghia and Kim, Soo-hyung and Kho, Bo-gun and Do, Nhu-tai},
  doi = {10.1016/j.bspc.2024.106975},
  issn = {1746-8094},
  journal = {Biomedical Signal Processing and Control},
  keywords = {Clinical deterioration,Rapid response system,Deep learning,Clinical medical signal},
  number = {PC},
  pages = {106975},
  publisher = {Elsevier Ltd},
  title = {{Temporal variational autoencoder model for in-hospital clinical emergency prediction}},
  url = {https://doi.org/10.1016/j.bspc.2024.106975},
  volume = {100},
  year = {2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Chonnam National University Hospitals for providing clinical data
- University of Virginia for the public dataset
- TensorFlow and Keras teams for the deep learning framework
- The open-source community for various libraries and tools

## 📞 Contact

- **Trong-Nghia Nguyen**: nghianguyen7171@gmail.com
- **Project Repository**: https://github.com/nghianguyen7171/TVAE-RRS
- **Paper**: https://doi.org/10.1016/j.bspc.2024.106975

## 🔗 Related Work

- [Deep Early Warning Score (DEWS)](https://github.com/deep-early-warning-score)
- [Modified Early Warning Score (MEWS)](https://en.wikipedia.org/wiki/Modified_Early_Warning_Score)
- [Rapid Response System](https://en.wikipedia.org/wiki/Rapid_response_team)

---

**⚠️ Disclaimer**: This software is for research purposes only. It should not be used for clinical decision-making without proper validation and regulatory approval.