"""
Simple test script to verify TVAE-RRS functionality
"""

import numpy as np
import tensorflow as tf
from src.models.tvae import build_tvae_model
from src.models.rnn_baseline import build_rnn_baseline
from src.utils.config import Config
from src.utils.window_processing import WindowProcessor
from src.evaluation.evaluate_metrics import ModelEvaluator


def test_tvae_model():
    """Test TVAE model creation and basic functionality"""
    print("Testing TVAE model...")
    
    # Create dummy data
    X = np.random.randn(100, 16, 25)  # 100 samples, 16 time steps, 25 features
    y = np.random.randint(0, 2, (100, 2))  # Binary classification
    
    # Build model
    model = build_tvae_model(
        input_shape=(16, 25),
        latent_dim=8,
        learning_rate=0.001
    )
    
    # Test forward pass
    predictions = model.predict(X[:10])
    print(f"TVAE prediction shape: {predictions[1].shape}")  # Classification output
    
    # Test training step
    model.fit(X[:50], y[:50], epochs=1, verbose=0)
    print("TVAE model test passed!")
    
    return model


def test_baseline_models():
    """Test baseline model creation"""
    print("\nTesting baseline models...")
    
    # Create dummy data
    X = np.random.randn(100, 16, 25)
    y = np.random.randint(0, 2, (100, 2))
    
    # Test RNN baseline
    rnn_model = build_rnn_baseline(input_shape=(16, 25))
    rnn_pred = rnn_model.predict(X[:10])
    print(f"RNN prediction shape: {rnn_pred.shape}")
    
    # Test training
    rnn_model.fit(X[:50], y[:50], epochs=1, verbose=0)
    print("RNN baseline test passed!")
    
    return rnn_model


def test_window_processor():
    """Test window processing functionality"""
    print("\nTesting window processor...")
    
    # Create dummy dataframe
    import pandas as pd
    data = {
        'Patient': [1] * 50 + [2] * 50,
        'target': np.random.randint(0, 2, 100),
        'SBP': np.random.normal(120, 20, 100),
        'HR': np.random.normal(80, 15, 100),
        'RR': np.random.normal(16, 4, 100),
        'BT': np.random.normal(37, 1, 100)
    }
    df = pd.DataFrame(data)
    
    # Test window processing
    processor = WindowProcessor(window_size=8, stride=1)
    X, y, y_onehot = processor.process_cnuh_data(df)
    
    print(f"Window processed data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"One-hot labels shape: {y_onehot.shape}")
    print("Window processor test passed!")
    
    return X, y, y_onehot


def test_evaluation():
    """Test evaluation functionality"""
    print("\nTesting evaluation...")
    
    # Create dummy predictions
    y_true = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    # Test evaluator
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred_proba)
    
    print(f"Evaluation metrics: {list(metrics.keys())}")
    print(f"AUROC: {metrics.get('auroc', 'N/A'):.4f}")
    print("Evaluation test passed!")
    
    return metrics


def test_config():
    """Test configuration functionality"""
    print("\nTesting configuration...")
    
    config = Config()
    print(f"Default window size: {config.data.window_size}")
    print(f"Default latent dim: {config.model.latent_dim}")
    print(f"Default epochs: {config.training.epochs}")
    print("Configuration test passed!")
    
    return config


def main():
    """Run all tests"""
    print("="*50)
    print("TVAE-RRS FUNCTIONALITY TEST")
    print("="*50)
    
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Run tests
        tvae_model = test_tvae_model()
        rnn_model = test_baseline_models()
        X, y, y_onehot = test_window_processor()
        metrics = test_evaluation()
        config = test_config()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*50)
        print("\nTVAE-RRS is ready for use!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
