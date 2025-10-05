"""
Fully Connected Neural Network (FCNN) Baseline Model for TVAE-RRS
Implements feedforward neural network for time series classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional


def build_fcnn_baseline(input_shape: Tuple[int, int],
                        hidden_layers: List[int] = None,
                        dropout: float = 0.3,
                        learning_rate: float = 0.0001,
                        name: str = "FCNN_Baseline") -> Model:
    """
    Build Fully Connected Neural Network baseline model
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        hidden_layers: List of hidden units for FC layers
        dropout: Dropout rate
        learning_rate: Learning rate
        name: Model name
        
    Returns:
        Compiled FCNN model
    """
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Flatten input
    x = layers.Flatten(name='flatten')(inputs)
    
    # Fully connected layers
    for i, hidden_units in enumerate(hidden_layers):
        x = layers.Dense(
            hidden_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            'precision',
            'recall',
            keras.metrics.AUC(name='auroc'),
            keras.metrics.AUC(name='auprc', curve='PR'),
        ],
        run_eagerly=True
    )
    
    return model
