"""
Deep Convolutional Neural Network (DCNN) Baseline Model for TVAE-RRS
Implements 1D CNN for time series classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional


def build_dcnn_baseline(input_shape: Tuple[int, int],
                        filters: List[int] = None,
                        kernel_sizes: List[int] = None,
                        dropout: float = 0.5,
                        learning_rate: float = 0.0001,
                        name: str = "DCNN_Baseline") -> Model:
    """
    Build Deep CNN baseline model
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        filters: List of filter numbers for Conv1D layers
        kernel_sizes: List of kernel sizes for Conv1D layers
        dropout: Dropout rate
        learning_rate: Learning rate
        name: Model name
        
    Returns:
        Compiled DCNN model
    """
    if filters is None:
        filters = [32, 64]
    if kernel_sizes is None:
        kernel_sizes = [3, 3]
    
    # Input layer
    input_layer = layers.Input(shape=input_shape, name='input')
    
    # Convolutional layers
    x = input_layer
    for i, (filt, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        x = layers.Conv1D(
            filters=filt,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        )(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
    
    # Flatten layer
    x = layers.Flatten(name='flatten')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout, name='dropout1')(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs, name=name)
    
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
