"""
RNN Baseline Model for TVAE-RRS
Implements the RNN baseline from Kwon et al.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional


def build_rnn_baseline(input_shape: Tuple[int, int],
                       hidden_layers: List[int] = None,
                       dropout: float = 0.2,
                       learning_rate: float = 0.0001,
                       name: str = "RNN_Baseline") -> Model:
    """
    Build RNN baseline model (Kwon et al.)
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        hidden_layers: List of LSTM hidden units
        dropout: Dropout rate
        learning_rate: Learning rate
        name: Model name
        
    Returns:
        Compiled RNN model
    """
    if hidden_layers is None:
        hidden_layers = [100, 50, 25]
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input')
    
    # LSTM layers
    x = inputs
    for i, hidden_units in enumerate(hidden_layers):
        return_sequences = i < len(hidden_layers) - 1
        x = layers.LSTM(
            hidden_units,
            activation='tanh',
            dropout=dropout,
            return_sequences=return_sequences,
            name=f'lstm_{i+1}'
        )(x)
    
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
