"""
BiLSTM + Attention Baseline Model for TVAE-RRS
Implements the BiLSTM with attention mechanism from Shamount et al.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from typing import Tuple, List, Optional


def attention_block(inputs, num: int):
    """
    Attention block used in DEWS
    
    Args:
        inputs: Input tensor
        num: Block number for naming
        
    Returns:
        Context vector
    """
    # Compute scoring function using feed forward neural network
    v1 = layers.Dense(10, use_bias=True, name=f'attention_dense1_{num}')(inputs)
    v1_tanh = layers.Activation('relu', name=f'attention_activation_{num}')(v1)
    e = layers.Dense(1, name=f'attention_dense2_{num}')(v1_tanh)
    e_exp = layers.Lambda(lambda x: K.exp(x), name=f'attention_exp_{num}')(e)
    
    # Normalize attention weights
    sum_a_probs = layers.Lambda(
        lambda x: 1 / K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()),
        name=f'attention_norm_{num}'
    )(e_exp)
    a_probs = layers.Multiply(name=f'attention_weights_{num}')([e_exp, sum_a_probs])
    
    # Compute context vector
    context = layers.Multiply(name=f'attention_context_{num}')([inputs, a_probs])
    context = layers.Lambda(lambda x: K.sum(x, axis=1), name=f'attention_sum_{num}')(context)
    
    return context


def build_bilstm_attention_baseline(input_shape: Tuple[int, int],
                                   hidden_size: int = 100,
                                   attention_dim: int = 10,
                                   dropout: float = 0.2,
                                   learning_rate: float = 0.0001,
                                   name: str = "BiLSTM_Attention_Baseline") -> Model:
    """
    Build BiLSTM + Attention baseline model (Shamount et al.)
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        hidden_size: Hidden size for BiLSTM
        attention_dim: Dimension for attention mechanism
        dropout: Dropout rate
        learning_rate: Learning rate
        name: Model name
        
    Returns:
        Compiled BiLSTM + Attention model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Bidirectional LSTM
    bilstm = layers.Bidirectional(
        layers.LSTM(
            hidden_size,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.05),
            kernel_initializer='random_uniform',
            dropout=dropout,
            name='bilstm'
        ),
        merge_mode='ave'
    )(inputs)
    
    # Attention mechanism
    attention_output = attention_block(bilstm, 1)
    
    # Classification layers
    x = layers.Dense(5, activation='relu', name='fc1')(attention_output)
    x = layers.Dropout(dropout, name='dropout1')(x)
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
