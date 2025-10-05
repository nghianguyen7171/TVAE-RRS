"""
Simplified TVAE Model for TVAE-RRS
Compatible with TensorFlow 2.20+
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from typing import Tuple, Dict, Optional, List
import numpy as np


class Sampling(layers.Layer):
    """
    Sampling layer for VAE reparameterization trick
    """
    
    def call(self, inputs):
        """
        Reparameterization trick for VAE
        
        Args:
            inputs: Tuple of (z_mean, z_log_var)
            
        Returns:
            Sampled latent vector z
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class TVAE(Model):
    """
    Simplified Temporal Variational Autoencoder (TVAE) Model
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 latent_dim: int = 8,
                 encoder_lstm_layers: List[int] = None,
                 reconstruction_lstm_layers: List[int] = None,
                 classification_fc_layers: List[int] = None,
                 learning_rate: float = 0.001,
                 name: str = "TVAE",
                 **kwargs):
        """
        Initialize TVAE model
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
            latent_dim: Dimension of latent space
            encoder_lstm_layers: List of LSTM hidden units for encoder
            reconstruction_lstm_layers: List of LSTM hidden units for reconstruction decoder
            classification_fc_layers: List of FC hidden units for classification decoder
            learning_rate: Learning rate
            name: Model name
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Filter out unknown arguments to avoid errors
        known_args = {
            'input_shape': input_shape,
            'latent_dim': latent_dim,
            'encoder_lstm_layers': encoder_lstm_layers,
            'reconstruction_lstm_layers': reconstruction_lstm_layers,
            'classification_fc_layers': classification_fc_layers,
            'learning_rate': learning_rate,
            'name': name
        }
        
        super().__init__(name=name)
        
        # Set default architectures
        if encoder_lstm_layers is None:
            encoder_lstm_layers = [100, 50, 25]
        if reconstruction_lstm_layers is None:
            reconstruction_lstm_layers = [25, 50, 100]
        if classification_fc_layers is None:
            classification_fc_layers = [8, 64, 32, 16]
        
        # Store parameters
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_lstm_layers = encoder_lstm_layers
        self.reconstruction_lstm_layers = reconstruction_lstm_layers
        self.classification_fc_layers = classification_fc_layers
        self.learning_rate = learning_rate
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build the complete model"""
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Encoder
        x = inputs
        for i, hidden_units in enumerate(self.encoder_lstm_layers):
            return_sequences = i < len(self.encoder_lstm_layers) - 1
            x = layers.LSTM(
                hidden_units,
                activation='tanh',
                return_sequences=return_sequences,
                name=f'encoder_lstm_{i+1}'
            )(x)
        
        # VAE latent space
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling(name="z")([z_mean, z_log_var])
        
        # Classification decoder (FCN)
        x_class = z
        for i, hidden_units in enumerate(self.classification_fc_layers):
            x_class = layers.Dense(
                hidden_units,
                activation='relu',
                name=f'classification_fc_{i+1}'
            )(x_class)
            x_class = layers.Dropout(0.2, name=f'classification_dropout_{i+1}')(x_class)
        
        # Output layer for classification
        classification_output = layers.Dense(
            2,
            activation='sigmoid',
            name='classification_output'
        )(x_class)
        
        # Create model
        self.model = Model(
            inputs=inputs,
            outputs=classification_output,
            name=self.name
        )
        
        # Store latent variables for loss computation
        self.z_mean = z_mean
        self.z_log_var = z_log_var
    
    def call(self, inputs, training=None):
        """Forward pass"""
        return self.model(inputs, training=training)
    
    def compile(self, **kwargs):
        """Compile the model"""
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = keras.optimizers.Adam(learning_rate=self.learning_rate)
        if 'loss' not in kwargs:
            kwargs['loss'] = 'binary_crossentropy'
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['accuracy', 'precision', 'recall']
        
        self.model.compile(**kwargs)
        return super().compile(**kwargs)
    
    def fit(self, x, y, **kwargs):
        """Fit the model"""
        return self.model.fit(x, y, **kwargs)
    
    def predict(self, x, **kwargs):
        """Predict using the model"""
        return self.model.predict(x, **kwargs)
    
    def evaluate(self, x, y, **kwargs):
        """Evaluate the model"""
        return self.model.evaluate(x, y, **kwargs)
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_lstm_layers': self.encoder_lstm_layers,
            'reconstruction_lstm_layers': self.reconstruction_lstm_layers,
            'classification_fc_layers': self.classification_fc_layers,
            'learning_rate': self.learning_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration"""
        return cls(**config)


def build_tvae_model(input_shape: Tuple[int, int],
                    latent_dim: int = 8,
                    encoder_lstm_layers: List[int] = None,
                    reconstruction_lstm_layers: List[int] = None,
                    classification_fc_layers: List[int] = None,
                    learning_rate: float = 0.001,
                    **kwargs) -> TVAE:
    """
    Build and compile TVAE model
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        latent_dim: Dimension of latent space
        encoder_lstm_layers: List of LSTM hidden units for encoder
        reconstruction_lstm_layers: List of LSTM hidden units for reconstruction decoder
        classification_fc_layers: List of FC hidden units for classification decoder
        learning_rate: Learning rate for optimizer
        **kwargs: Additional arguments for TVAE model
        
    Returns:
        Compiled TVAE model
    """
    # Create model
    model = TVAE(
        input_shape=input_shape,
        latent_dim=latent_dim,
        encoder_lstm_layers=encoder_lstm_layers,
        reconstruction_lstm_layers=reconstruction_lstm_layers,
        classification_fc_layers=classification_fc_layers,
        learning_rate=learning_rate,
        **kwargs
    )
    
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
        ]
    )
    
    return model


def get_model_summary(model: TVAE) -> str:
    """
    Get model summary
    
    Args:
        model: TVAE model
        
    Returns:
        Model summary string
    """
    return model.model.summary()


def save_model(model: TVAE, filepath: str) -> None:
    """
    Save TVAE model
    
    Args:
        model: TVAE model
        filepath: Path to save model
    """
    model.model.save_weights(filepath)


def load_model(filepath: str, input_shape: Tuple[int, int], **kwargs) -> TVAE:
    """
    Load TVAE model
    
    Args:
        filepath: Path to saved model
        input_shape: Input shape for model
        **kwargs: Additional arguments for TVAE model
        
    Returns:
        Loaded TVAE model
    """
    model = TVAE(input_shape=input_shape, **kwargs)
    model.model.load_weights(filepath)
    return model
