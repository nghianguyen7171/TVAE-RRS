"""
Temporal Variational Autoencoder (TVAE) Model for Rapid Response System
Implements the proposed TVAE architecture with 3-layer LSTM encoder, VAE latent space, and dual decoders
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
    Temporal Variational Autoencoder (TVAE) Model
    
    Architecture:
    - Encoder: 3-layer LSTM (100, 50, 25 hidden units)
    - VAE: Latent space with mean and log variance
    - Dual Decoders: LSTM reconstruction + FCN classification
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 latent_dim: int = 8,
                 encoder_lstm_layers: List[int] = None,
                 reconstruction_lstm_layers: List[int] = None,
                 classification_fc_layers: List[int] = None,
                 encoder_dropout: float = 0.2,
                 encoder_recurrent_dropout: float = 0.1,
                 classification_dropout: float = 0.2,
                 reconstruction_weight: float = 1.0,
                 classification_weight: float = 1.0,
                 kl_weight: float = 1.0,
                 clinical_weight: float = 1.0,
                 imbalance_weight: float = 1.0,
                 beta: float = 1.0,
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
            encoder_dropout: Dropout rate for encoder
            encoder_recurrent_dropout: Recurrent dropout rate for encoder
            classification_dropout: Dropout rate for classification decoder
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
            kl_weight: Weight for KL divergence loss
            clinical_weight: Weight for clinical loss
            imbalance_weight: Weight for imbalance loss
            beta: Beta parameter for beta-VAE
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
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
        self.encoder_dropout = encoder_dropout
        self.encoder_recurrent_dropout = encoder_recurrent_dropout
        self.classification_dropout = classification_dropout
        
        # Loss weights
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.kl_weight = kl_weight
        self.clinical_weight = clinical_weight
        self.imbalance_weight = imbalance_weight
        self.beta = beta
        
        # Build model components
        self._build_encoder()
        self._build_decoders()
        self._build_model()
        
        # Initialize metric trackers
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.classification_loss_tracker = keras.metrics.Mean(name='classification_loss')
    
    def _build_encoder(self):
        """Build the encoder network"""
        # Input layer
        self.encoder_inputs = layers.Input(shape=self.input_shape, name='encoder_input')
        
        # LSTM layers
        x = self.encoder_inputs
        for i, hidden_units in enumerate(self.encoder_lstm_layers):
            return_sequences = i < len(self.encoder_lstm_layers) - 1
            x = layers.LSTM(
                hidden_units,
                activation='tanh',
                return_sequences=return_sequences,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name=f'encoder_lstm_{i+1}'
            )(x)
        
        # VAE latent space
        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        self.z = Sampling(name="z")([self.z_mean, self.z_log_var])
    
    def _build_decoders(self):
        """Build the dual decoders"""
        # Reconstruction decoder (LSTM)
        self._build_reconstruction_decoder()
        
        # Classification decoder (FCN)
        self._build_classification_decoder()
    
    def _build_reconstruction_decoder(self):
        """Build LSTM reconstruction decoder"""
        # Start with latent vector
        decoder_input = layers.RepeatVector(self.input_shape[0])(self.z)
        
        # LSTM layers (reverse order of encoder)
        x = decoder_input
        for i, hidden_units in enumerate(self.reconstruction_lstm_layers):
            return_sequences = i < len(self.reconstruction_lstm_layers) - 1
            x = layers.LSTM(
                hidden_units,
                activation='tanh',
                return_sequences=return_sequences,
                dropout=self.encoder_dropout,
                recurrent_dropout=self.encoder_recurrent_dropout,
                name=f'reconstruction_lstm_{i+1}'
            )(x)
        
        # Output layer for reconstruction
        self.reconstruction_output = layers.Dense(
            self.input_shape[1],
            activation='linear',
            name='reconstruction_output'
        )(x)
    
    def _build_classification_decoder(self):
        """Build FCN classification decoder"""
        # Start with latent vector
        x = self.z
        
        # Fully connected layers
        for i, hidden_units in enumerate(self.classification_fc_layers):
            x = layers.Dense(
                hidden_units,
                activation='relu',
                name=f'classification_fc_{i+1}'
            )(x)
            x = layers.Dropout(
                self.classification_dropout,
                name=f'classification_dropout_{i+1}'
            )(x)
        
        # Output layer for classification
        self.classification_output = layers.Dense(
            2,
            activation='sigmoid',
            name='classification_output'
        )(x)
    
    def _build_model(self):
        """Build the complete model"""
        # Create model
        self.model = Model(
            inputs=self.encoder_inputs,
            outputs=[self.reconstruction_output, self.classification_output],
            name=self.name
        )
        
        # Loss trackers
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.classification_loss_tracker = keras.metrics.Mean(name='classification_loss')
    
    def call(self, inputs, training=None):
        """Forward pass"""
        return self.model(inputs, training=training)
    
    def compute_loss(self, x, y_true, y_pred):
        """
        Compute combined loss
        
        Args:
            x: Input data
            y_true: True labels
            y_pred: Predicted outputs (reconstruction, classification)
            
        Returns:
            Total loss
        """
        reconstruction_pred, classification_pred = y_pred
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var),
            axis=-1
        )
        kl_loss = tf.reduce_mean(kl_loss)
        
        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            tf.square(x - reconstruction_pred)
        )
        
        # Classification loss (Binary cross-entropy)
        classification_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(y_true, classification_pred)
        )
        
        # Clinical loss (temporal consistency)
        if len(classification_pred.shape) > 1 and classification_pred.shape[1] > 1:
            temporal_diff = tf.abs(classification_pred[:, 1:] - classification_pred[:, :-1])
            clinical_loss = tf.reduce_mean(temporal_diff)
        else:
            clinical_loss = tf.constant(0.0)
        
        # Imbalance loss (focal loss)
        alpha = 0.25
        gamma = 2.0
        y_pred_clipped = tf.clip_by_value(classification_pred, 1e-7, 1 - 1e-7)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        p_t = tf.where(tf.equal(y_true, 1), y_pred_clipped, 1 - y_pred_clipped)
        
        focal_loss = alpha_t * tf.pow(1 - p_t, gamma) * keras.losses.binary_crossentropy(
            y_true, classification_pred, from_logits=False
        )
        imbalance_loss = tf.reduce_mean(focal_loss)
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.classification_weight * classification_loss +
            self.kl_weight * self.beta * kl_loss +
            self.clinical_weight * clinical_loss +
            self.imbalance_weight * imbalance_loss
        )
        
        # Update metrics
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        
        return total_loss
    
    def train_step(self, data):
        """Custom training step"""
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred[1])  # Use classification output for metrics
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """Custom test step"""
        x, y = data
        
        y_pred = self(x, training=False)
        loss = self.compute_loss(x, y, y_pred)
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred[1])  # Use classification output for metrics
        
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'encoder_lstm_layers': self.encoder_lstm_layers,
            'reconstruction_lstm_layers': self.reconstruction_lstm_layers,
            'classification_fc_layers': self.classification_fc_layers,
            'encoder_dropout': self.encoder_dropout,
            'encoder_recurrent_dropout': self.encoder_recurrent_dropout,
            'classification_dropout': self.classification_dropout,
            'reconstruction_weight': self.reconstruction_weight,
            'classification_weight': self.classification_weight,
            'kl_weight': self.kl_weight,
            'clinical_weight': self.clinical_weight,
            'imbalance_weight': self.imbalance_weight,
            'beta': self.beta,
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
        **kwargs
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            'accuracy',
            'precision',
            'recall',
            keras.metrics.AUC(name='auroc'),
            keras.metrics.AUC(name='auprc', curve='PR'),
        ],
        run_eagerly=False
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
    model.save_weights(filepath)


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
    model.load_weights(filepath)
    return model
