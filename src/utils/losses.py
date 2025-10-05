"""
Loss functions for TVAE-RRS model
Includes VAE loss, clinical loss, and imbalance loss components
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import Dict, Optional, Tuple
import numpy as np


class VAELoss(keras.losses.Loss):
    """
    VAE Loss combining reconstruction and KL divergence losses
    """
    
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 kl_weight: float = 1.0,
                 beta: float = 1.0,
                 name: str = "vae_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.beta = beta
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute VAE loss
        
        Args:
            y_true: True values (not used in VAE loss, kept for compatibility)
            y_pred: Predicted values (not used in VAE loss, kept for compatibility)
            
        Returns:
            VAE loss tensor
        """
        # This is a placeholder - actual VAE loss is computed in the model
        # using z_mean, z_log_var, and reconstruction
        return tf.constant(0.0)


def compute_vae_loss(z_mean: tf.Tensor, 
                     z_log_var: tf.Tensor, 
                     reconstruction: tf.Tensor,
                     original_input: tf.Tensor,
                     reconstruction_weight: float = 1.0,
                     kl_weight: float = 1.0,
                     beta: float = 1.0) -> tf.Tensor:
    """
    Compute VAE loss components
    
    Args:
        z_mean: Mean of latent distribution
        z_log_var: Log variance of latent distribution
        reconstruction: Reconstructed input
        original_input: Original input data
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        beta: Beta parameter for beta-VAE
        
    Returns:
        Total VAE loss
    """
    # Reconstruction loss (MSE)
    reconstruction_loss = tf.reduce_mean(
        tf.square(original_input - reconstruction)
    )
    
    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=-1
    )
    kl_loss = tf.reduce_mean(kl_loss)
    
    # Total VAE loss
    total_loss = (reconstruction_weight * reconstruction_loss + 
                  kl_weight * beta * kl_loss)
    
    return total_loss


class ClinicalLoss(keras.losses.Loss):
    """
    Clinical-specific loss function for medical time series
    Incorporates clinical domain knowledge and temporal patterns
    """
    
    def __init__(self,
                 clinical_weight: float = 1.0,
                 temporal_weight: float = 0.5,
                 name: str = "clinical_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.clinical_weight = clinical_weight
        self.temporal_weight = temporal_weight
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute clinical loss
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Clinical loss tensor
        """
        # Binary cross-entropy loss
        bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Temporal consistency loss (penalize rapid changes in predictions)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            temporal_diff = tf.abs(y_pred[:, 1:] - y_pred[:, :-1])
            temporal_loss = tf.reduce_mean(temporal_diff)
        else:
            temporal_loss = tf.constant(0.0)
        
        # Total clinical loss
        total_loss = (self.clinical_weight * bce_loss + 
                      self.temporal_weight * temporal_loss)
        
        return total_loss


class ImbalanceLoss(keras.losses.Loss):
    """
    Loss function to handle class imbalance in medical data
    Uses focal loss and class weighting
    """
    
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 class_weights: Optional[Dict[int, float]] = None,
                 name: str = "imbalance_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss for imbalanced data
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Focal loss tensor
        """
        # Convert to binary if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred[:, 1]  # Take positive class probability
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute focal loss
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        focal_loss = alpha_t * tf.pow(1 - p_t, self.gamma) * tf.nn.binary_crossentropy(
            y_true, y_pred, from_logits=False
        )
        
        # Apply class weights
        if self.class_weights:
            weights = tf.where(tf.equal(y_true, 1), 
                              self.class_weights[1], 
                              self.class_weights[0])
            focal_loss = focal_loss * weights
        
        return tf.reduce_mean(focal_loss)


class CombinedLoss(keras.losses.Loss):
    """
    Combined loss function for TVAE model
    Combines VAE, clinical, and imbalance losses
    """
    
    def __init__(self,
                 vae_weight: float = 1.0,
                 clinical_weight: float = 1.0,
                 imbalance_weight: float = 1.0,
                 name: str = "combined_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.vae_weight = vae_weight
        self.clinical_weight = clinical_weight
        self.imbalance_weight = imbalance_weight
        
        # Initialize component losses
        self.clinical_loss = ClinicalLoss()
        self.imbalance_loss = ImbalanceLoss()
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute combined loss
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Combined loss tensor
        """
        # Clinical loss
        clinical_loss = self.clinical_loss(y_true, y_pred)
        
        # Imbalance loss
        imbalance_loss = self.imbalance_loss(y_true, y_pred)
        
        # Total combined loss (VAE loss is handled separately in the model)
        total_loss = (self.clinical_weight * clinical_loss + 
                      self.imbalance_weight * imbalance_loss)
        
        return total_loss


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Compute Dice coefficient for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient tensor
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Compute Dice loss (1 - Dice coefficient)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        smooth: Smoothing factor
        
    Returns:
        Dice loss tensor
    """
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def mean_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute mean accuracy across classes
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Mean accuracy tensor
    """
    y_true_label = K.argmax(y_true, axis=1)
    y_pred_label = K.argmax(y_pred, axis=1)
    cm = tf.math.confusion_matrix(y_true_label, y_pred_label)
    cm_norm = cm / tf.reshape(tf.reduce_sum(cm, axis=1), (-1, 1))
    zero_pos = tf.where(tf.math.is_nan(cm_norm))
    n_zero = tf.shape(zero_pos)[0]
    cm_norm = tf.tensor_scatter_nd_update(cm_norm, zero_pos, tf.zeros(n_zero, dtype=tf.double))
    mean_acc_val = tf.reduce_mean(tf.linalg.diag_part(cm_norm))
    return mean_acc_val


def get_loss_function(loss_type: str, **kwargs) -> keras.losses.Loss:
    """
    Get loss function by name
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'vae': VAELoss,
        'clinical': ClinicalLoss,
        'imbalance': ImbalanceLoss,
        'combined': CombinedLoss,
        'binary_crossentropy': keras.losses.BinaryCrossentropy,
        'categorical_crossentropy': keras.losses.CategoricalCrossentropy,
        'mse': keras.losses.MeanSquaredError,
        'mae': keras.losses.MeanAbsoluteError,
        'dice': dice_loss,
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)


def get_metrics() -> list:
    """
    Get list of metrics for model evaluation
    
    Returns:
        List of metric functions
    """
    return [
        'accuracy',
        'precision',
        'recall',
        tf.keras.metrics.AUC(name='auroc'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'),
        dice_coefficient,
        mean_accuracy,
    ]
