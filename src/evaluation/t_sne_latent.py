"""
t-SNE visualization for TVAE-RRS latent space
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


def visualize_latent_space_tvae(model, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               method: str = "tsne",
                               perplexity: int = 30,
                               n_iter: int = 1000,
                               save_path: Optional[str] = None,
                               title: str = "TVAE Latent Space Visualization") -> np.ndarray:
    """
    Visualize TVAE latent space using t-SNE or PCA
    
    Args:
        model: Trained TVAE model
        X: Input data
        y: Labels
        method: Visualization method ('tsne' or 'pca')
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        Embedded coordinates
    """
    # Extract latent features
    if hasattr(model, 'model'):
        # TVAE model
        encoder = model.model.layers[0]  # Get encoder
        latent_features = encoder.predict(X)
    else:
        # Direct model
        latent_features = model.predict(X)
    
    # Reshape if needed
    if len(latent_features.shape) > 2:
        latent_features = latent_features.reshape(latent_features.shape[0], -1)
    
    # Apply dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embedded = reducer.fit_transform(latent_features)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot with different colors for different classes
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return embedded


def compare_latent_spaces(models: dict,
                         X: np.ndarray,
                         y: np.ndarray,
                         method: str = "tsne",
                         save_path: Optional[str] = None,
                         title: str = "Latent Space Comparison") -> None:
    """
    Compare latent spaces of different models
    
    Args:
        models: Dictionary of trained models
        X: Input data
        y: Labels
        method: Visualization method
        save_path: Path to save plot
        title: Plot title
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i]
        
        # Extract latent features
        if hasattr(model, 'model'):
            encoder = model.model.layers[0]
            latent_features = encoder.predict(X)
        else:
            latent_features = model.predict(X)
        
        # Reshape if needed
        if len(latent_features.shape) > 2:
            latent_features = latent_features.reshape(latent_features.shape[0], -1)
        
        # Apply dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        
        embedded = reducer.fit_transform(latent_features)
        
        # Plot
        unique_labels = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=[colors[j]], label=f'Class {label}', alpha=0.7, s=30)
        
        ax.set_title(f'{model_name}')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_latent_evolution(model,
                              X: np.ndarray,
                              y: np.ndarray,
                              epochs: List[int],
                              save_path: Optional[str] = None,
                              title: str = "Latent Space Evolution") -> None:
    """
    Visualize how latent space evolves during training
    
    Args:
        model: TVAE model with training history
        X: Input data
        y: Labels
        epochs: List of epochs to visualize
        save_path: Path to save plot
        title: Plot title
    """
    n_epochs = len(epochs)
    fig, axes = plt.subplots(2, (n_epochs + 1) // 2, figsize=(5 * ((n_epochs + 1) // 2), 10))
    
    if n_epochs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, epoch in enumerate(epochs):
        ax = axes[i]
        
        # Load model weights from specific epoch
        # This would require saving model weights at each epoch
        # For now, we'll use the final model
        if hasattr(model, 'model'):
            encoder = model.model.layers[0]
            latent_features = encoder.predict(X)
        else:
            latent_features = model.predict(X)
        
        # Reshape if needed
        if len(latent_features.shape) > 2:
            latent_features = latent_features.reshape(latent_features.shape[0], -1)
        
        # Apply t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        embedded = reducer.fit_transform(latent_features)
        
        # Plot
        unique_labels = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=[colors[j]], label=f'Class {label}', alpha=0.7, s=30)
        
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(n_epochs, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_latent_quality(latent_features: np.ndarray,
                          y: np.ndarray,
                          save_path: Optional[str] = None,
                          title: str = "Latent Space Quality Analysis") -> None:
    """
    Analyze the quality of latent space representation
    
    Args:
        latent_features: Latent space features
        y: Labels
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Latent space distribution
    ax1 = axes[0, 0]
    ax1.hist(latent_features.flatten(), bins=50, alpha=0.7)
    ax1.set_title('Latent Space Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Class separation in latent space
    ax2 = axes[0, 1]
    unique_labels = np.unique(y)
    for i, label in enumerate(unique_labels):
        mask = y == label
        class_features = latent_features[mask]
        mean_features = np.mean(class_features, axis=0)
        ax2.plot(mean_features, label=f'Class {label}', marker='o')
    
    ax2.set_title('Mean Latent Features by Class')
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Mean Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Latent space variance
    ax3 = axes[1, 0]
    latent_var = np.var(latent_features, axis=0)
    ax3.bar(range(len(latent_var)), latent_var)
    ax3.set_title('Latent Space Variance by Dimension')
    ax3.set_xlabel('Latent Dimension')
    ax3.set_ylabel('Variance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation matrix
    ax4 = axes[1, 1]
    if latent_features.shape[1] <= 10:  # Only if not too many dimensions
        corr_matrix = np.corrcoef(latent_features.T)
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Latent Space Correlation Matrix')
        ax4.set_xlabel('Latent Dimension')
        ax4.set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'Too many dimensions\nfor correlation matrix', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Latent Space Correlation Matrix')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_reconstruction_quality(model,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   n_samples: int = 10,
                                   save_path: Optional[str] = None,
                                   title: str = "Reconstruction Quality") -> None:
    """
    Visualize reconstruction quality of TVAE model
    
    Args:
        model: Trained TVAE model
        X: Input data
        y: Labels
        n_samples: Number of samples to visualize
        save_path: Path to save plot
        title: Plot title
    """
    # Get random samples
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_samples = X[indices]
    y_samples = y[indices]
    
    # Get reconstructions
    if hasattr(model, 'model'):
        reconstructions = model.model.predict(X_samples)
        if isinstance(reconstructions, list):
            reconstructions = reconstructions[0]  # Get reconstruction output
    else:
        reconstructions = model.predict(X_samples)
    
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    for i in range(n_samples):
        # Original
        ax1 = axes[0, i]
        ax1.plot(X_samples[i], label='Original', alpha=0.7)
        ax1.set_title(f'Sample {i+1} (Class {y_samples[i]})')
        ax1.set_ylabel('Original')
        ax1.grid(True, alpha=0.3)
        
        # Reconstruction
        ax2 = axes[1, i]
        ax2.plot(reconstructions[i], label='Reconstruction', alpha=0.7)
        ax2.set_ylabel('Reconstruction')
        ax2.grid(True, alpha=0.3)
        
        # Difference
        ax3 = axes[2, i]
        diff = X_samples[i] - reconstructions[i]
        ax3.plot(diff, label='Difference', alpha=0.7)
        ax3.set_ylabel('Difference')
        ax3.set_xlabel('Time Step')
        ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
