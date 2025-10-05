"""
Visualization utilities for TVAE-RRS
Includes plotting functions for results visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         title: str = "Training History") -> None:
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        if metric in history:
            ax = axes[i]
            ax.plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            ax.set_title(f'{metric.title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = None,
                         save_path: Optional[str] = None,
                         title: str = "Model Comparison") -> None:
    """
    Plot model comparison
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        save_path: Path to save plot
        title: Plot title
    """
    if metrics is None:
        metrics = ['auroc', 'auprc', 'f1_optimal', 'kappa_optimal']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        model_names = list(results.keys())
        metric_values = [results[model].get(metric, 0) for model in model_names]
        
        bars = ax.bar(model_names, metric_values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_late_alarm_analysis(results: Dict[str, Dict[str, float]],
                            thresholds: List[float] = None,
                            save_path: Optional[str] = None,
                            title: str = "Late Alarm Rate Analysis") -> None:
    """
    Plot late alarm rate analysis
    
    Args:
        results: Dictionary of model results
        thresholds: List of thresholds to analyze
        save_path: Path to save plot
        title: Plot title
    """
    if thresholds is None:
        thresholds = [0.85, 0.90, 0.95, 0.99]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        late_alarm_rates = []
        for threshold in thresholds:
            key = f'late_alarm_rate_threshold_{threshold}'
            rate = results[model_name].get(key, 0)
            late_alarm_rates.append(rate)
        
        ax.plot(thresholds, late_alarm_rates, marker='o', 
               label=model_name, color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Late Alarm Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_names: List[str],
                           importance_scores: np.ndarray,
                           top_n: int = 20,
                           save_path: Optional[str] = None,
                           title: str = "Feature Importance") -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        top_n: Number of top features to show
        save_path: Path to save plot
        title: Plot title
    """
    # Get top N features
    top_indices = np.argsort(importance_scores)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_scores, color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(score + 0.001, i, f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latent_space_visualization(latent_features: np.ndarray,
                                   labels: np.ndarray,
                                   method: str = "tsne",
                                   save_path: Optional[str] = None,
                                   title: str = "Latent Space Visualization") -> None:
    """
    Plot latent space visualization
    
    Args:
        latent_features: Latent space features
        labels: Labels for coloring
        method: Visualization method ('tsne', 'pca')
        save_path: Path to save plot
        title: Plot title
    """
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    embedded = reducer.fit_transform(latent_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_dews_score_distribution(y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                save_path: Optional[str] = None,
                                title: str = "DEWS Score Distribution") -> None:
    """
    Plot DEWS score distribution
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Histogram
    plt.subplot(1, 3, 1)
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Abnormal', color='red', density=True)
    plt.xlabel('DEWS Score')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    plt.subplot(1, 3, 2)
    data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
    plt.boxplot(data_to_plot, labels=['Normal', 'Abnormal'])
    plt.ylabel('DEWS Score')
    plt.title('Score Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Violin plot
    plt.subplot(1, 3, 3)
    data_for_violin = []
    labels_for_violin = []
    for i, (data, label) in enumerate(zip(data_to_plot, ['Normal', 'Abnormal'])):
        data_for_violin.extend(data)
        labels_for_violin.extend([label] * len(data))
    
    df_violin = pd.DataFrame({'Score': data_for_violin, 'Class': labels_for_violin})
    sns.violinplot(data=df_violin, x='Class', y='Score')
    plt.title('Score Distribution Violin Plot')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_interactive_roc_curve(y_true: np.ndarray,
                              y_pred_proba: np.ndarray,
                              save_path: Optional[str] = None,
                              title: str = "Interactive ROC Curve") -> None:
    """
    Plot interactive ROC curve using Plotly
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
        title: Plot title
    """
    try:
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # Random line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
    except ImportError:
        print("Plotly not available, skipping interactive plot")


def plot_cross_validation_results(cv_results: Dict[str, List[float]],
                                 save_path: Optional[str] = None,
                                 title: str = "Cross-Validation Results") -> None:
    """
    Plot cross-validation results
    
    Args:
        cv_results: Cross-validation results
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['train_loss', 'val_loss', 'train_auroc', 'val_auroc']
    
    for i, metric in enumerate(metrics):
        if metric in cv_results:
            ax = axes[i]
            values = cv_results[metric]
            ax.boxplot(values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_dashboard(results: Dict[str, Dict[str, float]],
                    save_path: Optional[str] = None,
                    title: str = "TVAE-RRS Dashboard") -> None:
    """
    Create comprehensive dashboard
    
    Args:
        results: Dictionary of model results
        save_path: Path to save dashboard
        title: Dashboard title
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('AUROC Comparison', 'AUPRC Comparison', 
                           'F1 Score Comparison', 'Late Alarm Rate'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        model_names = list(results.keys())
        
        # AUROC
        auroc_values = [results[model].get('auroc', 0) for model in model_names]
        fig.add_trace(go.Bar(x=model_names, y=auroc_values, name='AUROC'), row=1, col=1)
        
        # AUPRC
        auprc_values = [results[model].get('auprc', 0) for model in model_names]
        fig.add_trace(go.Bar(x=model_names, y=auprc_values, name='AUPRC'), row=1, col=2)
        
        # F1 Score
        f1_values = [results[model].get('f1_optimal', 0) for model in model_names]
        fig.add_trace(go.Bar(x=model_names, y=f1_values, name='F1 Score'), row=2, col=1)
        
        # Late Alarm Rate
        thresholds = [0.85, 0.90, 0.95, 0.99]
        for model_name in model_names:
            late_alarm_rates = []
            for threshold in thresholds:
                key = f'late_alarm_rate_threshold_{threshold}'
                rate = results[model_name].get(key, 0)
                late_alarm_rates.append(rate)
            
            fig.add_trace(go.Scatter(
                x=thresholds, y=late_alarm_rates,
                mode='lines+markers',
                name=model_name
            ), row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
    except ImportError:
        print("Plotly not available, skipping dashboard creation")
