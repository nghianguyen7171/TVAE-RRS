"""
Evaluation module for TVAE-RRS
Includes metrics calculation, visualization, and model evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings


class ModelEvaluator:
    """
    Model evaluation class for TVAE-RRS
    """
    
    def __init__(self, 
                 primary_metrics: List[str] = None,
                 secondary_metrics: List[str] = None,
                 threshold_optimization: str = "youden"):
        """
        Initialize ModelEvaluator
        
        Args:
            primary_metrics: List of primary metrics to calculate
            secondary_metrics: List of secondary metrics to calculate
            threshold_optimization: Method for threshold optimization
        """
        if primary_metrics is None:
            primary_metrics = ["auroc", "auprc", "f1", "kappa"]
        if secondary_metrics is None:
            secondary_metrics = ["precision", "recall", "specificity"]
        
        self.primary_metrics = primary_metrics
        self.secondary_metrics = secondary_metrics
        self.threshold_optimization = threshold_optimization
        
        self.results = {}
        self.best_thresholds = {}
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred_proba: np.ndarray,
                         thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary of metrics
        """
        if thresholds is None:
            thresholds = [0.85, 0.90, 0.95, 0.99]
        
        metrics = {}
        
        # Primary metrics
        if "auroc" in self.primary_metrics:
            metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
        
        if "auprc" in self.primary_metrics:
            metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
        
        # Calculate metrics for different thresholds
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            if "f1" in self.primary_metrics:
                metrics[f"f1_threshold_{threshold}"] = f1_score(y_true, y_pred)
            
            if "kappa" in self.primary_metrics:
                metrics[f"kappa_threshold_{threshold}"] = cohen_kappa_score(y_true, y_pred)
            
            if "precision" in self.secondary_metrics:
                metrics[f"precision_threshold_{threshold}"] = precision_score(y_true, y_pred)
            
            if "recall" in self.secondary_metrics:
                metrics[f"recall_threshold_{threshold}"] = recall_score(y_true, y_pred)
            
            if "specificity" in self.secondary_metrics:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics[f"specificity_threshold_{threshold}"] = tn / (tn + fp)
        
        # Optimal threshold metrics
        optimal_threshold = self._find_optimal_threshold(y_true, y_pred_proba)
        y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
        
        metrics["optimal_threshold"] = optimal_threshold
        metrics["f1_optimal"] = f1_score(y_true, y_pred_optimal)
        metrics["precision_optimal"] = precision_score(y_true, y_pred_optimal)
        metrics["recall_optimal"] = recall_score(y_true, y_pred_optimal)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
        metrics["specificity_optimal"] = tn / (tn + fp)
        metrics["kappa_optimal"] = cohen_kappa_score(y_true, y_pred_optimal)
        
        return metrics
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Find optimal threshold based on specified method
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        if self.threshold_optimization == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            return thresholds[optimal_idx]
        
        elif self.threshold_optimization == "f1":
            thresholds = np.arange(0.1, 1.0, 0.01)
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred))
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        
        elif self.threshold_optimization == "precision_recall_curve":
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        
        else:
            raise ValueError(f"Unknown threshold optimization method: {self.threshold_optimization}")
    
    def calculate_late_alarm_rate(self, 
                                 y_true: np.ndarray, 
                                 y_pred_proba: np.ndarray,
                                 thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate late alarm rate for different thresholds
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary of late alarm rates
        """
        if thresholds is None:
            thresholds = [0.85, 0.90, 0.95, 0.99]
        
        late_alarm_rates = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Late alarm rate = False Negatives / (False Negatives + True Positives)
            late_alarm_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            late_alarm_rates[f"late_alarm_rate_threshold_{threshold}"] = late_alarm_rate
        
        return late_alarm_rates
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None,
                      title: str = "ROC Curve") -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
            title: Plot title
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark optimal thresholds
        thresholds_to_mark = [0.85, 0.90, 0.95, 0.99]
        colors = ['red', 'green', 'blue', 'yellow']
        
        for threshold, color in zip(thresholds_to_mark, colors):
            y_pred = (y_pred_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr_thresh = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_thresh = fp / (fp + tn) if (fp + tn) > 0 else 0
            plt.plot(fpr_thresh, tpr_thresh, 'o', color=color, markersize=8, 
                    label=f'TNR>{threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray, 
                                   y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None,
                                   title: str = "Precision-Recall Curve") -> None:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
            title: Plot title
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {ap_score:.3f})')
        
        # Mark optimal thresholds
        thresholds_to_mark = [0.85, 0.90, 0.95, 0.99]
        colors = ['red', 'green', 'blue', 'yellow']
        
        for threshold, color in zip(thresholds_to_mark, colors):
            y_pred = (y_pred_proba > threshold).astype(int)
            precision_thresh = precision_score(y_true, y_pred)
            recall_thresh = recall_score(y_true, y_pred)
            plt.plot(recall_thresh, precision_thresh, 'o', color=color, markersize=8,
                    label=f'Threshold {threshold:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             save_path: Optional[str] = None,
                             title: str = "Confusion Matrix") -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne(self, 
                  X: np.ndarray, 
                  y: np.ndarray,
                  perplexity: int = 30,
                  n_iter: int = 1000,
                  save_path: Optional[str] = None,
                  title: str = "t-SNE Visualization") -> None:
        """
        Plot t-SNE visualization
        
        Args:
            X: Input features
            y: Labels
            perplexity: t-SNE perplexity
            n_iter: Number of iterations
            save_path: Path to save plot
            title: Plot title
        """
        # Reshape data if needed
        if len(X.shape) > 2:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X_reshaped)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dews_scores(self, 
                        y_true: np.ndarray, 
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
        plt.figure(figsize=(12, 6))
        
        # Plot distribution for each class
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Abnormal', color='red')
        plt.xlabel('DEWS Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
        plt.boxplot(data_to_plot, labels=['Normal', 'Abnormal'])
        plt.ylabel('DEWS Score')
        plt.title('Score Distribution Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, 
                       y_true: np.ndarray, 
                       y_pred_proba: np.ndarray,
                       model_name: str = "Model",
                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save report
            
        Returns:
            Report string
        """
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred_proba)
        late_alarm_rates = self.calculate_late_alarm_rate(y_true, y_pred_proba)
        
        # Generate report
        report = f"""
# {model_name} Evaluation Report

## Primary Metrics
- AUROC: {metrics.get('auroc', 'N/A'):.4f}
- AUPRC: {metrics.get('auprc', 'N/A'):.4f}
- F1 Score (Optimal): {metrics.get('f1_optimal', 'N/A'):.4f}
- Kappa Score (Optimal): {metrics.get('kappa_optimal', 'N/A'):.4f}

## Optimal Threshold Performance
- Optimal Threshold: {metrics.get('optimal_threshold', 'N/A'):.4f}
- Precision: {metrics.get('precision_optimal', 'N/A'):.4f}
- Recall: {metrics.get('recall_optimal', 'N/A'):.4f}
- Specificity: {metrics.get('specificity_optimal', 'N/A'):.4f}

## Late Alarm Rates
"""
        
        for threshold, rate in late_alarm_rates.items():
            report += f"- {threshold}: {rate:.4f}\n"
        
        report += f"""
## Threshold Performance
"""
        
        thresholds = [0.85, 0.90, 0.95, 0.99]
        for threshold in thresholds:
            f1_key = f"f1_threshold_{threshold}"
            precision_key = f"precision_threshold_{threshold}"
            recall_key = f"recall_threshold_{threshold}"
            specificity_key = f"specificity_threshold_{threshold}"
            
            if all(key in metrics for key in [f1_key, precision_key, recall_key, specificity_key]):
                report += f"""
### Threshold {threshold}
- F1 Score: {metrics[f1_key]:.4f}
- Precision: {metrics[precision_key]:.4f}
- Recall: {metrics[recall_key]:.4f}
- Specificity: {metrics[specificity_key]:.4f}
"""
        
        # Save report if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def compare_models(self, 
                      results: Dict[str, Dict[str, float]],
                      save_path: Optional[str] = None,
                      title: str = "Model Comparison") -> None:
        """
        Compare multiple models
        
        Args:
            results: Dictionary of model results
            save_path: Path to save plot
            title: Plot title
        """
        # Extract metrics for comparison
        metrics_to_compare = ['auroc', 'auprc', 'f1_optimal', 'kappa_optimal']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_compare):
            ax = axes[i]
            
            model_names = list(results.keys())
            metric_values = [results[model].get(metric, 0) for model in model_names]
            
            bars = ax.bar(model_names, metric_values)
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


def evaluate_model(model: Any,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  model_name: str = "Model",
                  output_dir: str = "results",
                  plot_results: bool = True) -> Dict[str, float]:
    """
    Evaluate a model comprehensively
    
    Args:
        model: Model to evaluate
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        output_dir: Output directory
        plot_results: Whether to plot results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]  # Take positive class probability
    else:
        y_pred_proba = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred_proba)
    late_alarm_rates = evaluator.calculate_late_alarm_rate(y_test, y_pred_proba)
    
    # Combine all metrics
    all_metrics = {**metrics, **late_alarm_rates}
    
    # Generate plots if requested
    if plot_results:
        evaluator.plot_roc_curve(y_test, y_pred_proba, 
                               save_path=os.path.join(output_dir, f'{model_name}_roc.png'))
        evaluator.plot_precision_recall_curve(y_test, y_pred_proba,
                                            save_path=os.path.join(output_dir, f'{model_name}_pr.png'))
        evaluator.plot_dews_scores(y_test, y_pred_proba,
                                  save_path=os.path.join(output_dir, f'{model_name}_dews.png'))
    
    # Generate report
    report = evaluator.generate_report(y_test, y_pred_proba, model_name,
                                     save_path=os.path.join(output_dir, f'{model_name}_report.txt'))
    
    # Save metrics
    import pickle
    with open(os.path.join(output_dir, f'{model_name}_metrics.pickle'), 'wb') as f:
        pickle.dump(all_metrics, f)
    
    return all_metrics
