"""
Evaluation package for TVAE-RRS
"""

from .evaluate_metrics import ModelEvaluator, evaluate_model
from .visualize_results import (
    plot_training_history,
    plot_model_comparison,
    plot_late_alarm_analysis,
    plot_feature_importance,
    plot_latent_space_visualization,
    plot_dews_score_distribution,
    plot_interactive_roc_curve,
    plot_cross_validation_results,
    create_dashboard
)
from .t_sne_latent import (
    visualize_latent_space_tvae,
    compare_latent_spaces,
    visualize_latent_evolution,
    analyze_latent_quality,
    visualize_reconstruction_quality
)

__all__ = [
    "ModelEvaluator",
    "evaluate_model",
    "plot_training_history",
    "plot_model_comparison",
    "plot_late_alarm_analysis",
    "plot_feature_importance",
    "plot_latent_space_visualization",
    "plot_dews_score_distribution",
    "plot_interactive_roc_curve",
    "plot_cross_validation_results",
    "create_dashboard",
    "visualize_latent_space_tvae",
    "compare_latent_spaces",
    "visualize_latent_evolution",
    "analyze_latent_quality",
    "visualize_reconstruction_quality",
]
