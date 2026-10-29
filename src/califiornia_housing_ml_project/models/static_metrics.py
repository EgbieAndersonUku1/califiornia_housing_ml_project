"""Provides standardized evaluation metric categories for validation scoring hooks.

This module acts as the centralisedd repository defining the exact string identifiers 
recognized by internal scoring engines and cross-validation pipelines. Grouping these literal 
keys into specialized sets guarantees strict framework compatibility checks before 
initializing model compilation routines.

Available Metric Profiles:
    - CLASSIFICATION_METRICS: Set containing standardized identifiers for tracking 
      discrete category outcomes, error penalties, and probability distributions.
    - REGRESSION_METRICS: Set containing continuous variable performance metrics, 
      including absolute, squared, and percentage-based variance scores.
"""

CLASSIFICATION_METRICS = {
    'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'f1', 'f1_micro', 
    'f1_macro', 'f1_weighted', 'f1_samples', 'precision', 'recall', 
    'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'log_loss', 'jaccard'
}

REGRESSION_METRICS = {
    'r2', 'explained_variance', 'neg_mean_absolute_error', 
    'neg_mean_squared_error', 'neg_root_mean_squared_error', 
    'neg_mean_squared_log_error', 'neg_median_absolute_error', 
    'neg_mean_absolute_percentage_error'
}

