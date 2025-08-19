import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any

def is_classification_task(y_true):
    """
    Automatically determine if the task is classification.
    Rules:
    - If y contains only integers and number of unique values is small relative to length → classification
    - If dtype is object, bool, or category → classification
    """
    y_series = pd.Series(y_true)
    if y_series.dtype.name in ["object", "category", "bool"]:
        return True
    # Check if it's all integers and discrete labels
    unique_vals = y_series.unique()
    if np.issubdtype(y_series.dtype, np.integer) and len(unique_vals) < max(20, len(y_series) * 0.05):
        return True
    return False

def calculate_classification_metrics(y_true, y_pred, y_prob=None, average='weighted') -> Dict[str, Any]:
    """
    Calculate common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for positive class (for ROC AUC).
        average (str): Averaging method for multi-class metrics.

    Returns:
        dict: Dictionary containing classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics["roc_auc"] = None
    
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    return metrics

def calculate_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Calculate common regression metrics.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred)
    }
    return metrics

def calculate_metrics_auto(y_true, y_pred, y_proba=None):
    """
    Automatically detect if task is classification or regression and compute metrics.
    """
    if is_classification_task(y_true):
        return {
            "task_type": "classification",
            **calculate_classification_metrics(y_true, y_pred, y_proba)
        }
    else:
        return {
            "task_type": "regression",
            **calculate_regression_metrics(y_true, y_pred)
        }