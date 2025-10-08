"""
Utility functions for the project.
"""
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    Args:
        y_true: True target values
        y_pred: Predicted values
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'mape': float(mape)
    }
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print(f"\n=== {model_name} Metrics ===")
    print(f"MAE (Mean Absolute Error): ${metrics['mae']:.3f}k")
    print(f"RMSE (Root Mean Squared Error): ${metrics['rmse']:.3f}k")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     model_name: str, save_path: str = None):
    """
    Plot predicted vs actual values.
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($k)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($k)', fontsize=12)
    axes[0].set_title(f'{model_name}: Predicted vs Actual', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price ($k)', fontsize=12)
    axes[1].set_ylabel('Residuals ($k)', fontsize=12)
    axes[1].set_title(f'{model_name}: Residual Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.close()


def plot_feature_importance(model: Any, feature_names: list,
                            model_name: str, save_path: str = None):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        model_name: Name of the model
        save_path: Path to save figure
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} doesn't have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name}: Feature Importance', fontsize=14)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    plt.close()


def save_model_artifact(model: Any, preprocessor: Any,
                        metrics: Dict, model_path: str,
                        metadata: Dict = None):
    """
    Save model with metadata.
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        metrics: Model metrics
        model_path: Path to save model
        metadata: Additional metadata
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        'model': model,
        'preprocessor': preprocessor,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }

    joblib.dump(artifact, model_path)
    print(f"Model artifact saved to: {model_path}")

    # Save metrics separately as JSON for easy access
    metrics_path = model_path.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'timestamp': artifact['timestamp'],
            'metadata': artifact['metadata']
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def load_model_artifact(model_path: str) -> Tuple[Any, Any, Dict]:
    """
    Load model artifact.
    Args:
        model_path: Path to model file

    Returns:
        Tuple of (model, preprocessor, metrics)
    """
    artifact = joblib.load(model_path)

    return (
        artifact['model'],
        artifact['preprocessor'],
        artifact['metrics']
    )


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data',
        'models',
        'logs',
        'monitoring_data',
        'models/figures'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test utilities
    ensure_directories()

    # Test metrics calculation
    y_true = np.array([23, 25, 30, 35, 40])
    y_pred = np.array([22, 26, 29, 36, 41])

    metrics = calculate_regression_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Model")
