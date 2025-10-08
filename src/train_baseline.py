"""
Train baseline model (Linear Regression) for Boston Housing dataset.
Uses Pipeline for preprocessing + model.
"""
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_boston_housing_data
from src.utils import (
    calculate_regression_metrics,
    print_metrics,
    plot_predictions,
    ensure_directories
)
from src.preprocessing import (
    create_preprocessing_pipeline,
    prepare_train_test_split,
    prepare_features_target,
    print_missing_values_info
)

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'boston_housing_regression')
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))


def train_baseline_model():
    """Train a simple Linear Regression baseline model using Pipeline."""

    print("="*60)
    print("TRAINING BASELINE MODEL: Linear Regression")
    print("="*60)

    # Ensure directories exist
    ensure_directories()

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run(run_name=f"baseline_linear_regression_{datetime.now():%Y%m%d_%H%M%S}"):

        # Log parameters
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("model_category", "baseline")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param(
            "preprocessing", "SimpleImputer(median) + RobustScaler")

        # 1. Load data
        print("\n1. Loading data...")
        df = load_boston_housing_data()
        mlflow.log_param("total_samples", len(df))
        mlflow.log_param("n_features", len(df.columns) - 1)

        # 2. Split data
        print("\n2. Splitting data...")
        train_df, test_df = prepare_train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))

        # 3. Prepare features and target
        print("\n3. Preparing features and target...")
        X_train, y_train = prepare_features_target(
            train_df, target_column='PRICE')
        X_test, y_test = prepare_features_target(
            test_df, target_column='PRICE')

        # Print missing values info
        print_missing_values_info(X_train)

        # Store feature names
        feature_names = X_train.columns.tolist()

        # 4. Create Pipeline with preprocessing + model
        print("\n4. Creating Pipeline (Preprocessing + Model)...")
        pipeline = Pipeline([
            ('preprocessing', create_preprocessing_pipeline(
                imputation_strategy='median')),
            ('model', LinearRegression())
        ])

        print("Pipeline steps:")
        for name, step in pipeline.named_steps.items():
            print(f"  - {name}: {step.__class__.__name__}")

        # 5. Train pipeline
        print("\n5. Training pipeline...")
        pipeline.fit(X_train, y_train)

        # Get model coefficients
        model = pipeline.named_steps['model']
        coefficients = dict(zip(feature_names, model.coef_))

        print("\nModel Coefficients:")
        print(f"Intercept: {model.intercept_:.4f}")
        mlflow.log_param("intercept", float(model.intercept_))

        sorted_coefs = sorted(coefficients.items(),
                              key=lambda x: abs(x[1]), reverse=True)
        for feature, coef in sorted_coefs:
            print(f"  {feature}: {coef:.4f}")
            mlflow.log_param(f"coef_{feature}", float(coef))

        # 6. Evaluate on training set
        print("\n6. Evaluating on training set...")
        y_train_pred = pipeline.predict(X_train)
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)

        print_metrics(train_metrics, "Training Set")
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        # 7. Evaluate on test set
        print("\n7. Evaluating on test set...")
        y_test_pred = pipeline.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)

        print_metrics(test_metrics, "Test Set")
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Check for overfitting
        r2_diff = train_metrics['r2_score'] - test_metrics['r2_score']
        mlflow.log_metric("r2_train_test_diff", r2_diff)

        if r2_diff > 0.1:
            print(
                f"Warning: Possible overfitting detected (R² diff: {r2_diff:.4f})")
        else:
            print(f"No significant overfitting (R² diff: {r2_diff:.4f})")

        # 8. Generate visualizations
        print("\n8. Generating visualizations...")
        plot_path = f'models/figures/baseline_predictions_{datetime.now():%Y%m%d_%H%M%S}.png'
        plot_predictions(y_test, y_test_pred,
                         "Baseline Linear Regression", plot_path)
        mlflow.log_artifact(plot_path)

        # 9. Save pipeline (preprocessing + model together)
        print("\n9. Saving pipeline...")
        model_path = 'models/baseline_linear_regression.pkl'

        # Save complete pipeline
        joblib.dump(pipeline, model_path)

        # Save metadata separately
        metadata = {
            'model_type': 'linear_regression',
            'model_category': 'baseline',
            'feature_names': feature_names,
            'coefficients': coefficients,
            'intercept': float(model.intercept_),
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'preprocessing_steps': ['SimpleImputer(median)', 'RobustScaler'],
            'timestamp': datetime.now().isoformat()
        }

        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Pipeline saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="boston_housing_baseline",
            input_example=X_train.head(5)
        )

        # 10. Model interpretation
        print("\n10. Model Interpretation:")
        print("\nTop 3 Most Important Features (by absolute coefficient):")
        for i, (feature, coef) in enumerate(sorted_coefs[:3], 1):
            direction = "increases" if coef > 0 else "decreases"
            print(
                f"  {i}. {feature}: ${abs(coef):.4f}k {direction} per unit increase")

        # Tags
        mlflow.set_tag("model_type", "linear_regression")
        mlflow.set_tag("model_category", "baseline")
        mlflow.set_tag("dataset", "california_housing")
        mlflow.set_tag("interpretable", "yes")
        mlflow.set_tag("uses_pipeline", "yes")

        print("\n" + "="*60)
        print(" - BASELINE MODEL TRAINING COMPLETED")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"\nKey Metrics:")
        print(f"  - R² Score: {test_metrics['r2_score']:.4f}")
        print(f"  - RMSE: ${test_metrics['rmse']:.3f}k")
        print(f"  - MAE: ${test_metrics['mae']:.3f}k")

        return pipeline, test_metrics


if __name__ == "__main__":
    train_baseline_model()
