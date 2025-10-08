"""
Train advanced models with hyperparameter optimization for Boston Housing dataset.
Uses Pipeline for preprocessing + model with Bayesian optimization.
"""
import os
import sys
import mlflow
import mlflow.sklearn
# import mlflow.xgboost
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from dotenv import load_dotenv
import warnings
import joblib

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_boston_housing_data
from src.preprocessing import (
    create_preprocessing_pipeline,
    prepare_train_test_split,
    prepare_features_target,
    print_missing_values_info
)
from src.utils import (
    calculate_regression_metrics,
    print_metrics,
    plot_predictions,
    plot_feature_importance,
    ensure_directories
)

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'boston_housing_regression')
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
CV_FOLDS = int(os.getenv('CV_FOLDS', '5'))


def train_random_forest_optimized(X_train, X_test, y_train, y_test, 
                                  feature_names, n_iter=30):
    """
    Train Random Forest with Bayesian optimization using Pipeline.
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        feature_names: List of feature names
        n_iter: Number of optimization iterations
    Returns:
        Trained pipeline and metrics
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST WITH BAYESIAN OPTIMIZATION")
    print("="*60)
    
    with mlflow.start_run(run_name=f"random_forest_optimized_{datetime.now():%Y%m%d_%H%M%S}"):
        
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("model_category", "advanced")
        mlflow.log_param("optimization", "bayesian")
        mlflow.log_param("n_optimization_iterations", n_iter)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("preprocessing", "SimpleImputer(median) + RobustScaler")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessing', create_preprocessing_pipeline(imputation_strategy='median')),
            ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
        ])
        
        # Define search space (note the 'model__' prefix for pipeline)
        search_space = {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(3, 5),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 10),
            'model__max_features': Real(0.3, 1.0),
            'model__bootstrap': Categorical([True, False])
        }
        
        # Bayesian optimization
        print(f"\nStarting Bayesian optimization with {n_iter} iterations...")
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        bayes_search.fit(X_train, y_train)
        
        # Best parameters
        best_params = bayes_search.best_params_
        best_score = -bayes_search.best_score_
        
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            mlflow.log_param(f"best_{param}", value)
        
        print(f"\nBest CV RMSE: ${np.sqrt(best_score):.3f}k")
        mlflow.log_metric("best_cv_rmse", np.sqrt(best_score))
        
        # Get best pipeline
        best_pipeline = bayes_search.best_estimator_
        
        # Evaluate on training set
        y_train_pred = best_pipeline.predict(X_train)
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        
        print_metrics(train_metrics, "Training Set")
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        
        # Evaluate on test set
        y_test_pred = best_pipeline.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)
        
        print_metrics(test_metrics, "Test Set")
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        
        # Check overfitting
        r2_diff = train_metrics['r2_score'] - test_metrics['r2_score']
        mlflow.log_metric("r2_train_test_diff", r2_diff)
        
        if r2_diff > 0.1:
            print(f"Warning: Possible overfitting (R² diff: {r2_diff:.4f})")
        
        # Visualizations
        print("\nGenerating visualizations...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Predictions plot
        plot_path = f'models/figures/rf_predictions_{timestamp}.png'
        plot_predictions(y_test, y_test_pred, "Random Forest (Optimized)", plot_path)
        mlflow.log_artifact(plot_path)
        
        # Feature importance
        model = best_pipeline.named_steps['model']
        importance_path = f'models/figures/rf_feature_importance_{timestamp}.png'
        plot_feature_importance(model, feature_names, "Random Forest", importance_path)
        mlflow.log_artifact(importance_path)
        
        # Save pipeline
        model_path = 'models/random_forest_optimized.pkl'
        joblib.dump(best_pipeline, model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'random_forest',
            'model_category': 'advanced',
            'best_params': best_params,
            'optimization_method': 'bayesian',
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'preprocessing_steps': ['SimpleImputer(median)', 'RobustScaler'],
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Pipeline saved to: {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="boston_housing_random_forest",
            input_example=X_train.head(5)
        )
        
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("model_category", "advanced")
        mlflow.set_tag("optimization", "bayesian")
        mlflow.set_tag("uses_pipeline", "yes")
        
        return best_pipeline, test_metrics


def train_xgboost_optimized(X_train, X_test, y_train, y_test, 
                           feature_names, n_iter=30):
    """
    Train XGBoost with Bayesian optimization using Pipeline.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        feature_names: List of feature names
        n_iter: Number of optimization iterations
        
    Returns:
        Trained pipeline and metrics
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST WITH BAYESIAN OPTIMIZATION")
    print("="*60)
    
    with mlflow.start_run(run_name=f"xgboost_optimized_{datetime.now():%Y%m%d_%H%M%S}"):
        
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("model_category", "advanced")
        mlflow.log_param("optimization", "bayesian")
        mlflow.log_param("n_optimization_iterations", n_iter)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("preprocessing", "SimpleImputer(median) + RobustScaler")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessing', create_preprocessing_pipeline(imputation_strategy='median')),
            ('model', XGBRegressor(
                objective='reg:squarederror',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
        
        # Define search space
        search_space = {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(3, 10),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__subsample': Real(0.6, 1.0),
            'model__colsample_bytree': Real(0.6, 1.0),
            'model__min_child_weight': Integer(1, 10),
            'model__gamma': Real(0, 5),
            'model__reg_alpha': Real(0, 10),
            'model__reg_lambda': Real(0, 10)
        }
        
        # Bayesian optimization
        print(f"\nStarting Bayesian optimization with {n_iter} iterations...")
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        bayes_search.fit(X_train, y_train)
        
        # Best parameters
        best_params = bayes_search.best_params_
        best_score = -bayes_search.best_score_
        
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            mlflow.log_param(f"best_{param}", value)
        
        print(f"\nBest CV RMSE: ${np.sqrt(best_score):.3f}k")
        mlflow.log_metric("best_cv_rmse", np.sqrt(best_score))
        
        # Get best pipeline
        best_pipeline = bayes_search.best_estimator_
        
        # Evaluate on training set
        y_train_pred = best_pipeline.predict(X_train)
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        
        print_metrics(train_metrics, "Training Set")
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        
        # Evaluate on test set
        y_test_pred = best_pipeline.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)
        
        print_metrics(test_metrics, "Test Set")
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        
        # Check overfitting
        r2_diff = train_metrics['r2_score'] - test_metrics['r2_score']
        mlflow.log_metric("r2_train_test_diff", r2_diff)
        
        if r2_diff > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting (R² diff: {r2_diff:.4f})")
        
        # Visualizations
        print("\nGenerating visualizations...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Predictions plot
        plot_path = f'models/figures/xgb_predictions_{timestamp}.png'
        plot_predictions(y_test, y_test_pred, "XGBoost (Optimized)", plot_path)
        mlflow.log_artifact(plot_path)
        
        # Feature importance
        model = best_pipeline.named_steps['model']
        importance_path = f'models/figures/xgb_feature_importance_{timestamp}.png'
        plot_feature_importance(model, feature_names, "XGBoost", importance_path)
        mlflow.log_artifact(importance_path)
        
        # Save pipeline
        model_path = 'models/xgboost_optimized.pkl'
        joblib.dump(best_pipeline, model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'xgboost',
            'model_category': 'advanced',
            'best_params': best_params,
            'optimization_method': 'bayesian',
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'preprocessing_steps': ['SimpleImputer(median)', 'RobustScaler'],
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Pipeline saved to: {model_path}")
        
        # # Log model to MLflow
        # mlflow.xgboost.log_model(
        #     xgb_model=best_pipeline,
        #     artifact_path="model",
        #     registered_model_name="boston_housing_xgboost"
        # )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="boston_housing_xgboost_pipeline",
            input_example=X_train.head(5)
        )

        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("model_category", "advanced")
        mlflow.set_tag("optimization", "bayesian")
        mlflow.set_tag("uses_pipeline", "yes")
        
        return best_pipeline, test_metrics


def train_all_advanced_models():
    """Train all advanced models and compare."""
    
    print("="*60)
    print("TRAINING ALL ADVANCED MODELS")
    print("="*60)
    
    # Ensure directories
    ensure_directories()
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    df = load_boston_housing_data()
    train_df, test_df = prepare_train_test_split(df, TEST_SIZE, RANDOM_STATE)
    
    # Prepare features and target
    X_train, y_train = prepare_features_target(train_df, target_column='PRICE')
    X_test, y_test = prepare_features_target(test_df, target_column='PRICE')
    
    # Print missing values info
    print_missing_values_info(X_train)
    
    feature_names = X_train.columns.tolist()
    
    results = {}
    
    # Train Random Forest
    print("\n" + "="*60)
    print("1/2: RANDOM FOREST")
    print("="*60)
    rf_pipeline, rf_metrics = train_random_forest_optimized(
        X_train, X_test, y_train, y_test, feature_names, n_iter=15
    )
    results['Random Forest'] = rf_metrics
    
    # Train XGBoost
    print("\n" + "="*60)
    print("2/2: XGBOOST")
    print("="*60)
    xgb_pipeline, xgb_metrics = train_xgboost_optimized(
        X_train, X_test, y_train, y_test, feature_names, n_iter=15
    )
    results['XGBoost'] = xgb_metrics
    
    # Summary
    print("\n" + "="*60)
    print("ADVANCED MODELS SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  RMSE: ${metrics['rmse']:.3f}k")
        print(f"  MAE: ${metrics['mae']:.3f}k")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
    print(f"\n✓ Best Model: {best_model[0]} (R² = {best_model[1]['r2_score']:.4f})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train advanced models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'xgboost', 'all'],
        default='all',
        help='Model to train'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=15,
        help='Number of Bayesian optimization iterations'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_advanced_models()
    else:
        # Load data
        ensure_directories()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        df = load_boston_housing_data(use_california=True)
        train_df, test_df = prepare_train_test_split(df, TEST_SIZE, RANDOM_STATE)
        
        X_train, y_train = prepare_features_target(train_df, target_column='PRICE')
        X_test, y_test = prepare_features_target(test_df, target_column='PRICE')
        
        print_missing_values_info(X_train)
        feature_names = X_train.columns.tolist()
        
        if args.model == 'random_forest':
            train_random_forest_optimized(
                X_train, X_test, y_train, y_test, 
                feature_names, n_iter=args.n_iter
            )
        elif args.model == 'xgboost':
            train_xgboost_optimized(
                X_train, X_test, y_train, y_test, 
                feature_names, n_iter=args.n_iter
            )