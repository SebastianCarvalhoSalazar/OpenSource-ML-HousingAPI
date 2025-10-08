"""
Unit tests for model training and preprocessing.
Run with: pytest tests/test_models.py -v
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_boston_housing_data
from src.preprocessing import (
    prepare_train_test_split, 
    check_data_quality,
    create_preprocessing_pipeline,
    prepare_features_target
)
from src.utils import calculate_regression_metrics


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_data_returns_dataframe(self):
        """Test that data loading returns a DataFrame."""
        df = load_boston_housing_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_data_has_target_column(self):
        """Test that loaded data has PRICE target column."""
        df = load_boston_housing_data()
        assert 'PRICE' in df.columns
    
    def test_data_has_no_missing_values(self):
        """Test that data has no missing values."""
        df = load_boston_housing_data()
        assert df.isnull().sum().sum() == 0


class TestPreprocessing:
    """Test preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return load_boston_housing_data().head(100)
    
    def test_train_test_split(self, sample_data):
        """Test train-test split functionality."""
        train_df, test_df = prepare_train_test_split(sample_data, test_size=0.2)
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(sample_data)
    
    def test_prepare_features_target(self, sample_data):
        """Test feature and target separation."""
        X, y = prepare_features_target(sample_data, target_column='PRICE')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert 'PRICE' not in X.columns
    
    def test_preprocessing_pipeline_creation(self):
        """Test that preprocessing pipeline is created correctly."""
        pipeline = create_preprocessing_pipeline(imputation_strategy='median')
        
        # Check pipeline has correct steps
        assert hasattr(pipeline, 'steps')
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'imputer'
        assert pipeline.steps[1][0] == 'scaler'
    
    def test_preprocessing_pipeline_fit_transform(self, sample_data):
        """Test preprocessing pipeline fit and transform."""
        train_df, _ = prepare_train_test_split(sample_data)
        X_train, y_train = prepare_features_target(train_df, target_column='PRICE')
        
        pipeline = create_preprocessing_pipeline(imputation_strategy='median')
        X_transformed = pipeline.fit_transform(X_train)
        
        assert X_transformed.shape[0] == len(X_train)
        assert X_transformed.shape[1] == len(X_train.columns)
        assert isinstance(X_transformed, np.ndarray)
    
    def test_preprocessing_pipeline_scaling(self, sample_data):
        """Test that preprocessing pipeline scales data correctly (RobustScaler)."""
        train_df, _ = prepare_train_test_split(sample_data)
        X_train, y_train = prepare_features_target(train_df, target_column='PRICE')
        
        pipeline = create_preprocessing_pipeline(imputation_strategy='median')
        X_transformed = pipeline.fit_transform(X_train)
        
        # RobustScaler centers data around median (should be close to 0)
        # and scales by IQR
        medians = np.median(X_transformed, axis=0)
        assert np.abs(medians).max() < 1.0  # Medians should be close to 0
    
    def test_data_quality_check(self, sample_data):
        """Test data quality checking."""
        quality_report = check_data_quality(sample_data)
        
        assert 'total_samples' in quality_report
        assert 'total_features' in quality_report
        assert quality_report['total_samples'] == len(sample_data)


class TestPipeline:
    """Test complete pipeline with model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return load_boston_housing_data().head(100)
    
    def test_pipeline_with_model(self, sample_data):
        """Test pipeline with a simple model."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        
        # Split data
        train_df, test_df = prepare_train_test_split(sample_data, test_size=0.2)
        X_train, y_train = prepare_features_target(train_df, target_column='PRICE')
        X_test, y_test = prepare_features_target(test_df, target_column='PRICE')
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessing', create_preprocessing_pipeline(imputation_strategy='median')),
            ('model', LinearRegression())
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        assert len(y_pred) == len(y_test)
        assert y_pred.min() > 0  # Prices should be positive


class TestMetrics:
    """Test metrics calculation."""
    
    def test_metrics_calculation(self):
        """Test regression metrics calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2_score' in metrics
        assert metrics['r2_score'] >= 0  # Should be positive for reasonable predictions
    
    def test_perfect_prediction_metrics(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['r2_score'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])