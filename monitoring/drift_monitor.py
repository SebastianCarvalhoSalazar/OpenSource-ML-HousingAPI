"""
Drift monitoring system using Evidently for Boston Housing model.
Updated to work with Pipeline objects.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
import joblib

warnings.filterwarnings('ignore')

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
    DatasetMissingValuesMetric,
    RegressionQualityMetric
)
from evidently.pipeline.column_mapping import ColumnMapping

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

# Configuration
MONITORING_PATH = Path("monitoring_data")
MONITORING_PATH.mkdir(exist_ok=True)
REFERENCE_DATA_PATH = MONITORING_PATH / "reference_data.parquet"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))

class HousingDriftMonitor:
    """Monitor drift for housing price prediction model."""
    
    def __init__(self, model_path: str):
        """
        Initialize drift monitor.
        Args:
            model_path: Path to production model pipeline
        """
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        self.reference_data = None
        
        if REFERENCE_DATA_PATH.exists():
            self.reference_data = pd.read_parquet(REFERENCE_DATA_PATH)
            print(f"Reference data loaded: {len(self.reference_data)} samples")
        else:
            print("No reference data found. Run setup_monitoring.py first.")
    
    def create_reference_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Create reference dataset for drift comparison.
        Args:
            X: Features DataFrame
            y: Target Series
        """
        print("Creating reference data...")
        
        # Make predictions using pipeline
        y_pred = self.pipeline.predict(X)
        
        # Create reference dataframe
        reference_df = X.copy()
        reference_df['PRICE'] = y
        reference_df['prediction'] = y_pred
        reference_df['residual'] = y - y_pred
        reference_df['absolute_error'] = np.abs(reference_df['residual'])
        
        # Save
        reference_df.to_parquet(REFERENCE_DATA_PATH)
        self.reference_data = reference_df
        
        print(f"Reference data created: {len(reference_df)} samples")
        print(f"Saved to: {REFERENCE_DATA_PATH}")
        
        return reference_df
    
    def collect_production_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Collect and process production data.
        Args:
            X: Production features
            y: Production target
        Returns:
            Processed production dataframe
        """
        # Make predictions using pipeline
        y_pred = self.pipeline.predict(X)
        
        # Create production dataframe
        production_df = X.copy()
        production_df['PRICE'] = y
        production_df['prediction'] = y_pred
        production_df['residual'] = y - y_pred
        production_df['absolute_error'] = np.abs(production_df['residual'])
        production_df['timestamp'] = datetime.now()
        
        # Save production data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prod_file = MONITORING_PATH / f"production_data_{timestamp}.parquet"
        production_df.to_parquet(prod_file)
        
        print(f"Production data collected: {len(production_df)} samples")
        
        return production_df
    
    def generate_drift_report(self, production_data: pd.DataFrame) -> Dict:
        """
        Generate drift report comparing production vs reference data.
        Args:
            production_data: Recent production data
        Returns:
            Dictionary with drift metrics
        """
        if self.reference_data is None:
            raise ValueError("No reference data available. Run create_reference_data() first.")
        
        print("\n" + "="*60)
        print("DRIFT ANALYSIS")
        print("="*60)
        
        # Get feature columns (exclude target and derived columns)
        feature_cols = [col for col in self.reference_data.columns 
                       if col not in ['PRICE', 'prediction', 'residual', 
                                     'absolute_error', 'timestamp']]
        
        # Create drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ] + [ColumnDriftMetric(column_name=col) for col in feature_cols[:5]]
        )
        
        # Run report
        drift_report.run(
            reference_data=self.reference_data[feature_cols + ['PRICE']],
            current_data=production_data[feature_cols + ['PRICE']]
        )
        
        # Extract metrics
        report_dict = drift_report.as_dict()
        
        drift_metrics = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': False,
            'drift_share': 0.0,
            'drifted_features': []
        }
        
        # Parse drift results
        for metric in report_dict.get('metrics', []):
            metric_type = str(metric.get('metric', ''))
            
            if 'DatasetDriftMetric' in metric_type:
                result = metric.get('result', {})
                drift_metrics['dataset_drift_detected'] = result.get('dataset_drift', False)
                drift_metrics['drift_share'] = result.get('drift_share', 0.0)
                
                print(f"\nDataset Drift Detected: {'YES ⚠️' if drift_metrics['dataset_drift_detected'] else 'NO ✓'}")
                print(f"Drift Share: {drift_metrics['drift_share']:.2%}")
                print(f"Threshold: {DRIFT_THRESHOLD:.2%}")
            
            elif 'ColumnDriftMetric' in metric_type:
                result = metric.get('result', {})
                column_name = result.get('column_name', '')
                drift_detected = result.get('drift_detected', False)
                
                if drift_detected and column_name:
                    drift_metrics['drifted_features'].append(column_name)
                    print(f"  - Drift in '{column_name}'")
        
        # Save HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = MONITORING_PATH / f"drift_report_{timestamp}.html"
        drift_report.save_html(str(report_path))
        print(f"\nDrift report saved: {report_path}")
        
        return drift_metrics
    
    def generate_performance_report(self, production_data: pd.DataFrame) -> Dict:
        """
        Generate model performance report on production data.
        Args:
            production_data: Production data with actual labels
        Returns:
            Performance metrics dictionary
        """
        print("\n" + "="*60)
        print("PERFORMANCE MONITORING")
        print("="*60)
        
        column_mapping = ColumnMapping(
            target='PRICE',
            prediction='prediction',
            numerical_features=[col for col in production_data.columns if col not in ['PRICE', 'prediction', 'residual', 'absolute_error', 'timestamp']],
            categorical_features=[]
        )        

        # Create performance report
        performance_report = Report(metrics=[
            RegressionPreset(),
            RegressionQualityMetric()
        ])
        
        # Run report
        performance_report.run(
            reference_data=None,
            current_data=production_data,
            column_mapping=column_mapping
        )
        
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'mean_error': float(production_data['residual'].mean()),
            'mean_absolute_error': float(production_data['absolute_error'].mean()),
            'rmse': float(np.sqrt(np.mean(production_data['residual']**2)))
        }
        
        print(f"\nProduction Performance:")
        print(f"  MAE: ${performance_metrics['mean_absolute_error']:.3f}k")
        print(f"  RMSE: ${performance_metrics['rmse']:.3f}k")
        print(f"  Mean Error: ${performance_metrics['mean_error']:.3f}k")
        
        # Save HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = MONITORING_PATH / f"performance_report_{timestamp}.html"
        performance_report.save_html(str(report_path))
        print(f"\nPerformance report saved: {report_path}")
        
        return performance_metrics
    
    def should_retrain(self, drift_metrics: Dict, performance_metrics: Dict) -> bool:
        """
        Determine if model needs retraining.
        Args:
            drift_metrics: Drift detection results
            performance_metrics: Performance metrics

        Returns:
            True if retraining is recommended
        """
        print("\n" + "="*60)
        print("RETRAINING RECOMMENDATION")
        print("="*60)
        
        reasons = []
        
        # Check drift
        if drift_metrics['drift_share'] > DRIFT_THRESHOLD:
            reasons.append(f"Drift share ({drift_metrics['drift_share']:.2%}) exceeds threshold ({DRIFT_THRESHOLD:.2%})")
        
        # Check performance degradation
        if 'test_metrics' in self.metadata:
            baseline_mae = self.metadata['test_metrics'].get('mae', 0)
            if performance_metrics['mean_absolute_error'] > baseline_mae * 1.2:
                reasons.append(f"MAE increased by >20% (${performance_metrics['mean_absolute_error']:.3f}k vs ${baseline_mae:.3f}k)")
        
        needs_retrain = len(reasons) > 0
        
        if needs_retrain:
            print("\n⚠️  RETRAINING RECOMMENDED")
            print("\nReasons:")
            for reason in reasons:
                print(f"  - {reason}")
        else:
            print("\n✓ Model performance is stable. No retraining needed.")
        
        return needs_retrain


def main():
    """Test drift monitoring."""
    from src.data_loader import load_boston_housing_data
    from src.preprocessing import prepare_train_test_split, prepare_features_target
    
    # Check if production model exists
    production_model = 'models/production.pkl'
    if not os.path.exists(production_model):
        print(f"Error: Production model not found at {production_model}")
        print("Please run model comparison first to promote a model to production.")
        return
    
    # Initialize monitor
    monitor = HousingDriftMonitor(production_model)
    
    # Simulate production data
    df = load_boston_housing_data()
    production_sample = df.sample(n=min(200, len(df)), random_state=99)
    
    X_prod, y_prod = prepare_features_target(production_sample, target_column='PRICE')
    
    # Collect production data
    production_data = monitor.collect_production_data(X_prod, y_prod)
    
    # Check if we have reference data
    if monitor.reference_data is not None:
        # Generate drift report
        drift_metrics = monitor.generate_drift_report(production_data)
        
        # Generate performance report
        performance_metrics = monitor.generate_performance_report(production_data)
        
        # Check if retraining needed
        needs_retrain = monitor.should_retrain(drift_metrics, performance_metrics)
        
        # Save monitoring summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'drift_metrics': drift_metrics,
            'performance_metrics': performance_metrics,
            'needs_retrain': needs_retrain
        }
        
        summary_path = MONITORING_PATH / 'latest_monitoring_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Monitoring summary saved: {summary_path}")
    else:
        print("\nNo reference data available. Please run setup_monitoring.py first.")


if __name__ == "__main__":
    main()