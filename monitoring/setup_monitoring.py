"""
Setup monitoring system with reference data.
Updated to work with Pipeline objects.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_boston_housing_data
from src.preprocessing import prepare_train_test_split, prepare_features_target
from monitoring.drift_monitor import HousingDriftMonitor


def setup_monitoring():
    """Initialize monitoring system with reference data."""
    
    print("="*60)
    print("SETTING UP MONITORING SYSTEM")
    print("="*60)
    
    # Check if production model exists
    production_model = 'models/production.pkl'
    if not os.path.exists(production_model):
        print(f"\nError: Production model not found at {production_model}")
        print("\nPlease run the following commands first:")
        print("  1. python src/train_baseline.py")
        print("  2. python src/train_advanced.py")
        print("  3. python src/model_comparator.py")
        return
    
    print(f"\n✓ Production model found: {production_model}")
    
    # Load data
    print("\nLoading data...")
    df = load_boston_housing_data()
    _, test_df = prepare_train_test_split(df)
    
    # Prepare features and target
    X_test, y_test = prepare_features_target(test_df, target_column='PRICE')
    
    # Initialize monitor
    print("\nInitializing drift monitor...")
    monitor = HousingDriftMonitor(production_model)
    
    # Create reference data from test set
    print("\nCreating reference data...")
    reference_df = monitor.create_reference_data(X_test, y_test)
    
    print("\n" + "="*60)
    print("✓ MONITORING SYSTEM SETUP COMPLETE")
    print("="*60)
    print(f"\nReference data: {len(reference_df)} samples")
    print(f"Location: monitoring_data/reference_data.parquet")
    print("\nYou can now run drift monitoring with:")
    print("  python monitoring/drift_monitor.py")


if __name__ == "__main__":
    setup_monitoring()