"""
Data preprocessing utilities for Boston Housing dataset.
Creates reusable preprocessing pipelines.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple


def create_preprocessing_pipeline(imputation_strategy='median'):
    """
    Create a preprocessing pipeline with imputation and scaling.
    Args:
        imputation_strategy: Strategy for SimpleImputer ('mean', 'median', 'most_frequent')
    Returns:
        Pipeline with imputer and scaler
    """
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', RobustScaler())
    ])
    return preprocessing_pipeline


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    Args:
        df: DataFrame with features and target
        test_size: Proportion of test set
        random_state: Random seed
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    return train_df, test_df


def prepare_features_target(df: pd.DataFrame, target_column: str = 'PRICE',
                            drop_missing_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target, optionally dropping rows with missing target.
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        drop_missing_target: If True, drop rows where target is missing
    Returns:
        Tuple of (X, y)
    """
    # Drop rows with missing target if specified
    if drop_missing_target:
        missing_target = df[target_column].isnull().sum()
        if missing_target > 0:
            print(f"Dropping {missing_target} rows with missing target")
            df = df.dropna(subset=[target_column])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality metrics.
    Args:
        df: DataFrame to check

    Returns:
        Dictionary with quality metrics
    """
    missing_per_column = df.isnull().sum()
    missing_columns = missing_per_column[missing_per_column > 0].to_dict()
    quality_report = {
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,
        'missing_values_total': df.isnull().sum().sum(),
        'missing_values_per_column': missing_columns,
        'duplicate_rows': df.duplicated().sum()
    }
    print("\n=== Data Quality Report ===")
    print(f"Total samples: {quality_report['total_samples']}")
    print(f"Total features: {quality_report['total_features']}")
    print(f"Missing values (total): {quality_report['missing_values_total']}")

    if missing_columns:
        print(f"Missing values by column:")
        for col, count in missing_columns.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")

    print(f"Duplicate rows: {quality_report['duplicate_rows']}")
    return quality_report


def print_missing_values_info(X: pd.DataFrame):
    """
    Print information about missing values in features.
    Args:
        X: Features DataFrame
    """
    missing_total = X.isnull().sum().sum()
    if missing_total > 0:
        print(f"\nMissing values detected in features:")
        missing_per_feature = X.isnull().sum()
        for feature in missing_per_feature[missing_per_feature > 0].index:
            count = missing_per_feature[feature]
            pct = (count / len(X)) * 100
            print(f"  - {feature}: {count} ({pct:.2f}%)")
    else:
        print("\n✓ No missing values in features")
