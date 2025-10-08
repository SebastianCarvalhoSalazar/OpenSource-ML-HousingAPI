"""
Compare baseline and advanced models to determine which to promote to production.
Updated to work with Pipeline objects.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import joblib

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import calculate_regression_metrics, print_metrics
from src.preprocessing import prepare_train_test_split, prepare_features_target
from src.data_loader import load_boston_housing_data


class ModelComparator:
    """Compare multiple models and recommend best for production."""

    def __init__(self):
        """Initialize comparator."""
        self.models = {}
        self.results = {}

    def load_model(self, model_name: str, model_path: str):
        """
        Load a trained pipeline.
        Args:
            model_name: Name to identify the model
            model_path: Path to model file
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return False

        try:
            # Load pipeline
            pipeline = joblib.load(model_path)

            # Load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            self.models[model_name] = {
                'pipeline': pipeline,
                'path': model_path,
                'metadata': metadata
            }
            print(f"Loaded {model_name} from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False

    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all loaded models on test data.
        Args:
            X_test: Test features DataFrame
            y_test: Test target Series
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)

        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")

            # Get pipeline
            pipeline = model_info['pipeline']

            # Predict (pipeline handles preprocessing internally)
            y_pred = pipeline.predict(X_test)

            # Calculate metrics
            metrics = calculate_regression_metrics(y_test, y_pred)

            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'y_test': y_test
            }

            print_metrics(metrics, model_name)

    def compare_metrics(self) -> pd.DataFrame:
        """
        Create comparison table of all models.
        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'R² Score': metrics['r2_score'],
                'RMSE ($k)': metrics['rmse'],
                'MAE ($k)': metrics['mae'],
                'MAPE (%)': metrics['mape']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('R² Score', ascending=False)

        return df

    def plot_comparison(self, save_path: str = None):
        """
        Create comparison visualization.
        Args:
            save_path: Path to save figure
        """
        if len(self.results) < 2:
            print("Need at least 2 models to compare")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = list(self.results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        # 1. R² Score comparison
        r2_scores = [self.results[m]['metrics']['r2_score'] for m in models]
        axes[0, 0].barh(models, r2_scores, color=colors)
        axes[0, 0].set_xlabel('R² Score', fontsize=12)
        axes[0, 0].set_title('R² Score Comparison',
                             fontsize=14, fontweight='bold')
        axes[0, 0].axvline(x=0.7, color='red', linestyle='--',
                           alpha=0.5, label='Good threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. RMSE comparison
        rmse_scores = [self.results[m]['metrics']['rmse'] for m in models]
        axes[0, 1].barh(models, rmse_scores, color=colors)
        axes[0, 1].set_xlabel('RMSE ($k)', fontsize=12)
        axes[0, 1].set_title(
            'RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. MAE comparison
        mae_scores = [self.results[m]['metrics']['mae'] for m in models]
        axes[1, 0].barh(models, mae_scores, color=colors)
        axes[1, 0].set_xlabel('MAE ($k)', fontsize=12)
        axes[1, 0].set_title('MAE Comparison (Lower is Better)',
                             fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Prediction scatter for best model
        best_model = max(self.results.items(),
                         key=lambda x: x[1]['metrics']['r2_score'])[0]
        y_test = self.results[best_model]['y_test']
        y_pred = self.results[best_model]['predictions']

        axes[1, 1].scatter(y_test, y_pred, alpha=0.5,
                           edgecolors='k', label=f'{best_model}')
        axes[1, 1].plot([y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        'r--', lw=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('Actual Price ($k)', fontsize=12)
        axes[1, 1].set_ylabel('Predicted Price ($k)', fontsize=12)
        axes[1, 1].set_title(
            f'Best Model: {best_model}', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")

        plt.close()

    def recommend_model(self,
                        min_r2_improvement: float = 0.03,
                        max_complexity_increase: str = 'moderate') -> Dict:
        """
        Recommend which model to promote to production.
        Args:
            min_r2_improvement: Minimum R² improvement to justify complexity
            max_complexity_increase: Maximum acceptable complexity increase

        Returns:
            Recommendation dictionary
        """
        if 'Baseline' not in self.results:
            print("Warning: Baseline model not found for comparison")
            # Just return best model
            best_model = max(self.results.items(),
                             key=lambda x: x[1]['metrics']['r2_score'])
            return {
                'recommended_model': best_model[0],
                'reason': 'Best R² score (no baseline for comparison)',
                'metrics': best_model[1]['metrics']
            }

        baseline_r2 = self.results['Baseline']['metrics']['r2_score']
        baseline_rmse = self.results['Baseline']['metrics']['rmse']

        print("\n" + "="*60)
        print("MODEL RECOMMENDATION ANALYSIS")
        print("="*60)

        print(f"\nBaseline Performance:")
        print(f"  R² Score: {baseline_r2:.4f}")
        print(f"  RMSE: ${baseline_rmse:.3f}k")

        # Analyze each advanced model
        recommendations = []

        for model_name, result in self.results.items():
            if model_name == 'Baseline':
                continue

            metrics = result['metrics']
            r2_improvement = metrics['r2_score'] - baseline_r2
            rmse_improvement = baseline_rmse - metrics['rmse']

            print(f"\n{model_name}:")
            print(
                f"  R² Score: {metrics['r2_score']:.4f} ({r2_improvement:+.4f})")
            print(f"  RMSE: ${metrics['rmse']:.3f}k ({rmse_improvement:+.3f})")

            # Determine if improvement justifies complexity
            if r2_improvement >= min_r2_improvement:
                justification = f"Significant improvement: R² +{r2_improvement:.4f}, RMSE {rmse_improvement:+.3f}"
                recommendations.append({
                    'model': model_name,
                    'r2_improvement': r2_improvement,
                    'rmse_improvement': rmse_improvement,
                    'justification': justification,
                    'metrics': metrics
                })
            else:
                print(
                    f" *** Improvement not significant enough (+{r2_improvement:.4f} < {min_r2_improvement})")

        # Make final recommendation
        if not recommendations:
            recommendation = {
                'recommended_model': 'Baseline',
                'reason': f'No advanced model shows sufficient improvement (>{min_r2_improvement} R²). ' +
                'Baseline is simpler and interpretable.',
                'metrics': self.results['Baseline']['metrics'],
                'interpretability': 'High - Linear coefficients are directly interpretable'
            }
        else:
            # Choose best advanced model
            best_advanced = max(
                recommendations, key=lambda x: x['r2_improvement'])
            recommendation = {
                'recommended_model': best_advanced['model'],
                'reason': best_advanced['justification'],
                'metrics': best_advanced['metrics'],
                'r2_improvement_over_baseline': best_advanced['r2_improvement'],
                'interpretability': 'Lower - Tree-based ensemble model'
            }

        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)
        print(f"\nRecommended Model: {recommendation['recommended_model']}")
        print(f"\nReason: {recommendation['reason']}")
        print(f"\nMetrics:")
        for metric, value in recommendation['metrics'].items():
            if metric in ['r2_score']:
                print(f"  {metric}: {value:.4f}")
            elif metric in ['mape']:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: ${value:.3f}k")

        if 'interpretability' in recommendation:
            print(f"\nInterpretability: {recommendation['interpretability']}")

        return recommendation

    def save_comparison_report(self, output_path: str = 'models/comparison_report.json'):
        """
        Save comparison report to JSON.
        Args:
            output_path: Path to save report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(self.results.keys()),
            'metrics': {
                model_name: result['metrics']
                for model_name, result in self.results.items()
            },
            'recommendation': self.recommend_model()
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nComparison report saved to: {output_path}")


def main():
    """Main comparison workflow."""

    print("="*60)
    print("MODEL COMPARISON TOOL")
    print("="*60)

    # Initialize comparator
    comparator = ModelComparator()

    # Load models (pipelines)
    models_to_compare = [
        ('Baseline', 'models/baseline_linear_regression.pkl'),
        ('Random Forest', 'models/random_forest_optimized.pkl'),
        ('XGBoost', 'models/xgboost_optimized.pkl')
    ]

    loaded_count = 0
    for model_name, model_path in models_to_compare:
        if comparator.load_model(model_name, model_path):
            loaded_count += 1

    if loaded_count < 2:
        print("\nError: Need at least 2 models to compare.")
        print("Please train models first:")
        print("  python src/train_baseline.py")
        print("  python src/train_advanced.py")
        return

    # Load test data
    print("\nLoading test data...")
    df = load_boston_housing_data()
    _, test_df = prepare_train_test_split(df)

    # Prepare features and target
    X_test, y_test = prepare_features_target(test_df, target_column='PRICE')

    # Evaluate all models
    comparator.evaluate_all_models(X_test, y_test)

    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    comparison_df = comparator.compare_metrics()
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("\nComparison table saved to: models/model_comparison.csv")

    # Create visualization
    plot_path = f'models/figures/model_comparison_{datetime.now():%Y%m%d_%H%M%S}.png'
    comparator.plot_comparison(plot_path)

    # Get recommendation
    recommendation = comparator.recommend_model(
        min_r2_improvement=0.03
    )

    # Save report
    comparator.save_comparison_report()

    # Promote recommended model to production
    recommended_model_name = recommendation['recommended_model']
    if recommended_model_name in comparator.models:
        source_path = comparator.models[recommended_model_name]['path']
        production_path = 'models/production.pkl'

        import shutil
        shutil.copy2(source_path, production_path)
        print(f"Model promoted to production: {production_path}")

        # Copy metadata as well
        source_metadata = source_path.replace('.pkl', '_metadata.json')
        if os.path.exists(source_metadata):
            shutil.copy2(source_metadata, production_path.replace(
                '.pkl', '_metadata.json'))

        # Save production metadata
        production_metadata = {
            'model_name': recommended_model_name,
            'source_path': source_path,
            'promoted_at': datetime.now().isoformat(),
            'metrics': recommendation['metrics'],
            'reason': recommendation['reason']
        }

        with open('models/production_metadata.json', 'w') as f:
            json.dump(production_metadata, f, indent=2)

        print("Production metadata saved")


if __name__ == "__main__":
    main()
