import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
from preprocess import DataPreprocessor
from train import RandomForestTrainer


class ModelTester:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.predictions = None
        self.test_metrics = None

    def load_models(self, model_path: str = 'models/random_forest_model.pkl',
                    preprocessor_path: str = 'models/preprocessor.pkl'):
        """Load trained model and preprocessor"""
        # Load model
        trainer = RandomForestTrainer()
        trainer.load_model(model_path)
        self.model = trainer.model

        # Load preprocessor
        self.preprocessor = DataPreprocessor()

        print("Models loaded successfully!")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")

        self.predictions = self.model.predict(X_test)
        return self.predictions

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various regression metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'sum_of_absolute_errors': np.sum(np.abs(y_true - y_pred)),
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }

        return metrics

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        print("Evaluating model on test data...")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        self.test_metrics = self.calculate_metrics(y_test, y_pred)

        # Display results
        print("\nTest Set Performance:")
        print("-" * 40)
        for metric, value in self.test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return self.test_metrics

    def plot_predictions(self, y_test: pd.Series, y_pred: np.ndarray,
                         figsize: Tuple[int, int] = (12, 10)):
        """Plot various prediction analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Residuals)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_error_distribution(self, y_test: pd.Series, y_pred: np.ndarray):
        """Plot error distribution analysis"""
        errors = np.abs(y_test - y_pred)
        percentage_errors = (errors / np.abs(y_test)) * 100

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Absolute errors
        axes[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Absolute Error Distribution')
        axes[0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Percentage errors
        axes[1].hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Percentage Error Distribution')
        axes[1].axvline(np.mean(percentage_errors), color='red', linestyle='--',
                        label=f'Mean: {np.mean(percentage_errors):.2f}%')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plt.savefig('plots/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def prediction_intervals(self, X_test: pd.DataFrame, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals using bootstrap"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")

        print(f"Calculating {confidence * 100}% prediction intervals...")

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in self.model.estimators_])

        # Calculate percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)

        return lower_bound, upper_bound


    def plot_prediction_intervals(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  confidence: float = 0.95, sample_size: int = 100):
        """Plot prediction intervals for a sample of test data"""
        # Get a sample for visualization
        sample_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]

        # Get predictions and intervals
        y_pred_sample = self.predict(X_sample)
        lower_bound, upper_bound = self.prediction_intervals(X_sample, confidence)

        # Sort by actual values for better visualization
        sort_idx = np.argsort(y_sample)
        x_axis = np.arange(len(y_sample))

        plt.figure(figsize=(12, 8))
        plt.plot(x_axis, y_sample.iloc[sort_idx], 'o', label='Actual', color='blue', markersize=6)
        plt.plot(x_axis, y_pred_sample[sort_idx], 'o', label='Predicted', color='red', markersize=6)
        plt.fill_between(x_axis, lower_bound[sort_idx], upper_bound[sort_idx],
                         alpha=0.3, color='gray', label=f'{confidence * 100}% Prediction Interval')

        plt.xlabel('Sample Index (sorted by actual value)')
        plt.ylabel('Target Value')
        plt.title(f'Prediction Intervals ({confidence * 100}% Confidence)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig('plots/prediction_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Return the indices for consistent use elsewhere if needed
        return sample_indices

    def compare_with_baseline(self, y_test: pd.Series, y_pred: np.ndarray):
        """Compare model performance with simple baselines, ensuring matching lengths"""
        # Align indices and lengths
        y_test = pd.Series(y_test).reset_index(drop=True)
        y_pred = np.array(y_pred).flatten()
        if len(y_test) != len(y_pred):
            min_len = min(len(y_test), len(y_pred))
            y_test = y_test.iloc[:min_len]
            y_pred = y_pred[:min_len]

        # Baseline 1: Mean prediction
        mean_baseline = np.full_like(y_pred, y_test.mean())

        # Baseline 2: Median prediction
        median_baseline = np.full_like(y_pred, y_test.median())

        # Calculate metrics for all models
        model_metrics = self.calculate_metrics(y_test, y_pred)
        mean_metrics = self.calculate_metrics(y_test, mean_baseline)
        median_metrics = self.calculate_metrics(y_test, median_baseline)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Random Forest': model_metrics,
            'Mean Baseline': mean_metrics,
            'Median Baseline': median_metrics
        })

        print("\nModel Comparison:")
        print("-" * 50)
        print(comparison_df.round(4))

        return comparison_df

    def test_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Run the complete testing pipeline"""
        print("Starting model testing...")

        # Evaluate model
        test_metrics = self.evaluate_model(X_test, y_test)

        # Plot predictions and errors
        self.plot_predictions(y_test, self.predictions)
        self.plot_error_distribution(y_test, self.predictions)
        self.plot_prediction_intervals(X_test, y_test)

        # Compare with baselines
        comparison_df = self.compare_with_baseline(y_test, self.predictions)

        return test_metrics, comparison_df


def main():
    """Main testing function"""
    # Initialize tester
    tester = ModelTester()

    # Load models
    tester.load_models()

    # Load test data (assuming it's already preprocessed)
    # In practice, you might need to load raw data and preprocess it
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        'data/dataset.csv', 'target', test_size=0.2, random_state=42
    )

    # Evaluate model
    test_metrics = tester.evaluate_model(X_test, y_test)

    # Generate plots
    tester.plot_predictions(y_test, tester.predictions)
    tester.plot_error_distribution(y_test, tester.predictions)
    tester.plot_prediction_intervals(X_test, y_test)

    # Compare with baselines
    comparison_df = tester.compare_with_baseline(y_test, tester.predictions)

    # Save results
    os.makedirs('results', exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv('results/test_metrics.csv', index=False)

    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv')

    print("\nTesting completed successfully!")
    print("Results saved to 'results/' directory")
    print("Plots saved to 'plots/' directory")


if __name__ == "__main__":
    main()