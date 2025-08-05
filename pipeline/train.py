import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import DataPreprocessor


class RandomForestTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None

    def create_model(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest model with specified parameters"""
        default_params = {
            'n_estimators': 500,
            'max_depth': 40,
            'min_samples_leaf': 6,
            'max_features': 'sqrt',
            'random_state': None,
            'min_samples_split': 10,
            'bootstrap': True,
            'criterion': 'poisson',
            'n_jobs': -1
        }

        # Update with provided parameters
        default_params.update(kwargs)

        self.model = RandomForestRegressor(**default_params)
        print(f"Random Forest model created with parameters: {default_params}")
        return self.model


    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    tune_hyperparameters: bool = False, **model_params) -> RandomForestRegressor:
        """Train the Random Forest model"""
        print("Training Random Forest model...")

        self.create_model(**model_params)
        self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Model training completed!")
        return self.model

    # def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series,
    #                    cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
    #     """Perform cross-validation"""
    #     if self.model is None:
    #         raise ValueError("Model not trained yet. Call train_model() first.")
    #
    #     print(f"Performing {cv}-fold cross-validation...")
    #
    #     # Different scoring metrics
    #     scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    #     cv_results = {}
    #
    #     for metric in scoring_metrics:
    #         scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=metric)
    #         cv_results[metric] = {
    #             'mean': scores.mean(),
    #             'std': scores.std(),
    #             'scores': scores
    #         }
    #         print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    #
    #     return cv_results

    def evaluate_training(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Evaluate model on training data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        y_pred = self.model.predict(X_train)

        metrics = {
            'mse': mean_squared_error(y_train, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
            'mae': mean_absolute_error(y_train, y_pred),
            'r2': r2_score(y_train, y_pred)
        }

        print("Training metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return metrics

    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model first.")

        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)

        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()

        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath: str = 'models/random_forest_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str = 'models/random_forest_model.pkl'):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.best_params = model_data.get('best_params')
            self.feature_importance = model_data.get('feature_importance')
            self.random_state = model_data.get('random_state', 42)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def main():
    """Main training function"""
    # Configuration
    DATA_PATH = 'FlightsByDay-SRQ-2025_07_21_09_49_46.csv'  # Change this to your data file path
    TARGET_COLUMN = 'target'  # Change this to your target column name

    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor()
    trainer = RandomForestTrainer()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        DATA_PATH, TARGET_COLUMN, test_size=0.2
    )

    # Train model
    trainer.train_model(
        X_train, y_train,
    )

    # Cross-validation
    #cv_results = trainer.cross_validate(X_train, y_train)

    # Evaluate on training data
    training_metrics = trainer.evaluate_training(X_train, y_train)

    # Plot feature importance
    trainer.plot_feature_importance()

    # Save models
    trainer.save_model()

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()