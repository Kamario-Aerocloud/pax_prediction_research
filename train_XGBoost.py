import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sklearn
import warnings
import joblib
import numpy as np
from pipeline.preprocess import DataPreprocessor

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

class XGBoostTrainer:
    def __init__(self):
        """Initialize the XGBoost trainer with necessary components"""
        self.model = None
        self.feature_names = ['Destination Airport_encoded', 'Airline_encoded', 'Time_Category_encoded',
                              'time_sin', 'time_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                              'dow_sin', 'dow_cos', 'route_mean_x', 'route_median_x', 'route_std_x',
                              'max_seats']
        self.target_name = 'Boarded'

    def train_model(self, X_train, y_train):
        """Train the XGBoost model"""
        self.model = XGBRegressor(objective='reg:squarederror')
        self.model.fit(X_train, y_train)

    def cross_validate(self, X, y):
        """Perform cross-validation on the model"""
        tscv = TimeSeriesSplit(n_splits=5)
        grid_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(estimator=self.model, param_grid=grid_params,
                                   scoring='neg_mean_squared_error', cv=tscv, verbose=1)
        grid_search.fit(X, y)

        print("Best parameters found: ", grid_search.best_params_)
        return grid_search.best_score_

    def evaluate_training(self, X_train, y_train):
        """Evaluate the model on training data"""
        y_pred = self.model.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)

        print(f"Training R^2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        return {'r2': r2, 'mse': mse, 'mae': mae}

    def plot_feature_importance(self):
        """Plot feature importance of the trained model"""
        if self.model is None:
            print("Model not trained yet.")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.model.feature_importances_, y=self.feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

    def plot_predictions(self, y_pred, y_test):
        """Plot predictions vs actual values"""
        if self.model is None:
            print("Model not trained yet.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values[:100], label='Actual', marker='o', markersize=4)
        plt.plot(y_pred[:100], label='Predicted', marker='s', markersize=4)
        plt.title('Actual vs Predicted (First 100 samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Passenger Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.grid()
        plt.show()

    def plot_linear_regression(self, y_pred, y_test):
        """Plot linear regression line for predictions"""
        if self.model is None:
            print("Model not trained yet.")
            return

            # Ensure y_test is a Series
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test['Boarded']

        sns.regplot(x=y_test.values, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title('Linear Regression of Predictions vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.xlim(y_test.min(), y_test.max())
        plt.ylim(y_test.min(), y_test.max())
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Define the file path to your dataset
    file_path = 'Datasets/SRQ_flights.csv'

    df = pd.read_csv(file_path)
    df['actual_date'] = pd.to_datetime(df['actual_date'], errors='coerce')

    # Extract time of day in seconds
    df['seconds_in_day'] = (
            df['actual_date'].dt.hour * 3600 +
            df['actual_date'].dt.minute * 60 +
            df['actual_date'].dt.second
    )

    # Normalize to [0, 2π]
    seconds_in_day_total = 24 * 60 * 60  # 86400
    df['time_angle'] = 2 * np.pi * df['seconds_in_day'] / seconds_in_day_total

    # Compute sin and cos
    df['time_sin'] = np.sin(df['time_angle'])
    df['time_cos'] = np.cos(df['time_angle'])

    df['Hour'] = df['actual_date'].dt.hour

    # Create time-based categories
    df['Time_Category'] = pd.cut(df['Hour'],
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])

    df['Time_Category_encoded'] = LabelEncoder().fit_transform(df['Time_Category'])

    feature_names = ['Destination Airport_encoded', 'Airline_encoded', 'Time_Category_encoded', # encoded data
                       'time_sin', 'time_cos',  # cyclic time features
                       'day_sin', 'day_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',  # cyclic date features
                       'route_mean', 'route_median', 'route_std',
                       'max_seats']

    target_name = ['Boarded', 'max_seats']

    #  get the inputs and outputs
    X = df[feature_names]
    y = df[['Boarded', 'max_seats']]  # or whatever your target column is

    # Find columns with NaN values
    nan_indices = df[X.isna().any(axis=1)].index.tolist()
    print("Row indices with NaNs:", nan_indices)
    X = X.drop(index=nan_indices).reset_index(drop=True)
    y = y.drop(index=nan_indices).reset_index(drop=True)

    # get the load factor by dividing boarded by max_seats
    y['LoadFactor'] = y['Boarded'] / y['max_seats']
    s = pd.to_numeric(y['LoadFactor'], errors='coerce')
    nan_indices = y[y['LoadFactor'].isna()].index.tolist()
    inf_indices = y[np.isinf(s)].index.tolist()
    print("Rows with NaNs in LoadFactor:", nan_indices)
    y = y.drop(index=nan_indices).reset_index(drop=True)
    y = y.drop(index=inf_indices).reset_index(drop=True)
    X = X.drop(index=inf_indices).reset_index(drop=True)
    X = X.drop(index=nan_indices).reset_index(drop=True)

    # ensure load factor is between 0-1
    y['LoadFactor'] = y['LoadFactor'].clip(lower=0.0, upper=1.0)
    y_dataset_full = copy.deepcopy(y)

    y = y['LoadFactor']  # use only the LoadFactor column as target

    split_ratio = 0.8  # 80% for training, 20% for testing
    split_index = int(len(df) * split_ratio)

    # Split data chronologically
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Training target range: {y_train.min()} to {y_train.max()}")
    print(f"Testing target range: {y_test.min()} to {y_test.max()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the trainer
    trainer = XGBoostTrainer()

    # Train the model
    trainer.train_model(X_train, y_train)

    # Cross-validation
    cv_score = trainer.cross_validate(X_train, y_train)

    # Evaluate on training data
    training_metrics = trainer.evaluate_training(X_train, y_train)

    y_pred = trainer.model.predict(X_test_scaled)
    y_pred = np.clip(y_pred, 0, 1)  # Ensure predictions are between 0 and 1

    # multiply the load factor by max seats to get the actual passenger count
    y_pred_pax = (np.squeeze(y_pred) * y_dataset_full['max_seats'].iloc[split_index:].values).astype(
        int)  # Get predicted seats
    y_test_pax = np.array(y_dataset_full['Boarded'].iloc[split_index:].values)  # Get actual seats
    y_test_pax = np.clip(y_test_pax, 0, y_dataset_full['max_seats'].iloc[split_index:].values)

    # Plot the outputs of the NN with actual passenger counts
    plt.figure(1)
    plt.plot(y_test_pax, label='Actual Passengers', marker='o', alpha=0.7, markersize=4)
    plt.plot(y_pred_pax, label='Predicted Passengers', marker='x', alpha=0.7, markersize=4)
    plt.title('Passenger Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('PAX Prediction')
    plt.legend()

    # multiply by the load factor to get the actual passenger count
    y_pred_lf = np.squeeze(y_pred)  # Get predicted lf
    y_test_lf = np.array(y_dataset_full['LoadFactor'].iloc[split_index:].values)  # Get actual lf
    y_test_lf = np.clip(y_test_lf, 0, 1)

    # Plot the outputs of the NN with LF
    plt.figure(2)
    plt.plot(y_test_lf, label='Actual Load Factor', marker='o', alpha=0.7, markersize=4)
    plt.plot(y_pred_lf, label='Predicted Load Factor', marker='x', alpha=0.7, markersize=4)
    plt.title('Load Factor Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Load Factor')
    plt.legend()
    plt.show()

    print("XGBoost training pipeline completed successfully!")