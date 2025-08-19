import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with necessary components"""

        self.feature_names = ['Destination Airport_encoded', 'Airline_encoded', 'Time_Category_encoded', # encoded data
                           'time_sin', 'time_cos',  # cyclic time features
                           'day_sin', 'day_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'time_sin', 'time_cos',  # cyclic date features
                           'route_mean_x', 'route_median_x', 'route_std_x',
                           'max_seats']
        self.target_name = 'Boarded'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['actual_date'] = pd.to_datetime(df['actual_date'], errors='coerce')
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df[self.feature_names + [self.target_name, 'actual_date', 'max_seats']].copy()
        nan_indices = df[df.isna().any(axis=1)].index.tolist()
        df = df.drop(index=nan_indices).reset_index(drop=True)

        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        print("Encoding categorical features...")
        
        # Extract time-based features
        df['Day_of_Week'] = df['actual_date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['actual_date'].dt.month
        df['Day_of_Month'] = df['actual_date'].dt.day
        df['Hour'] = df['actual_date'].dt.hour



        # Create time-based categories
        df['Time_Category'] = pd.cut(df['Hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'])

        # Encode categorical variables
        label_encoders = {}
        categorical_columns = ['Destination Airport', 'Airline', 'Time_Category']

        for col in categorical_columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        return df

    def cyclical_encode_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cyclical encoding for time features"""
        print("Cyclical encoding of time features...")

        # Hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        # Day of Week
        df['dow_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

        # Day of Month
        df['day_sin'] = np.sin(2 * np.pi * df['Day_of_Month'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['Day_of_Month'] / 31)

        # Month
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # Extract time components
        df['hour'] = df['actual_date'].dt.hour
        df['minute'] = df['actual_date'].dt.minute
        df['second'] = df['actual_date'].dt.second
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

        print("Cyclical encoding completed.")
        return df

    def group_by_airline_and_destination(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group by airline and destination airport to aggregate passenger counts"""
        print("Grouping by airline and destination airport...")

        route_stats = df.groupby(['Airline', 'Destination Airport'])['Boarded'].agg([
            ('route_mean', 'mean'),
            ('route_median', 'median'),
            ('route_std', 'std'),
        ]).reset_index()

        df = df.merge(route_stats, on=['Airline', 'Destination Airport'], how='left')

        print(f"Grouped data shape: {df.shape}")

        return df
    
    def preprocess_data(self, file_path: str,
                         test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Load data
        df = self.load_data(file_path)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Cyclical encode time features
        df = self.cyclical_encode_time_features(df)

        # group by airline and destination airport
        df = self.group_by_airline_and_destination(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Split into train and test sets
        split_factor = 0.8
        split_index = int(len(df) * split_factor)

        train_df = df[:split_index]
        test_df = df[split_index:]

        # Extract X and y from train and test sets
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_name]

        X_test = test_df[self.feature_names]
        y_test = test_df[self.target_name]
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print("Data preprocessing completed successfully!")
        
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data('Datasets/flights_with_counts.csv')

    print("Preprocessing complete. Ready for model training.")
