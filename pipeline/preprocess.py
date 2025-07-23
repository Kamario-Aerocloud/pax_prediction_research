import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with necessary components"""

        self.feature_names = [
            'Destination Airport_encoded', 'Airline_encoded', 'Day_of_Week', 'Month','Day_of_Month', 'day_sin',
            'day_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'route_mean', 'route_median', 'route_std'
            #'Time_Category_encoded', 'Hour_sin', 'Hour_cos'
        ]
        self.target_name = 'Boarded'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        #TODO: Implement more sophisticated missing value handling if needed
        print(f"Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        print("Encoding categorical features...")
        
        # Extract time-based features
        df['Hour'] = df['Date'].dt.hour
        df['Day_of_Week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['Date'].dt.month
        df['Day_of_Month'] = df['Date'].dt.day

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
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Cyclical encode time features
        df = self.cyclical_encode_time_features(df)

        # group by airline and destination airport
        df = self.group_by_airline_and_destination(df)

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
