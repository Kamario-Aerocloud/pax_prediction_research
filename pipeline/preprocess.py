import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with necessary components"""

        self.feature_names = [
            'Destination Airport_encoded', 
            'Airline_encoded', 'Hour', 'Day_of_Week', 'Month', 
            'Day_of_Month', 'Time_Category_encoded'
        ]
        self.target_name = 'PassengerCount'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            df['Flight Time'] = pd.to_datetime(df['Flight Time'], errors='coerce')
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
        df['Hour'] = df['Flight Time'].dt.hour
        df['Day_of_Week'] = df['Flight Time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['Flight Time'].dt.month
        df['Day_of_Month'] = df['Flight Time'].dt.day

        # Create time-based categories
        df['Time_Category'] = pd.cut(df['Hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'])

        # Encode categorical variables
        label_encoders = {}
        categorical_columns = ['Source Airport', 'Destination Airport', 'Airline', 'Time_Category']

        for col in categorical_columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        return df

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split the DataFrame into features and target variable"""
        print("Splitting features and target variable...")
        
        X = df[self.feature_names]
        y = df[self.target_name]
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    
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
        
        # Split features and target
        X, y = self.split_features_target(df)
        
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
