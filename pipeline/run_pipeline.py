from preprocess import DataPreprocessor
from train import RandomForestTrainer
from test import ModelTester

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Define the file path to your dataset
    file_path = '../Datasets/SRQ_flights.csv'

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(file_path)

    # Print shapes of the resulting datasets
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Initialize the trainer
    trainer = RandomForestTrainer()

    # Train the model
    trainer.train_model(X_train, y_train)

    # Cross-validation
    #cv_results = trainer.cross_validate(X_train, y_train)

    # Evaluate on training data
    training_metrics = trainer.evaluate_training(X_train, y_train)

    # Plot feature importance
    trainer.plot_feature_importance()

    # Save models
    trainer.save_model()

    # Initialize the tester
    tester = ModelTester()

    # Load the trained model
    tester.load_models()

    # Test the model
    test_metrics, comparison_df = tester.test_model(X_test, y_test)

    print("Training pipeline completed successfully!")