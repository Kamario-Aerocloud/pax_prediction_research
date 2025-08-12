import tensorflow as tf

print(tf.__version__)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LearningRateScheduler

# load the data
# df = pd.read_csv(r"C:\git\pax_prediction_research\Datasets\augmented_SRQ_data_v3.csv")
df = pd.read_csv(r"C:\git\pax_prediction_research\Datasets\SRQ_flights.csv")
# df = pd.read_csv(r"C:\git\pax_prediction_research\Datasets\SRQ_flights_small.csv")
df = df.drop(columns='Unnamed: 0')

# Convert to datetime
df['actual_date'] = pd.to_datetime(df['actual_date'])

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

feature_columns = ['Destination Airport_encoded', 'Airline_encoded', # encoded data
                   'time_sin', 'time_cos', # cyclic time features
                   'day_sin', 'day_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',  # cyclic date features
                   'route_mean', 'route_median', 'route_std',
                   'max_seats']

#  get the inputs and outputs
X = df[feature_columns]
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

# Remove and Nans and infs from the data



# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get the split index (number of training samples)
split_index = len(y_train)

# Extract date and passenger count
df_timestamps = df['Date']
passenger_counts = df['Boarded']

# Define scheduler function
def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * 0.925  # decay by 10%


lr_callback = LearningRateScheduler(scheduler)

# model definition
input_encoded = tf.keras.Input(shape=(2,))  # 11 features as per feature_columns
input_time = tf.keras.Input(shape=(6,))  # 11 features as per feature_columns
input_route = tf.keras.Input(shape=(3,))  # 11 features as per feature_columns

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1024, activation='relu', input_shape=(11,)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='linear'),
# ])

# use a standard model with multiple inputs


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'mse']
              )

# Take last 20% of training data as validation set
val_split_index = int(len(X_train) * 0.8)

X_train_final = X_train_scaled[:val_split_index]
X_val = X_train_scaled[val_split_index:]
y_train_final = y_train.iloc[:val_split_index]
y_val = y_train.iloc[val_split_index:]

# train the model
history = model.fit(X_train_final, y_train_final,
                    epochs=200,
                    batch_size=128,
                    validation_data=(X_val, y_val),  # or use x_test/y_test for validation if you prefer
                    verbose=1,
                    callbacks=[lr_callback])

# output = model.evaluate(X_test_scaled, y_test_log, verbose=1)
#
# # Generate predictions
# y_pred_log = model.predict(X_test_scaled)
# y_pred = y_pred_log.flatten()
# y_true = y_test_log.values
#
# # Plot predictions vs actual - now dates align correctly
# plt.figure(figsize=(15, 8))
#
# # Plot first 100 test samples
# n_samples = min(100, len(y_test))
# dates_plot = dates_test.iloc[:n_samples]
# y_true_plot = y_true[:n_samples]
# y_pred_plot = y_pred[:n_samples]
#
# plt.subplot(2, 1, 1)
# plt.plot(dates_plot, y_true_plot, label='True', marker='o', alpha=0.7, markersize=4)
# plt.plot(dates_plot, y_pred_plot, label='Predicted', marker='x', alpha=0.7, markersize=4)
# plt.title('First 100 Test Samples - Chronological Order')
# plt.xlabel('Date')
# plt.ylabel('Passenger Count')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
