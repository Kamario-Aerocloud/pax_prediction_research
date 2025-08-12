import copy
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

from pax_nn_model import pax_model

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

feature_columns = ['Destination Airport_encoded', 'Airline_encoded',  # encoded data
                   'time_sin', 'time_cos',  # cyclic time features
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


# Define scheduler function
def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * 0.925  # decay by 10%


lr_callback = LearningRateScheduler(scheduler)

# TensorBoard log directory with timestamp to avoid overwriting
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define the model
model = pax_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mae', metrics=['mae', 'mse'])

checkpoint = ModelCheckpoint(
    'best_model.h5',  # file to save the model
    monitor='val_loss',  # quantity to monitor
    save_best_only=True,  # save only the best model
    mode='min',  # minimize validation loss
    verbose=1
)

# train the model
history = model.fit(X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_scaled, y_test),  # or use x_test/y_test for validation if you prefer
                    verbose=1,
                    callbacks=[checkpoint, lr_callback, tensorboard_callback])

# load the best model
model = tf.keras.models.load_model('best_model.h5')

# Generate predictions
y_pred = model.predict(X_test_scaled)

# multiply by the load factor to get the actual passenger count
y_pred = (np.squeeze(y_pred) * y_dataset_full['max_seats'].iloc[split_index:].values).astype(int)  # Get predicted seats
y_test = np.array(y_dataset_full['Boarded'].iloc[split_index:].values)  # Get actual seats
y_test = np.clip(y_test, 0, y_dataset_full['max_seats'].iloc[split_index:].values)

# Plot the outputs of the NN
plt.figure()
plt.plot(y_test, label='Actual Passengers', marker='o', alpha=0.7, markersize=4)
plt.plot(y_pred, label='Predicted Passengers', marker='x', alpha=0.7, markersize=4)
plt.title('Load Factor Predictions vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('Load Factor')
plt.legend()
plt.show()
