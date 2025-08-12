import tensorflow as tf
print(tf.__version__)


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Add, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

df = pd.read_csv(r"C:\git\pax_prediction_research\Datasets\augmented_SRQ_data_v3.csv")
df = df.drop(columns='Unnamed: 0')

plt.figure(figsize=(20,8))
plt.plot(df['Date'], df['Boarded'], marker='o', linestyle='-')
plt.title('Passenger Count Over Time')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.grid(True)
plt.tight_layout()
plt.show()

feature_columns = [  'Destination Airport_encoded', 'Airline_encoded',
    #'Day_of_Week', 'Month', 'Day_of_Month',
    'day_sin',  'month_sin', 'dow_sin',
    'day_cos', 'month_cos', 'dow_cos',
    'route_mean', 'route_median', 'route_std',]

X = df[feature_columns]
y = df['Boarded']  # or whatever your target column is
dates = pd.to_datetime(df['Date'])


split_index = int(len(df) * 0.8)

# Split data chronologically
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
dates_train = dates.iloc[:split_index]
dates_test = dates.iloc[split_index:]

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training target range: {y_train.min()} to {y_train.max()}")
print(f"Testing target range: {y_test.min()} to {y_test.max()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Log transform targets
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Get the split index (number of training samples)
split_index = len(y_train)

# Extract date and passenger count
df_timestamps = df['Date']
passenger_counts = df['Boarded']

# Plot
plt.figure(figsize=(20, 8))

plt.plot(df_timestamps[:split_index], passenger_counts[:split_index], label='Train', color='blue')
plt.plot(df_timestamps[split_index:], passenger_counts[split_index:], label='Test', color='orange')

plt.title('Passenger Count Over Time (Train/Test Split)', fontsize=16)
plt.xlabel('Flight Date', fontsize=14)
plt.ylabel('Passenger Count', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from tensorflow.keras.callbacks import LearningRateScheduler


# Define scheduler function
def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * 0.925  # decay by 10%


lr_callback = LearningRateScheduler(scheduler)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear'),
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'mse']
             )

# For validation during training, use chronological split within training data
# Take last 20% of training data as validation set
val_split_index = int(len(X_train) * 0.8)

X_train_final = X_train_scaled[:val_split_index]
X_val = X_train_scaled[val_split_index:]
y_train_final = y_train_log.iloc[:val_split_index]
y_val = y_train_log.iloc[val_split_index:]

print("NaNs in X_val:", np.isnan(X_val).sum())
print("NaNs in y_val:", np.isnan(y_val).sum())

# Check for infinities
print("Infs in X_val:", np.isinf(X_val).sum())
print("Infs in y_val:", np.isinf(y_val).sum())

history = model.fit(X_train_final, y_train_final,
                    epochs=200,
                    batch_size=128,
                    validation_data=(X_val, y_val),  # or use x_test/y_test for validation if you prefer
                    verbose=1,
                    callbacks=[lr_callback])

output = model.evaluate(X_test_scaled, y_test_log, verbose=1)
print(output)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Generate predictions
y_pred_log = model.predict(X_test_scaled)

y_pred = y_pred_log.flatten()
y_true = y_test_log.values

# Plot predictions vs actual - now dates align correctly
plt.figure(figsize=(15, 8))

# Plot first 100 test samples
n_samples = min(100, len(y_test))
dates_plot = dates_test.iloc[:n_samples]
y_true_plot = y_true[:n_samples]
y_pred_plot = y_pred[:n_samples]

plt.subplot(2, 1, 1)
plt.plot(dates_plot, y_true_plot, label='True', marker='o', alpha=0.7, markersize=4)
plt.plot(dates_plot, y_pred_plot, label='Predicted', marker='x', alpha=0.7, markersize=4)
plt.title('First 100 Test Samples - Chronological Order')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

