import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Create artifacts directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

print("Loading dataset...")
# Load the enhanced dataset with weather data
created_df = pd.read_csv("C:\\Users\\debas\\OneDrive\\Desktop\\ahu1_model_data_with_weather.csv")

# Add timestamp column if it exists in the original data
# We'll need to recreate it or load from the pivoted data
pivoted_table = pd.read_csv("C:\\Users\\debas\\OneDrive\\Desktop\\ahu1_pivoted_data.csv")
pivoted_table['timestamp'] = pd.to_datetime(pivoted_table['timestamp'])

# Extract hour of day feature
pivoted_table['hour_of_day'] = pivoted_table['timestamp'].dt.hour

# Merge hour_of_day with created_df
# Since created_df was created from pivoted_table, they should align by index
created_df['hour_of_day'] = pivoted_table['hour_of_day'].values

print(f"Dataset loaded with shape: {created_df.shape}")

# Select features and targets
features = ['outdoor_temp', 'outdoor_humidity', 'hour_of_day']
targets = ['TrAvg', 'HuAvg1']

# Create a clean dataset with no missing values
lstm_df = created_df[features + targets].copy()
lstm_df = lstm_df.dropna()

print(f"Dataset shape after removing NaN: {lstm_df.shape}")
print(f"\nDataset statistics:")
print(lstm_df.describe())
print(f"\nFeatures: {features}")
print(f"Targets: {targets}")

# Scale the features and targets
print("\nScaling features and targets...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale the features and targets
df_features = feature_scaler.fit_transform(lstm_df[features])
df_targets = target_scaler.fit_transform(lstm_df[targets])

print(f"Features shape after scaling: {df_features.shape}")
print(f"Targets shape after scaling: {df_targets.shape}")

# Create sequences for LSTM
timesteps = 24  # Use 24 previous timesteps to predict the next value

X = []
y = []

print(f"\nCreating sequences with timesteps={timesteps}...")
for i in range(len(df_features) - timesteps):
    # Create sequence of features (outdoor_temp, outdoor_humidity, hour_of_day)
    feature_seq = df_features[i:(i + timesteps)]
    X.append(feature_seq)
    
    # Target is the next value after the sequence (TrAvg and HuAvg1)
    target = df_targets[i + timesteps]
    y.append(target)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Print shapes to verify
print(f"X shape: {X.shape}")  # Should be (samples, timesteps, 3) - 3 features
print(f"y shape: {y.shape}")  # Should be (samples, 2) - 2 targets

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing set - X: {X_test.shape}, y: {y_test.shape}")

# Build the LSTM model
print("\nBuilding LSTM model...")
model = Sequential()
# LSTM layer to capture temporal dependencies from outdoor weather data and time features
model.add(LSTM(50, activation='relu', input_shape=(timesteps, len(features))))
# Dense output layer with 2 neurons for TrAvg and HuAvg1 prediction
model.add(Dense(len(targets)))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print model summary
model.summary()

# Train the model
print("\n" + "="*60)
print("Training the LSTM model...")
print("="*60)
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32,
    verbose=1, 
    validation_data=(X_test, y_test)
)

print("\nTraining complete!")

# Plot training history
print("\nGenerating training history plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot MAE
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Model Mean Absolute Error', fontsize=14, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training history plot saved to 'artifacts/lstm_training_history.png'")

# Make predictions and inverse transform to original scale
print("\nMaking predictions on test set...")
y_pred = model.predict(X_test)

# Inverse transform to get actual values
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred)

# Calculate evaluation metrics
mse_travg = mean_squared_error(y_test_actual[:, 0], y_pred_actual[:, 0])
mae_travg = mean_absolute_error(y_test_actual[:, 0], y_pred_actual[:, 0])
r2_travg = r2_score(y_test_actual[:, 0], y_pred_actual[:, 0])

mse_huavg = mean_squared_error(y_test_actual[:, 1], y_pred_actual[:, 1])
mae_huavg = mean_absolute_error(y_test_actual[:, 1], y_pred_actual[:, 1])
r2_huavg = r2_score(y_test_actual[:, 1], y_pred_actual[:, 1])

print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"\nTrAvg (Indoor Temperature) Prediction:")
print(f"  MSE: {mse_travg:.4f}")
print(f"  MAE: {mae_travg:.4f}")
print(f"  R² Score: {r2_travg:.4f}")

print(f"\nHuAvg1 (Indoor Humidity) Prediction:")
print(f"  MSE: {mse_huavg:.4f}")
print(f"  MAE: {mae_huavg:.4f}")
print(f"  R² Score: {r2_huavg:.4f}")
print("="*60)

# Visualize predictions vs actual values
print("\nGenerating prediction visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: TrAvg (Indoor Temperature) - Time Series
ax1 = axes[0, 0]
sample_size = min(200, len(y_test_actual))
ax1.plot(range(sample_size), y_test_actual[:sample_size, 0], label='Actual TrAvg', linewidth=2, alpha=0.7)
ax1.plot(range(sample_size), y_pred_actual[:sample_size, 0], label='Predicted TrAvg', linewidth=2, alpha=0.7)
ax1.set_title('Indoor Temperature (TrAvg): Actual vs Predicted', fontsize=14, fontweight='bold')
ax1.set_ylabel('Temperature', fontsize=12)
ax1.set_xlabel('Sample', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: TrAvg - Scatter Plot
ax2 = axes[0, 1]
ax2.scatter(y_test_actual[:, 0], y_pred_actual[:, 0], alpha=0.5, s=20)
ax2.plot([y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 
         [y_test_actual[:, 0].min(), y_test_actual[:, 0].max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual TrAvg', fontsize=12)
ax2.set_ylabel('Predicted TrAvg', fontsize=12)
ax2.set_title(f'TrAvg Scatter Plot (R²={r2_travg:.4f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: HuAvg1 (Indoor Humidity) - Time Series
ax3 = axes[1, 0]
ax3.plot(range(sample_size), y_test_actual[:sample_size, 1], label='Actual HuAvg1', linewidth=2, alpha=0.7)
ax3.plot(range(sample_size), y_pred_actual[:sample_size, 1], label='Predicted HuAvg1', linewidth=2, alpha=0.7)
ax3.set_title('Indoor Humidity (HuAvg1): Actual vs Predicted', fontsize=14, fontweight='bold')
ax3.set_ylabel('Humidity', fontsize=12)
ax3.set_xlabel('Sample', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: HuAvg1 - Scatter Plot
ax4 = axes[1, 1]
ax4.scatter(y_test_actual[:, 1], y_pred_actual[:, 1], alpha=0.5, s=20, color='green')
ax4.plot([y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 
         [y_test_actual[:, 1].min(), y_test_actual[:, 1].max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual HuAvg1', fontsize=12)
ax4.set_ylabel('Predicted HuAvg1', fontsize=12)
ax4.set_title(f'HuAvg1 Scatter Plot (R²={r2_huavg:.4f})', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/lstm_prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("Prediction visualization saved to 'artifacts/lstm_prediction_results.png'")

# Save the model and scalers
print("\nSaving model and scalers to artifacts folder...")
model.save('artifacts/lstm_indoor_prediction_model.h5')
model.save('artifacts/lstm_indoor_prediction_model.keras')

# Save the scalers for future predictions
joblib.dump(feature_scaler, 'artifacts/feature_scaler.pkl')
joblib.dump(target_scaler, 'artifacts/target_scaler.pkl')

# Save model metadata
metadata = {
    'features': features,
    'targets': targets,
    'timesteps': timesteps,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'metrics': {
        'TrAvg': {
            'MSE': float(mse_travg),
            'MAE': float(mae_travg),
            'R2': float(r2_travg)
        },
        'HuAvg1': {
            'MSE': float(mse_huavg),
            'MAE': float(mae_huavg),
            'R2': float(r2_huavg)
        }
    }
}

import json
with open('artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n" + "="*60)
print("MODEL AND ARTIFACTS SAVED SUCCESSFULLY")
print("="*60)
print("✓ Model saved as 'artifacts/lstm_indoor_prediction_model.h5' and '.keras'")
print("✓ Feature scaler saved as 'artifacts/feature_scaler.pkl'")
print("✓ Target scaler saved as 'artifacts/target_scaler.pkl'")
print("✓ Model metadata saved as 'artifacts/model_metadata.json'")
print("✓ Training history plot: 'artifacts/lstm_training_history.png'")
print("✓ Prediction results plot: 'artifacts/lstm_prediction_results.png'")
print("="*60)

print("\nModel training and saving complete!")

# Create a prediction function for new data
def predict_indoor_conditions(outdoor_temp_sequence, outdoor_humidity_sequence, hour_sequence):
    """
    Predict indoor temperature (TrAvg) and humidity (HuAvg1) from outdoor conditions and time
    
    Args:
        outdoor_temp_sequence: A sequence of outdoor temperature values (length = timesteps)
        outdoor_humidity_sequence: A sequence of outdoor humidity values (length = timesteps)
        hour_sequence: A sequence of hour values (0-23) (length = timesteps)
        
    Returns:
        Dictionary with predicted TrAvg and HuAvg1 values
    """
    if len(outdoor_temp_sequence) != timesteps or len(outdoor_humidity_sequence) != timesteps or len(hour_sequence) != timesteps:
        raise ValueError(f"Input sequences must have length {timesteps}")
    
    # Combine features
    input_features = np.column_stack([outdoor_temp_sequence, outdoor_humidity_sequence, hour_sequence])
    
    # Scale the input
    scaled_input = feature_scaler.transform(input_features)
    
    # Reshape for LSTM (1 sample, timesteps, 3 features)
    scaled_input = scaled_input.reshape(1, timesteps, 3)
    
    # Make prediction
    prediction_scaled = model.predict(scaled_input, verbose=0)
    
    # Inverse transform to get actual values
    prediction_actual = target_scaler.inverse_transform(prediction_scaled)
    
    return {
        'TrAvg': prediction_actual[0, 0], 
        'HuAvg1': prediction_actual[0, 1]
    }

print("\n" + "="*60)
print("PREDICTION FUNCTION USAGE")
print("="*60)
print("Function: predict_indoor_conditions(outdoor_temp_seq, outdoor_humidity_seq, hour_seq)")
print(f"Each sequence must have {timesteps} values")
print("Example:")
print("  temps = [25.0, 25.5, 26.0, ...] # 24 values")
print("  humidity = [60.0, 61.0, 62.0, ...] # 24 values")
print("  hours = [0, 1, 2, 3, ...] # 24 values")
print("  result = predict_indoor_conditions(temps, humidity, hours)")
print("="*60)
