import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Set paths
PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Load dataset
X = np.load(os.path.join(DATA_PATH, "hw_data.npy"))
y = np.load(os.path.join(DATA_PATH, "hw_labels.npy"))

# Normalize
X = X / 255.0  

# Define CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(672, activation='softmax')
])

# Compile
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"])

# Train
model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2)

# Save
model.save(os.path.join(MODEL_PATH, "cnn_model.h5"))
print("✅ CNN Model saved successfully!")
