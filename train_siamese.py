import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.model_selection import train_test_split

# Disable OneDNN optimizations for stability
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set paths
PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# Image settings
IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

# Load preprocessed signature pairs
pairs = np.load(os.path.join(DATA_PATH, "sig_pairs.npy"))
labels = np.load(os.path.join(DATA_PATH, "sig_labels.npy"))

# Normalize images
pairs = pairs.astype("float32") / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)
X_train_1, X_train_2 = X_train[:, 0], X_train[:, 1]
X_test_1, X_test_2 = X_test[:, 0], X_test[:, 1]

# **Build the Siamese Base Model**
def build_base_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu")
    ])
    return model

base_model = build_base_model()

# Input layers
input1 = layers.Input(shape=INPUT_SHAPE)
input2 = layers.Input(shape=INPUT_SHAPE)

# Pass through base model
encoded1 = base_model(input1)
encoded2 = base_model(input2)

# **L1 Distance Function**
def l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Compute Distance
l1_distance_layer = layers.Lambda(l1_distance)([encoded1, encoded2])
output = layers.Dense(1, activation="sigmoid")(l1_distance_layer)

# Create and Compile Model
siamese_model = keras.Model(inputs=[input1, input2], outputs=output)
siamese_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0003), metrics=["accuracy"])

# Train Model
siamese_model.fit(
    [X_train_1, X_train_2], y_train,
    validation_data=([X_test_1, X_test_2], y_test),
    batch_size=32,
    epochs=50
)
# Save Model
siamese_model.save(os.path.join(MODEL_PATH, "siamese_model.h5"))
print("✅ Siamese Model saved successfully!")
