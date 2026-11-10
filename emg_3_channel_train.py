import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------
# 1. Load 3-Channel EMG Dataset
# -------------------------------------------------------------
DATA_DIR = "./emg_3ch_dataset"
X_PATH = os.path.join(DATA_DIR, "X_3ch.npy")
Y_PATH = os.path.join(DATA_DIR, "y_3ch.npy")

if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    raise FileNotFoundError(
        f"Dataset files not found in {DATA_DIR}. Run collect_3ch_emg.py first!"
    )

print("Loading 3-channel EMG dataset...")
X = np.load(X_PATH)  # Expected shape: (N, 200, 3)
y = np.load(Y_PATH)  # Expected shape: (N,)

print(f"Dataset shape: X = {X.shape}, y = {y.shape}")

CLASS_NAMES = [
    "Rest",
    "Wrist Close",
    "Wrist Open",
    "Wrist Up",
    "Wrist Down"
]
NUM_CLASSES = len(CLASS_NAMES)

print("Sample counts per class:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  [{idx}] {name:<12}: {np.sum(y == idx)}")

# -------------------------------------------------------------
# 2. Stratified Train / Test Split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining windows: {len(X_train)} | Test windows: {len(X_test)}")

# -------------------------------------------------------------
# 3. Model Architecture (3-Channel 1D-CNN)
# -------------------------------------------------------------
# Input shape: (200 time steps, 3 channels)
model = models.Sequential([
    layers.Input(shape=(200, 3)),

    # Block 1: Multi-compartment local spatial-temporal features
    layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Block 2: Cross-channel contraction dynamics
    layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Block 3: High-level frequency/activation profiling
    layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),

    # Classification Head
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------------
# 4. Training Callbacks
# -------------------------------------------------------------
training_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='best_emg_3ch_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------------------------
# 5. Model Training (80 Epochs)
# -------------------------------------------------------------
EPOCHS = 80
BATCH_SIZE = 32

print(f"\nTraining 3-channel classifier for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=training_callbacks
)

# -------------------------------------------------------------
# 6. Evaluation
# -------------------------------------------------------------
print("\nEvaluating model on test set...")
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------
# 7. Convert & Save to TensorFlow Lite
# -------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "emg_3ch_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"\nTFLite model exported successfully: {os.path.abspath(tflite_path)}")