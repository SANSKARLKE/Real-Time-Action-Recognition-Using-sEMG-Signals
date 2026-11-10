import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------------------
# 1. Load Doubled 2-Channel EMG Dataset
# -------------------------------------------------------------
DATA_DIR = "./emg_2ch_dataset"
X_PATH = os.path.join(DATA_DIR, "X_2ch.npy")
Y_PATH = os.path.join(DATA_DIR, "y_2ch.npy")

if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    raise FileNotFoundError(
        f"Dataset files not found in {DATA_DIR}. Run the collection script first!"
    )

print("Loading 2-channel EMG dataset...")
X = np.load(X_PATH)  # Expected shape: (N, 200, 2)
y = np.load(Y_PATH)  # Expected shape: (N,)

print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
print(f"Class counts -> Rest (0): {np.sum(y == 0)}, Close (1): {np.sum(y == 1)}, Open (2): {np.sum(y == 2)}")

# -------------------------------------------------------------
# 2. Stratified Train / Test Split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training windows: {len(X_train)} | Test windows: {len(X_test)}")

# -------------------------------------------------------------
# 3. Model Architecture (Dual-Channel 1D-CNN)
# -------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(200, 2)),

    # Block 1: Multi-channel spatial/temporal feature extraction
    layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Block 2: Intermediate temporal dynamics
    layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    # Block 3: High-level frequency profile representation
    layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),

    # Classification Head
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # 3 Classes: Rest (0), Close (1), Open (2)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------------
# 4. Training Callbacks (Extended Patience for 80 Epochs)
# -------------------------------------------------------------
training_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,               # Extended from 8 to allow deeper convergence
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
        filepath='best_emg_2ch_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# -------------------------------------------------------------
# 5. Train Model (Doubled to 80 Epochs)
# -------------------------------------------------------------
EPOCHS = 80  # Doubled from 40

print(f"\nTraining 2-Channel EMG Gesture Classifier for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=training_callbacks
)

# -------------------------------------------------------------
# 6. Evaluation on Test Set
# -------------------------------------------------------------
print("\nEvaluating model on unseen test set...")
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

target_names = ['Rest', 'Wrist Close', 'Wrist Open']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------
# 7. Convert & Export to TensorFlow Lite
# -------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "emg_2ch_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"\nTFLite model exported successfully: {os.path.abspath(tflite_path)}")