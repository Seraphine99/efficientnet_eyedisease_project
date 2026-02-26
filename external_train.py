import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# 1. PATHS
TRAIN_DIR = 'external_dataset/train'
VAL_DIR = 'external_dataset/test'

# 2. DATA GENERATORS (Using EfficientNet Preprocessing)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Required for EfficientNet
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Detect folders and images
print("Scanning folders...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# 3. DYNAMIC CLASS SETUP (Fixes NameError)
num_classes = len(train_gen.class_indices)
labels = list(train_gen.class_indices.keys())

# Calculate Balanced Weights for Imbalance (Pterygium)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(weights))

print("-" * 30)
print(f"Found {num_classes} classes: {labels}")
print(f"Class Weights: {class_weights_dict}")
print("-" * 30)

# 4. BUILD MODEL WITH FINE-TUNING
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# UNFREEZE the last 20 layers so the model can learn eye-specific features
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(), # Stabilizes training
    tf.keras.layers.Dropout(0.5),         # Prevents overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 5. COMPILE (Using a slower Learning Rate for Fine-Tuning)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. TRAIN
print("Starting Fine-Tuning Training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights_dict
)

# 7. SAVE & ANALYZE
model.save('external_eye_model.keras')
print("Model saved as external_eye_model.keras")

# Visualizing results
def plot_results(history, model, val_gen, labels):
    # Plot Accuracy/Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.show()

    # Confusion Matrix
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(val_gen.classes, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(val_gen.classes, y_pred, target_names=labels))

plot_results(history, model, val_gen, labels)