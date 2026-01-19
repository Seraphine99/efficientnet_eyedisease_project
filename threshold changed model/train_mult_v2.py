import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. SETTINGS ---
DATA_DIR = 'dataset_fundus'  # Ensure your folders are: /train, /val, /test
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_NAME = 'eye_master_v1_final.keras'

# --- 2. DATA AUGMENTATION (The "Secret Sauce") ---
# These parameters are tuned for medical eye scans
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,       # Slight rotations are okay for fundus
    width_shift_range=0.1,   # Account for eyes not being perfectly centered
    height_shift_range=0.1,
    horizontal_flip=True,    # Mirrored eyes are still medically valid
    zoom_range=0.1,          # Account for different camera distances
    fill_mode='constant',    # Keep background dark when rotating
    cval=0                   # Padding value for black background
)

# Standard preprocessing for Validation and Testing
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Training Generator (Shuffled for learning)
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'), 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical'
)

# Eval Generator (NOT Shuffled for the Training Confusion Matrix)
train_eval_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_gen.class_indices.keys())

# --- 3. MODEL ARCHITECTURE ---
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAINING ---
csv_logger = callbacks.CSVLogger('eye_master_v1_history.csv', append=False)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\n🚀 Training Starting...")
history = model.fit(
    train_gen, 
    validation_data=test_gen, 
    epochs=EPOCHS, 
    callbacks=[csv_logger, early_stop]
)

model.save(MODEL_NAME)

# --- 5. ANALYTICS ---

# A. Loss and Accuracy Curves
history_df = pd.read_csv('eye_master_v1_history.csv')
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='Train Acc')
plt.plot(history_df['val_accuracy'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()
# Replace plt.show() with this:
plt.savefig('training_results.png')
print("Graph saved as training_results.png")
# No plt.show() here

# B. Dual Confusion Matrices
def plot_matrices(model, train_it, test_it, labels):
    train_preds = np.argmax(model.predict(train_it), axis=1)
    test_preds = np.argmax(model.predict(test_it), axis=1)
    
    cm_train = confusion_matrix(train_it.classes, train_preds)
    cm_test = confusion_matrix(test_it.classes, test_preds)
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax[0], xticklabels=labels, yticklabels=labels)
    ax[0].set_title('Training Set Performance')
    
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax[1], xticklabels=labels, yticklabels=labels)
    ax[1].set_title('Test Set Performance (Unseen)')
    plt.show()

plot_matrices(model, train_eval_gen, test_gen, class_labels)

# C. Final Text Report
print("\n📝 FINAL CLASSIFICATION REPORT (TEST DATA):")
print(classification_report(test_gen.classes, np.argmax(model.predict(test_gen), axis=1), target_names=class_labels))