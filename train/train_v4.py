import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. SETUP
DATASET_PATH = 'cataract_dataset_fundus'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. UPDATED PREPROCESSING
# EfficientNet's official function to match its pre-trained brain
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    f'{DATASET_PATH}/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    f'{DATASET_PATH}/val', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# 3. SMART CALLBACKS
# If accuracy stops improving for 3 epochs, it cuts the learning rate by 80%
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1
)

# 4. MODEL BUILD
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# 5. PHASE 1: Warm up
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

print("Phase 1: Warming up the top layers...")
# Store history from Phase 1
history1 = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])

# 6. PHASE 2: Deep Learning
print("Unfreezing layers...")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

print("Phase 2: Starting High-Intensity Training...")
# Store history from Phase 2
history2 = model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[reduce_lr])

model.save('cataract_expert_v4_final.keras')

# ==========================================
# 7. ANALYTICS SECTION (FOR YOUR SUPERVISOR)
# ==========================================

# A. Combine History from both phases
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(len(acc))

# Plot 1: Loss and Accuracy Curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Total Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Total Epochs')
plt.legend()
plt.show()



# B. Predictions for Confusion Matrix
# We need to reset the generator to get predictions in the right order
val_generator.reset()
predictions = model.predict(val_generator)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix: Cataract vs Normal')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Plot 3: F1-Score and Classification Report
# We plot the F1-score as a bar chart for professional presentation
report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
f1_scores = [report_dict[label]['f1-score'] for label in class_labels]

plt.figure(figsize=(8, 5))
sns.barplot(x=class_labels, y=f1_scores, palette='viridis')
plt.title('F1-Score per Class')
plt.ylim(0, 1)
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

print("\nFull Classification Report for your records:")
print(classification_report(y_true, y_pred, target_names=class_labels))