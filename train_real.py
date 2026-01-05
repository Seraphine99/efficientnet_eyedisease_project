import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import ssl
import os

# 1. FIX SSL (For downloading weights on Mac)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. SETTINGS
DATASET_DIR = 'Cataract_dataset' # Update this to your folder name
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 

# 3. DATA PREPARATION (With Augmentation)
# Adding rotation and zoom helps the model not "memorize" specific photos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 4. BUILD MODEL
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Prevents overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. TRAIN
print(f"🚀 Training starting on {len(train_generator)} batches...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 6. SAVE THE MODEL
model.save('cataract_classifier_model.h5')
print("✅ Model saved as 'cataract_classifier_model.h5'")

# 7. PLOT RESULTS
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()