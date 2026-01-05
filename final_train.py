import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# 1. SETUP PATHS (Ensure these match your 'dataset_fundus' folder)
DATASET_PATH = 'cataract_dataset_fundus'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. DATA AUGMENTATION (Tailored for medical fundus images)
# We use horizontal flips and slight rotations since eyes can be tilted
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test should ONLY be rescaled (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# 3. GENERATORS
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_PATH}/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    f'{DATASET_PATH}/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 4. BUILD THE MODEL (Transfer Learning)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- STEP A: Freeze the base model first ---
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # High dropout to prevent overfitting on 1000 images
    layers.Dense(1, activation='sigmoid')
])

# 5. COMPILE & TRAIN (Stage 1: Top Layers)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Phase 1: Training top layers...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# --- STEP B: Fine-Tuning (Unfreeze last 20 layers) ---
print("🔓 Unfreezing top layers for Fine-Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Use a MUCH smaller learning rate for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Phase 2: Fine-Tuning the brain...")
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 6. SAVE MODEL
model.save('cataract_expert_v3.keras') # Use .keras (modern format)
print("✅ Training complete! Model saved as cataract_expert_v3.keras")