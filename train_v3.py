import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input # 🔥 CRITICAL
from tensorflow.keras import layers, models, optimizers

# 1. SETUP
DATASET_PATH = 'cataract_dataset_fundus'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. UPDATED PREPROCESSING (No more 1./255!)
# We use EfficientNet's official function to match its pre-trained brain
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

# 5. PHASE 1: Warm up (High learning rate)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Phase 1: Warming up the top layers...")
model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])

# 6. PHASE 2: Deep Learning (Unfreeze more layers)
print("🔓 Unfreezing the last 50 layers for deep learning...")
base_model.trainable = True
for layer in base_model.layers[:-50]: # Unfreeze the top 50 layers
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Phase 2: Starting High-Intensity Training...")
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[reduce_lr])

model.save('cataract_expert_v3_final.keras')
print("✅ Success! Model saved as cataract_expert_v3_final.keras")