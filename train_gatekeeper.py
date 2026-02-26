import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils import class_weight
import numpy as np
import os

# 1. SETTINGS
DATA_DIR = 'router datset'  # Using your folder name with the typo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. DATA GENERATORS WITH VALIDATION SPLIT
# We use validation_split so you don't have to manually move files
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=10,
    horizontal_flip=True
)

print("📂 Loading Training Data...")
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("📂 Loading Validation Data...")
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 3. CALCULATE CLASS WEIGHTS
# This fixes the imbalance (1801 External vs 4217 Fundus)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(weights))

print("-" * 30)
print(f"Classes found: {list(train_gen.class_indices.keys())}")
print(f"Applied Weights: {class_weights_dict}")
print("-" * 30)

# 4. BUILD MOBILENETV2 (Lightweight & Fast)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # The 'brain' is already pre-trained

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax') # [external, fundus, invalid]
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. TRAIN
print("Training Router Model (Expected time: 5-10 mins)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10, 
    class_weight=class_weights_dict
)

# 6. SAVE
model.save('eye_router_model.keras')
print("\nSuccess! Saved as 'eye_router_model.keras'")