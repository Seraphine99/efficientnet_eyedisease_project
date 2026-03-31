import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from clahe import apply_clahe_rgb

# Global Settings
IMG_SIZE = (380, 380) # B4 Native resolution
BATCH_SIZE = 8        # B4 is heavy, lower batch size to avoid memory errors
EPOCHS_HEAD= 8        # Phase 1 epochs
EPOCHS_FINE= 8        # Phase 2 epochs
NUM_CLASSES = 4


# Model Architecture of EfficientNet-B4
base_model = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(380, 380, 3)
)

# Freeze base model for phase 1
base_model.trainable = False

#Custom Layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)             # Slightly higher dropout for B4 to prevent overfitting
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Data Generator (Augmentatio and CLAHE)
train_datagen = ImageDataGenerator(
    preprocessing_function = apply_clahe_rgb,
    rotation_range = 20,
    zoom_range=0.15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)

val_datagen = ImageDataGenerator(
    preprocessing_function = apply_clahe_rgb
)

train_gen = train_datagen.flow_from_directory(
    'dataset_fundus/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'dataset_fundus/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Callbacks and training control
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'eye_model_b4_best.keras',
    monitor='val_accuracy',
    save_best_only=True
)

# Phase 1 to train classification head
print("Phase 1: Training classifier head")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# Phase 2 to fine tune last layers
print("Phase 2: Fine-tuning EfficientNet-B4")

base_model.trainable = True

# Unfreeze only last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)



model.save('eye_model_b4_master.keras')
print("Final model saved successfully!")
