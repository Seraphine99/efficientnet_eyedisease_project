import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. CONFIGURATION ---
DATA_DIR = 'dataset_fundus'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Use a VERY small learning rate so we don't destroy the Phase 1 progress
FINE_TUNE_LR = 1e-5 

# --- 2. DATA PREP ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1, fill_mode='constant', cval=0
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# --- 3. UNFREEZE AND RECOMPILE ---
print("📂 Loading Model for Phase 2...")
model = load_model('eye_master_v1_final.keras')

# Unfreeze the base model
model.get_layer('efficientnetb0').trainable = True

# Recompile with the tiny learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. TRAINING ---
# We use EarlyStopping to prevent overfitting during fine-tuning
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.CSVLogger('phase2_finetune_log.csv')
]

print("🚀 Starting Fine-Tuning... Focusing on Glaucoma precision.")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=15,
    callbacks=callbacks
)

# --- 5. SAVE FINAL MODEL ---
model.save('eye_master_final_tuned.keras')
print("✅ Final Tuned Model Saved!")