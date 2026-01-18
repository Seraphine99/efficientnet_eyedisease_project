import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETTINGS ---
DATA_DIR = 'dataset_fundus'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. DATA PREP (Same as Phase 1) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.1,
    zoom_range=[0.7, 1.1], 
    brightness_range=[0.6, 1.4],
    fill_mode='constant', 
    cval=0
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'), target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# --- 3. THE SURGICAL UNFREEZE ---
print("📂 Loading the original GOOD model (v1)...")
model = load_model('eye_master_v1_final.keras')

base_model = model.get_layer('efficientnetb0')
base_model.trainable = True

# We FREEZE everything EXCEPT the last 30 layers
# EfficientNetB0 has about 237 layers total
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"✅ Frozen early layers. Unfrozen the last 30 layers for specialization.")

# --- 4. COMPILE WITH STABLE LEARNING RATE ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Tiny LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. TRAIN WITH MORE PATIENCE ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, # Give it more time to beat the previous score
    restore_best_weights=True
)
# --- CALCULATE CLASS WEIGHTS ---
# 0: Cataract, 1: DR, 2: Glaucoma, 3: Normal
# We give Glaucoma (Index 2) a much higher weight
class_weights = {
    0: 1.0,  # Cataract
    1: 1.0,  # Diabetic Retinopathy
    2: 7.0,  # GLAUCOMA (The target fix)
    3: 5.8  # Normal
}

print("🚀 Starting Surgical Fine-Tuning...")
model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save('eye_master_surgical_tuned_v3.keras')
print("✅ Done! This model should now have higher Glaucoma precision.")