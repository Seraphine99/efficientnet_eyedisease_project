import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# --- 1. SETTINGS ---
IMG_SIZE = (380, 380) # B4 Native resolution
BATCH_SIZE = 8        # B4 is heavy, lower batch size to avoid memory errors
EPOCHS = 10

# --- 2. CLAHE PREPROCESSOR ---
def apply_clahe_rgb(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return tf.keras.applications.efficientnet.preprocess_input(final_img.astype(np.float32))

# --- 3. MODEL ARCHITECTURE ---
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x) # Slightly higher dropout for B4 to prevent overfitting
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe_rgb,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe_rgb)

train_gen = train_datagen.flow_from_directory('dataset_fundus/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = test_datagen.flow_from_directory('dataset_fundus/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# --- 5. COMPILE & TRAIN ---
# Glaucoma Weight increased slightly more to push for that 90%
class_weights = {0: 1.0, 1: 1.0, 2: 5.0, 3: 0.8} 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Training B4 Master Model...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, class_weight=class_weights)

model.save('eye_model_b4_master.keras')