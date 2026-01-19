import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE CLAHE FUNCTION ---
def apply_clahe_rgb(img):
    # The generator provides images in float32, we need uint8 for OpenCV
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    # Convert back to float and apply EfficientNet preprocessing
    return preprocess_input(final_img.astype(np.float32))

# --- 2. SETUP ---
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
MODEL_PATH = 'eye_model_b3_weights.keras' 
model = load_model(MODEL_PATH)

# Weight 2 (Glaucoma) is set to 4.0 to force the model to prioritize it
class_weights = {0: 1.0, 1: 1.0, 2: 4.0, 3: 0.6} 

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe_rgb, 
    rotation_range=20,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe_rgb)

train_gen = train_datagen.flow_from_directory('dataset_fundus/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = test_datagen.flow_from_directory('dataset_fundus/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# --- 4. THE FINAL RUN ---
print("Running Final CLAHE-Enhanced Training...")
history = model.fit(train_gen, validation_data=val_gen, epochs=8, class_weight=class_weights)

model.save('eye_model_final_clahe.keras')
print("MASTER MODEL SAVED: eye_model_final_clahe.keras")

# --- 5. GENERATE LOSS & ACCURACY GRAPHS ---
def plot_final_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Final Model Accuracy (CLAHE + Weights)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Final Model Loss (CLAHE + Weights)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('final_clahe_report.png')
    plt.show()
    print("Final report saved as 'final_clahe_report.png'")

plot_final_results(history)