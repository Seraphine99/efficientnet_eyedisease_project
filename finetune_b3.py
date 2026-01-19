import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# --- 1. SETUP ---
IMG_SIZE = (300, 300)
BATCH_SIZE = 16 # Lowered to avoid memory errors on Mac
MODEL_PATH = 'eye_model_b3_tuned.keras'

# Load the model we just trained
model = load_model(MODEL_PATH)
class_weights = {
    0: 1.0,  # Cataract
    1: 1.0,  # DR
    2: 3.5,  # HEAVY FOCUS ON GLAUCOMA
    3: 0.7   # Slightly de-emphasize Normal to reduce False Negatives
}

# --- 2. SURGICAL UNFREEZE ---
# Reach into the nested efficientnetb3 layer
base_model = model.get_layer('efficientnetb3')
base_model.trainable = True

# Fine-tuning depth: Unfreeze the last 40 layers
# B3 has about 380+ layers total
for layer in base_model.layers[:-40]:
    layer.trainable = False

# --- 3. RECOMPILE WITH LOW LEARNING RATE ---
# Essential: A very small learning rate prevents 'breaking' the ImageNet weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3, # B3 handles zoom well at 300px
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    'dataset_fundus/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = test_datagen.flow_from_directory(
    'dataset_fundus/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- 5. FINE-TUNING ---
print("Starting B3 Surgical Fine-Tuning...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    class_weight=class_weights
)

# --- 6. SAVE & PLOT ---
model.save('eye_model_b3_weights.keras')

# Create a side-by-side plot for Accuracy and Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy Plot
ax1.plot(history.history['accuracy'], label='Train Acc', marker='o')
ax1.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
ax1.set_title('B3 Fine-Tuning Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss Plot
ax2.plot(history.history['loss'], label='Train Loss', marker='o')
ax2.plot(history.history['val_loss'], label='Val Loss', marker='o')
ax2.set_title('B3 Fine-Tuning Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('b3_finetune_report.png')
plt.show()

print("Fine-tuning complete! Model saved and report generated.")