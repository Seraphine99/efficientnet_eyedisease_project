import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# 1. LOAD THE CATARACT MODEL
MODEL_PATH = 'cataract_expert_v3_final.keras'
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Please ensure the file is in this folder.")
    exit()

print("Loading pre-trained cataract model...")
# Load as a Sequential model
multi_model = tf.keras.models.load_model(MODEL_PATH)

# 2. THE "BRAIN TRANSPLANT"
# Our previous model had: ... -> GlobalAveragePooling -> Dropout -> Dense(1)
# We pop the last 2 layers to remove the binary setup
multi_model.pop() # Removes the Dense(1) layer
multi_model.pop() # Removes the old Dropout layer

# 3. ADD NEW MULTI-CLASS LAYERS
num_classes = 4 # cataract, diabetic_retinopathy, glaucoma, normal

# Add a fresh Dropout and a new Dense layer with 4 nodes + Softmax
multi_model.add(layers.Dropout(0.5, name="multi_dropout"))
multi_model.add(layers.Dense(num_classes, activation='softmax', name="disease_predictions"))

# 4. COMPILE FOR MULTI-CLASS
# We use a very low learning rate (1e-5) to fine-tune without "forgetting" cataracts
multi_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Model successfully converted to Multi-Class!")
multi_model.summary()

# 5. SETUP DATA GENERATORS
# We use the exact same preprocessing as before
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    'dataset_fundus/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'dataset_fundus/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 6. TRAIN THE MULTI-DISEASE EXPERT
print("Starting Multi-Disease Training...")

# Save the best version as 'eye_expert_v1.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'eye_expert_v1.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

history = multi_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20, # 20 epochs is usually enough for fine-tuning
    callbacks=[checkpoint, early_stop]
)

print("Training Complete! Best model saved as 'eye_expert_v1.keras'")