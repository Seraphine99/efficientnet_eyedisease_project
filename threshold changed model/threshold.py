import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Model and Data
model = load_model('eye_master_surgical_tuned.keras')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    'dataset_fundus/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. Get the raw probability scores (0.0 to 1.0) instead of just the final answer
print("Analyzing probabilities...")
predictions = model.predict(test_gen)

# 3. SET THE SENSITIVITY THRESHOLD
# 0: Cataract, 1: DR, 2: Glaucoma, 3: Normal
GLAUCOMA_INDEX = 2
THRESHOLD = 0.25  # If the AI is >25% sure it's Glaucoma, we take it.

final_preds = []
for pred in predictions:
    # If Glaucoma probability is above our custom threshold, pick Glaucoma
    if pred[GLAUCOMA_INDEX] > THRESHOLD:
        final_preds.append(GLAUCOMA_INDEX)
    else:
        # Otherwise, pick the class with the highest overall score
        final_preds.append(np.argmax(pred))

# 4. Show the New Report
print(f"REPORT WITH {THRESHOLD*100}% GLAUCOMA THRESHOLD:")
print(classification_report(test_gen.classes, final_preds, target_names=test_gen.class_indices.keys()))