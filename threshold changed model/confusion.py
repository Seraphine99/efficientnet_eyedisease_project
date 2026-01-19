import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# --- SETTINGS ---
DATA_DIR = 'dataset_fundus'
IMG_SIZE = (300, 300)
MODEL_NAME = 'eye_model_final_clahe.keras'

# --- DATA PREP ---
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
class_labels = list(test_gen.class_indices.keys())

# --- EVALUATE ---
print("📂 Loading Surgically Tuned Model...")
model = load_model(MODEL_NAME)

print("🧠 Generating Predictions...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# --- PLOT CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Final Confusion Matrix (After Surgical Fine-Tuning)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --- PRINT FINAL REPORT ---
print("\n📝 FINAL SURGICAL TUNING REPORT:")
print(classification_report(y_true, y_pred, target_names=class_labels))