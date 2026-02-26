import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. THE CLAHE PREPROCESSOR ---
def apply_clahe_rgb(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return preprocess_input(final_img.astype(np.float32))

# --- 2. SETUP ---
IMG_SIZE = (300, 300)
MODEL_PATH = 'eye_model_final_clahe.keras' 
model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe_rgb)
test_gen = test_datagen.flow_from_directory(
    'raw_data',
    target_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

# --- 3. PREDICT ---
print("Generating Predictions...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# --- 4. VISUALIZE ---
print("Generating Confusion Matrix Plot...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Final Model Confusion Matrix (CLAHE + Weights)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('final_confusion_matrix.png')
plt.show()

print("FINAL MASTER REPORT:")
print(classification_report(y_true, y_pred, target_names=class_labels))
from sklearn.metrics import f1_score

# --- 5. GENERATE F1-SCORE GRAPH ---
print("Generating F1-Score Report...")
f1_scores = f1_score(y_true, y_pred, average=None)

plt.figure(figsize=(10, 6))
colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'] # Distinct colors for each class
plt.bar(class_labels, f1_scores, color=colors)

# Add value labels on top of bars
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

plt.ylim(0, 1.1) # Give some space at the top
plt.title('Final Model F1-Scores per Class')
plt.ylabel('F1-Score')
plt.xlabel('Condition')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('f1_score_report.png')
plt.show()

print("F1 Graph saved as 'f1_score_report.png'")