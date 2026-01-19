import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. LOAD THE MODEL
model_path = 'eye_expert_v1.keras'
print(f"🚀 Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

# 2. PREPARE TEST DATA
# Ensure your directory matches the one used in train_multi.py
test_dir = 'dataset_fundus/test' 

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # CRITICAL: Keep False to match labels with predictions
)

# 3. GENERATE PREDICTIONS
print("🧠 Generating predictions...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# --- VISUAL 1: MULTI-CLASS CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix: Eye Expert V1')
plt.ylabel('Actual Disease')
plt.xlabel('AI Predicted Disease')
plt.show()

# --- VISUAL 2: DETAILED F1-SCORE REPORT ---
print("\n📝 CLASSIFICATION REPORT:")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Global F1-Score
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"⭐ Overall Weighted F1-Score: {weighted_f1:.4f}")

# --- VISUAL 3: F1-SCORE BAR CHART ---
plt.figure(figsize=(10, 6))
f1_per_class = f1_score(y_true, y_pred, average=None)
plt.bar(class_labels, f1_per_class, color='skyblue', edgecolor='navy')
plt.title('F1-Score per Eye Condition')
plt.ylabel('F1-Score')
plt.ylim(0, 1.1)
for i, v in enumerate(f1_per_class):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()