import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. LOAD MODEL AND TEST DATA
print("🚀 Loading model for full analysis...")
model = tf.keras.models.load_model('cataract_expert_v3_final.keras')

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    'cataract_fundus',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # MUST be False for metrics to align correctly
)

# 2. GENERATE PREDICTIONS
print("🧠 Analyzing test images...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# --- VISUAL 1: CONFUSION MATRIX ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix: Eye Disease Detection')
plt.ylabel('True Disease')
plt.xlabel('AI Predicted Disease')
plt.show()

# --- VISUAL 2: F1-SCORE & DETAILED REPORT ---
print("\n📝 CLASSIFICATION REPORT (F1-Scores per class):")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Calculate global F1-score
overall_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"⭐ Overall Weighted F1-Score: {overall_f1:.4f}")

# --- VISUAL 3: LOSS CURVES (If history is available) ---
# Note: If you just ran training, you can plot 'history.history'
# If not, we typically plot the classification report metrics as a bar chart:
plt.figure(figsize=(10, 6))
f1_per_class = f1_score(y_true, y_pred, average=None)
plt.bar(class_labels, f1_per_class, color='skyblue', edgecolor='navy')
plt.title('F1-Score per Disease Category')
plt.ylabel('F1-Score (0.0 to 1.0)')
plt.ylim(0, 1)
for i, v in enumerate(f1_per_class):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()