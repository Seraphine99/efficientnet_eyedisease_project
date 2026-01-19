import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETUP ---
MODEL_PATH = 'eye_master_surgical_tuned.keras'
DATA_DIR = 'dataset_fundus/test'
THRESHOLD = 0.25
GLAUCOMA_INDEX = 2

model = load_model(MODEL_PATH)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(DATA_DIR, target_size=(224, 224), 
                                           batch_size=32, class_mode='categorical', shuffle=False)

# --- 2. PREDICT WITH THRESHOLD ---
probs = model.predict(test_gen)
y_true = test_gen.classes
y_pred = []

for p in probs:
    if p[GLAUCOMA_INDEX] > THRESHOLD:
        y_pred.append(GLAUCOMA_INDEX)
    else:
        y_pred.append(np.argmax(p))

y_pred = np.array(y_pred)
class_labels = list(test_gen.class_indices.keys())

# --- 3. PLOT CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix (Glaucoma Threshold: {THRESHOLD})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('final_confusion_matrix.png')
plt.show()

# --- 4. PLOT F1-SCORE BAR CHART ---
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
f1_scores = [report[label]['f1-score'] for label in class_labels]

plt.figure(figsize=(10, 6))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'] # Different colors for each class
bars = plt.bar(class_labels, f1_scores, color=colors)
plt.ylim(0, 1.0)
plt.title('Final F1-Scores by Disease Category')
plt.ylabel('F1-Score')

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.savefig('f1_score_comparison.png')
plt.show()