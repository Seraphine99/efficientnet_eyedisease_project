import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as process_external
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as process_router

# 1. LOAD MODELS
print("Loading system models...")
router_model = tf.keras.models.load_model('eye_router_model.keras')
fundus_model = tf.keras.models.load_model('eye_model_final_clahe.keras')
external_model = tf.keras.models.load_model('external_eye_model.keras')

# 2. CONFIGURATION
ROUTER_LABELS = ['external', 'fundus', 'invalid']
FUNDUS_LABELS = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
EXTERNAL_LABELS = ['Cataract', 'Conjunctivitis', 'Normal', 'Pterygium']

def run_eye_screening(image_path):
    # Load image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print("Error: Could not find image file.")
        return
    
    # Pre-process for models
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    # --- STAGE 1: ROUTING ---
    # Prepare input for the MobileNetV2 Gatekeeper
    router_input = process_router(img_array.copy())
    router_preds = router_model.predict(router_input, verbose=0)
    
    img_type_idx = np.argmax(router_preds)
    img_type = ROUTER_LABELS[img_type_idx]
    router_conf = np.max(router_preds)

    print("-" * 40)
    print(f"STEP 1: Analysis - {img_type.upper()} detected ({router_conf*100:.2f}%)")

    # --- STAGE 2: VALIDATION CHECK ---
    if img_type == 'invalid' or router_conf < 0.70:
        print("RESULT: Access Denied. Please upload a clear eye photograph.")
        print("-" * 40)
        return

    # --- STAGE 3: SPECIALIZED DIAGNOSIS ---
    if img_type == 'fundus':
        # Fundus model uses 1/255 scaling
        diag_input = img_array / 255.0
        preds = fundus_model.predict(diag_input, verbose=0)
        labels = FUNDUS_LABELS
    else:
        # External model uses EfficientNet preprocessing
        diag_input = process_external(img_array.copy())
        preds = external_model.predict(diag_input, verbose=0)
        labels = EXTERNAL_LABELS

    # Final Output
    final_idx = np.argmax(preds)
    diagnosis = labels[final_idx]
    confidence = np.max(preds) * 100

    print("STEP 2: Specialized Diagnosis Complete")
    print("=" * 40)
    print(f"FINAL REPORT")
    print(f"View Type : {img_type.title()}")
    print(f"Condition : {diagnosis}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 40)

# Example usage:
# run_eye_screening('path_to_your_test_image.jpg')