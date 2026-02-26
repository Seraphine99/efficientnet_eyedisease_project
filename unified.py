import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as process_external
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as process_router

# 1. LOAD MODELS
router_model = tf.keras.models.load_model('eye_router_model.keras')
fundus_model = tf.keras.models.load_model('eye_model_final_clahe.keras') # Your CLAHE model
external_model = tf.keras.models.load_model('external_eye_model.keras')

def apply_clahe_logic(img_rgb):
    """
    Applies CLAHE to Fundus images to match the training environment.
    """
    img_uint8 = img_rgb.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def final_prediction_system(img_path):
    # Load raw image
    img_raw = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    
    # --- STAGE 1: ROUTER ---
    router_img = cv2.resize(img_rgb, (224, 224))
    router_input = process_router(np.expand_dims(router_img, axis=0))
    router_preds = router_model.predict(router_input, verbose=0)
    
    types = ['external', 'fundus', 'invalid']
    eye_type = types[np.argmax(router_preds)]
    
    if eye_type == 'invalid':
        return "Rejected: Not a valid eye image."

    # --- STAGE 2: BRANCHING LOGIC ---
    if eye_type == 'fundus':
        # Apply CLAHE + Resizing for Fundus Specialist
        fundus_img = cv2.resize(img_rgb, (300, 300))
        fundus_img = apply_clahe_logic(fundus_img)
        # Assuming your fundus model used EfficientNet Preprocess
        final_input = process_external(np.expand_dims(fundus_img, axis=0))
        
        preds = fundus_model.predict(final_input, verbose=0)
        labels = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        
    else: # External Eye
        # NO CLAHE - Just standard resize and preprocess
        ext_img = cv2.resize(img_rgb, (224, 224))
        final_input = process_external(np.expand_dims(ext_img, axis=0))
        
        preds = external_model.predict(final_input, verbose=0)
        labels = ['Cataract', 'Conjunctivitis', 'Normal', 'Pterygium']

    # Output Result
    res_idx = np.argmax(preds)
    return f"View: {eye_type.upper()} | Result: {labels[res_idx]} ({np.max(preds)*100:.2f}%)"

# --- EXECUTION BLOCK ---
# 1. Update this path to match your actual test image file
test_image_path = '/Users/samridda/efficientnet_cataract_project/router datset/invalid/image1.jpg' 

# 2. Call the function
result = final_prediction_system(test_image_path)

# 3. Print the outcome
print(result)