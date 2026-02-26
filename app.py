import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.applications.efficientnet import preprocess_input as process_external
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as process_router

# 1. LOAD MODELS
print("System initializing... Loading all models.")
router_model = tf.keras.models.load_model('eye_router_model.keras')
fundus_model = tf.keras.models.load_model('eye_model_final_clahe.keras')
external_model = tf.keras.models.load_model('external_eye_model.keras')

# 2. LABELS
ROUTER_LABELS = ['external', 'fundus', 'invalid']
FUNDUS_LABELS = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
EXTERNAL_LABELS = ['Cataract', 'Conjunctivitis', 'Normal', 'Pterygium']

def apply_clahe_logic(img_rgb):
    img_uint8 = img_rgb.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def predict_eye_app(input_img):
    if input_img is None:
        return "No image uploaded", "N/A", 0.0

    # Convert Gradio input to RGB
    img_rgb = input_img.astype(np.uint8)
    
    # --- STAGE 1: ROUTER ---
    router_img = cv2.resize(img_rgb, (224, 224))
    router_input = process_router(np.expand_dims(router_img, axis=0))
    router_preds = router_model.predict(router_input, verbose=0)
    
    eye_type = ROUTER_LABELS[np.argmax(router_preds)]
    router_conf = np.max(router_preds)

    if eye_type == 'invalid' or router_conf < 0.65:
        return "REJECTED", "Invalid Image Detected", router_conf

    # --- STAGE 2: BRANCHING ---
    if eye_type == 'fundus':
        # Fundus Path: Resize 300x300 + CLAHE
        spec_img = cv2.resize(img_rgb, (300, 300))
        spec_img = apply_clahe_logic(spec_img)
        final_input = process_external(np.expand_dims(spec_img, axis=0))
        labels = FUNDUS_LABELS
        preds = fundus_model.predict(final_input, verbose=0)
    else:
        # External Path: Resize 224x224
        spec_img = cv2.resize(img_rgb, (224, 224))
        final_input = process_external(np.expand_dims(spec_img, axis=0))
        labels = EXTERNAL_LABELS
        preds = external_model.predict(final_input, verbose=0)

    diagnosis = labels[np.argmax(preds)]
    diag_conf = np.max(preds)

    return eye_type.upper(), diagnosis, float(diag_conf)

# 3. GRADIO INTERFACE


interface = gr.Interface(
    fn=predict_eye_app,
    inputs=gr.Image(label="Upload Eye Image (Fundus or External)"),
    outputs=[
        gr.Textbox(label="Detection Phase (Router Result)"),
        gr.Textbox(label="Medical Diagnosis"),
        gr.Number(label="Diagnosis Confidence Score")
    ],
    title="Ocular Disease Screening System",
    description="Multi-stage AI: First, the system validates the image type. Then, it routes to a specialized specialist model."
)

if __name__ == "__main__":
    interface.launch()