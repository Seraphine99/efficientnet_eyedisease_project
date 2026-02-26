import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

# --- 1. ADD THE FUNCTION HERE ---
def build_model():
    # This creates the architecture exactly as it was during training
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = Sequential([
        Input(shape=(224, 224, 3)),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])
    return model

# --- 2. INITIALIZE AND LOAD WEIGHTS ---
print("Loading AI Model architecture and weights...")
model = build_model()
model.load_weights('eye_weights.weights.h5') # Use the exact filename from your Mac

# Define your categories in the correct order
CATEGORIES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return {
            "success": True,
            "diagnosis": CATEGORIES[class_idx],
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}