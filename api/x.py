import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your model
model = load_model('eye_master_surgical_tuned.keras')

# Save weights with the EXACT extension required
model.save_weights('eye_weights.weights.h5') 

print("Weights saved successfully as eye_weights.weights.h5")