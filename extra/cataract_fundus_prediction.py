import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# 1. Load the model
model = tf.keras.models.load_model('cataract_expert_v3_final.keras')

def predict_eye(img_path):
    # 2. Load and Prepare the Image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Make it (1, 224, 224, 3)
    img_array = preprocess_input(img_array)       # Must use same preprocessing as training!

    # 3. Predict
    prediction = model.predict(img_array)
    
    # 4. Result (0 is typically Cataract, 1 is Normal based on folder alphabet)
    if prediction[0][0] < 0.5:
        print(f"🔍 Result: CATARACT (Confidence: {100 - (prediction[0][0]*100):.2f}%)")
    else:
        print(f"🔍 Result: NORMAL (Confidence: {prediction[0][0]*100:.2f}%)")

# Test it on an image from your 'test' folder
test_image = '/Users/samridda/efficientnet_cataract_project/raw_data/cataract/_96_5515894.jpg' # Update this to a real filename
if os.path.exists(test_image):
    predict_eye(test_image)
else:
    print("Oops! Please update the test_image path to an actual file in your test folder.")