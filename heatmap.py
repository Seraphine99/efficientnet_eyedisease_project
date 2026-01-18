import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETTINGS ---
DATA_DIR = 'dataset_fundus'
IMG_SIZE = (224, 224)
MODEL_PATH = 'eye_master_v1_final.keras'
TARGET_DISEASE = 'glaucoma'  # Change to 'cataract', 'normal', etc. to see others

# --- 2. GRAD-CAM FUNCTION (Handles Nested EfficientNet) ---
def make_gradcam_heatmap(img_input, model):
    # Reach inside the Sequential model to get the EfficientNet base
    base_model = model.get_layer('efficientnetb0')
    last_conv_layer_name = "top_activation"
    
    # Create a model that maps the base input to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        base_model.inputs, 
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, base_preds = grad_model(img_input)
        
        # Pass the base output through the Sequential top layers
        x = model.get_layer('global_average_pooling2d')(last_conv_layer_output)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dropout')(x)
        preds = model.get_layer('dense')(x)
        
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 3. EXECUTION ---
print("📂 Loading Model...")
model = load_model(MODEL_PATH)

# Find a real image automatically
test_class_dir = os.path.join(DATA_DIR, 'test', TARGET_DISEASE)
available_images = [f for f in os.listdir(test_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not available_images:
    print(f"❌ No images found in {test_class_dir}")
else:
    img_path = os.path.join(test_class_dir, available_images[0])
    print(f"🖼️ Analyzing: {img_path}")

    # Correctly define img_array
    img = load_img(img_path, target_size=IMG_SIZE)
    img_raw = img_to_array(img)
    img_array = np.expand_dims(img_raw, axis=0)
    img_array = preprocess_input(img_array)

    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # Visualization
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_resize = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap_resize), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_cv, 0.4, 0)

    # Show Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {TARGET_DISEASE}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("AI Focus (Grad-CAM)")
    plt.axis('off')
    
    plt.show()
    print("✅ Done!")