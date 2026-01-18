import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2 # <-- New: For Grad-CAM visualization

# --- CONFIGURATION ---
MODEL_PATH = 'eye_master_surgical_tuned.keras'
IMG_SIZE = (224, 224)
CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
THRESHOLD = 0.25 # Our clinical sensitivity fix
LAST_CONV_LAYER_NAME = "top_activation" # This is typical for EfficientNetB0

# Load Model
print("📂 Loading Model...")
model = load_model(MODEL_PATH)

# --- GRAD-CAM FUNCTIONS ---
def generate_gradcam(model, img_array, last_conv_layer_name):
    """Generates a Grad-CAM heatmap by reaching into the nested EfficientNet layer."""
    # 1. Get the base efficientnet model (the first layer of your loaded model)
    base_model = model.get_layer('efficientnetb0')
    
    # 2. Create a model that maps the input to the activations of the last conv layer 
    # and the output of the base model
    inner_grad_model = tf.keras.models.Model(
        [base_model.inputs], 
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # 3. To get the final gradients, we need to connect the base model to your custom top layers
    with tf.GradientTape() as tape:
        # Get base model outputs
        last_conv_layer_output, base_output = inner_grad_model(img_array)
        
        # Pass base output through your custom top layers (pooling, dropout, dense, etc.)
        x = model.get_layer('global_average_pooling2d')(base_output)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dropout')(x)
        preds = model.get_layer('dense')(x)
        
        top_pred_index = tf.argmax(preds[0])
        class_channel = preds[:, top_pred_index]

    # 4. Standard Grad-CAM math
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def apply_heatmap(heatmap, original_img_pil, alpha=0.4):
    """Applies the heatmap onto the original PIL image."""
    original_img_np = np.array(original_img_pil) # Convert PIL to NumPy
    
    # Rescale heatmap to 0-255 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to original image size
    jet_heatmap = cv2.resize(jet_heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    
    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + original_img_np
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8) # Clip to 0-255 range
    
    return Image.fromarray(superimposed_img) # Convert back to PIL

# --- PREDICTION FUNCTION ---
def predict_image(file_path):
    img_pil = Image.open(file_path).convert('RGB')
    img_resized_pil = img_pil.resize(IMG_SIZE)
    
    img_array = np.array(img_resized_pil)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)

    # Predict
    preds = model.predict(img_preprocessed)[0]
    
    # Apply Glaucoma Threshold
    if preds[2] > THRESHOLD:
        result_idx = 2
    else:
        result_idx = np.argmax(preds)
    
    label_text = f"Diagnosis: {CLASSES[result_idx]}\n"
    label_text += f"Confidence: {preds[result_idx]*100:.1f}%"

    # Generate Heatmap
    heatmap = generate_gradcam(model, img_preprocessed, LAST_CONV_LAYER_NAME)
    heatmap_pil = apply_heatmap(heatmap, img_resized_pil)

    return label_text, img_resized_pil, heatmap_pil # Return both images

# --- UI SETUP ---
def drop(event):
    file_path = event.data.strip().strip('{}') 
    if file_path.startswith('"') and file_path.endswith('"'): # For paths with spaces
        file_path = file_path[1:-1]

    print(f"📂 Processing: {file_path}")

    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            result_text, original_img_pil, heatmap_img_pil = predict_image(file_path)
            
            # Update Original Image Display
            img_tk_orig = ImageTk.PhotoImage(original_img_pil)
            original_image_label.config(image=img_tk_orig)
            original_image_label.image = img_tk_orig
            
            # Update Heatmap Image Display
            img_tk_heatmap = ImageTk.PhotoImage(heatmap_img_pil)
            heatmap_image_label.config(image=img_tk_heatmap)
            heatmap_image_label.image = img_tk_heatmap # Keep a reference!
            
            # Update Text
            result_label.config(text=result_text, fg="#2E7D32" if "Normal" in result_text else "#C62828")
            
            root.update_idletasks() # Refresh UI
            
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}", fg="red")
            print(f"Prediction Error: {e}")
    else:
        result_label.config(text="Please drop an image file!", fg="orange")

root = TkinterDnD.Tk()
root.title("VisionEye: Explainable Disease Classifier")
root.geometry("800x650") # Make window wider for two images
root.config(bg="#f0f0f0")

title_lbl = tk.Label(root, text="VisionEye AI Diagnosis", font=("Arial", 20, "bold"), bg="#f0f0f0")
title_lbl.pack(pady=10)

# Frame for Drop Zone and Image Displays
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(pady=5)

# Drop Zone
drop_frame = tk.Label(main_frame, text="\nDRAG & DROP\nIMAGE HERE", 
                      font=("Arial", 12), bg="#ffffff", 
                      width=20, height=5, relief="ridge")
drop_frame.grid(row=0, column=0, columnspan=2, pady=10)
drop_frame.drop_target_register(DND_FILES)
drop_frame.dnd_bind('<<Drop>>', drop)

# Image Display Frame
image_display_frame = tk.Frame(main_frame, bg="#f0f0f0")
image_display_frame.grid(row=1, column=0, columnspan=2, pady=10)

# Original Image Label
tk.Label(image_display_frame, text="Original Image", font=("Arial", 10, "bold"), bg="#f0f0f0").grid(row=0, column=0, padx=5)
original_image_label = tk.Label(image_display_frame, bg="#f0f0f0", borderwidth=2, relief="groove")
original_image_label.grid(row=1, column=0, padx=10)

# Heatmap Image Label
tk.Label(image_display_frame, text="AI Focus (Heatmap)", font=("Arial", 10, "bold"), bg="#f0f0f0").grid(row=0, column=1, padx=5)
heatmap_image_label = tk.Label(image_display_frame, bg="#f0f0f0", borderwidth=2, relief="groove")
heatmap_image_label.grid(row=1, column=1, padx=10)


result_label = tk.Label(root, text="Waiting for image...", font=("Arial", 16, "bold"), bg="#f0f0f0")
result_label.pack(pady=20)

# --- COLOR SPECTRUM LEGEND ---
legend_frame = tk.Frame(root, bg="#f0f0f0")
legend_frame.pack(pady=10)

# Labels for the spectrum
tk.Label(legend_frame, text="Low Focus", font=("Arial", 9), bg="#f0f0f0").grid(row=0, column=0)
tk.Label(legend_frame, text="High Focus", font=("Arial", 9), bg="#f0f0f0").grid(row=0, column=2)

# Create the Gradient Canvas
canvas_width = 200
canvas_height = 20
spec_canvas = tk.Canvas(legend_frame, width=canvas_width, height=canvas_height, 
                        highlightthickness=0, bg="#f0f0f0")
spec_canvas.grid(row=0, column=1, padx=10)

# Generate the Jet Color Spectrum
def draw_spectrum(canvas, width, height):
    # Create a 1x256 grayscale strip and apply Jet colormap
    gradient = np.linspace(0, 255, 256).astype(np.uint8).reshape(1, 256)
    jet_strip = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    jet_strip = cv2.cvtColor(jet_strip, cv2.COLOR_BGR2RGB) # Fix colors for Tkinter
    
    # Resize to fit canvas
    jet_strip = cv2.resize(jet_strip, (width, height))
    
    # Convert to PhotoImage and display
    spec_img = Image.fromarray(jet_strip)
    spec_tk = ImageTk.PhotoImage(spec_img)
    canvas.create_image(0, 0, anchor="nw", image=spec_tk)
    canvas.image = spec_tk # Keep reference

# Draw it once at startup
draw_spectrum(spec_canvas, canvas_width, canvas_height)

print("🚀 Explainable UI Ready! Drag an image onto the window.")
root.mainloop()