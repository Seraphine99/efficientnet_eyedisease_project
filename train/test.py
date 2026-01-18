# Save this as 'create_dummy_images.py' inside your project folder
import numpy as np
from PIL import Image
import os

# Define the structure and image size
BASE_DIR = '/Users/samridda/efficientnet_cataract_project/dummy_data'
SUBDIRS = ['train', 'validation']
CLASSES = ['Class_A', 'Class_B']
NUM_IMAGES = 3  # Create 3 images per folder
IMG_SIZE = (224, 224) # The size your model expects

for subdir in SUBDIRS:
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, subdir, cls)
        
        for i in range(NUM_IMAGES):
            # Create a simple image array (e.g., a random grayscale image)
            # Class A gets mostly high values (whiteish noise)
            if cls == 'Class_A':
                data = np.random.randint(200, 256, size=IMG_SIZE, dtype=np.uint8)
            # Class B gets mostly low values (darkish noise)
            else:
                data = np.random.randint(0, 50, size=IMG_SIZE, dtype=np.uint8)
                
            img = Image.fromarray(data).convert('RGB') # Convert to 3-channel RGB
            img.save(os.path.join(path, f'dummy_img_{i}.png'))

print("Dummy images created successfully in the 'dummy_data' folder.")