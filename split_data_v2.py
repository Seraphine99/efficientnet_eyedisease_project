import splitfolders
import os

# 1. Configuration
input_folder = "raw_data" 
output_folder = "cataract_dataset_fundus"

# 2. Perform the Split
print(f"🚀 Splitting images from '{input_folder}' into '{output_folder}'...")
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1))

# 3. Summary Function
def print_dataset_summary(base_path):
    print("\n📊 DATASET STRUCTURE & COUNTS:")
    print("=" * 45)
    
    # Sort folders to keep Train/Val/Test order
    for split in sorted(os.listdir(base_path)):
        split_path = os.path.join(base_path, split)
        
        # Skip hidden files like .DS_Store
        if not os.path.isdir(split_path):
            continue
            
        print(f"📁 {split.upper()}")
        
        # Count images in each disease subfolder
        for disease in sorted(os.listdir(split_path)):
            disease_path = os.path.join(split_path, disease)
            if os.path.isdir(disease_path):
                # Count common image extensions
                images = [f for f in os.listdir(disease_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"   ├── {disease:<20} : {len(images)} images")
        print("-" * 45)

# 4. Run the summary
print_dataset_summary(output_folder)
print("✅ Folders created and verified successfully!")