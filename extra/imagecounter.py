import os

def count_images_in_tree(base_path):
    # Common image extensions for medical datasets
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    print(f"{'Folder Path':<60} | {'Image Count':<12}")
    print("-" * 75)
    
    total_images = 0
    folder_stats = []

    # os.walk traverses everything inside the directory
    for root, dirs, files in os.walk(base_path):
        # Count only files that end with the valid extensions
        image_count = len([f for f in files if f.lower().endswith(valid_extensions)])
        
        if image_count > 0:
            print(f"{root:<60} | {image_count:<12}")
            total_images += image_count
            folder_stats.append((root, image_count))

    print("-" * 75)
    print(f"{'TOTAL IMAGES FOUND:':<60} | {total_images:<12}")

# --- Set your path here ---
my_dataset_path = 'router datset' # or 'dataset_fundus'
count_images_in_tree(my_dataset_path)