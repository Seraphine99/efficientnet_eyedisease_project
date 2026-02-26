import splitfolders
import os

# 1. Define your paths
input_folder = "external_eye" 
output_folder = "external_dataset"

def count_files(directory):
    """Helper function to count files in subdirectories"""
    summary = {}
    for root, dirs, files in os.walk(directory):
        if root == directory:
            continue
        category = os.path.basename(root)
        summary[category] = len(files)
    return summary

# --- BEFORE SPLIT ---
print("--- Dataset Summary (Before Split) ---")
initial_counts = count_files(input_folder)
for category, count in initial_counts.items():
    print(f"Category [{category}]: {count} images")
print(f"Total Images: {sum(initial_counts.values())}\n")

# --- PERFORM SPLIT ---
# Split 80% Train, 20% Test
splitfolders.ratio(input_folder, output=output_folder, 
                   seed=1337, ratio=(0.8,0,0.2), 
                   move=False)

# --- AFTER SPLIT ---
print("--- Dataset Summary (After Split) ---")
train_counts = count_files(os.path.join(output_folder, 'train'))
test_counts = count_files(os.path.join(output_folder, 'test'))

print(f"{'Category':<20} | {'Train (80%)':<12} | {'Test (20%)':<12}")
print("-" * 50)
for category in initial_counts.keys():
    tr = train_counts.get(category, 0)
    ts = test_counts.get(category, 0)
    print(f"{category:<20} | {tr:<12} | {ts:<12}")

print(f"\nTotal Processed: {sum(train_counts.values()) + sum(test_counts.values())}")