import splitfolders

# Path to your "Raw_Data" folder
input_folder = "/Users/samridda/efficientnet_cataract_project/cataract" 

# Where the organized folders will be created
output_folder = "cataract_dataset_fundus"

# Split with a ratio: 80% Train, 10% Val, 10% Test
# This creates three folders: train, val, and test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1))

print("✅ Folders created successfully!")