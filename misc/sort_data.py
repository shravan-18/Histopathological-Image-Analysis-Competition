import pandas as pd
import os
import shutil

df = pd.read_csv('train_labels.csv')

# Directory paths
source_dir = 'train'
target_dir = 'Root'

# Iterate through the filenames in the sampled_df DataFrame
for _, row in df.iterrows():
    filename = row['id']
    
    # Add the.tif extension
    full_filename = filename + '.tif'
    
    # Construct the full source and target paths
    src_path = os.path.join(source_dir, full_filename)
    tgt_path = os.path.join(target_dir, str(row['label']), full_filename)
    
    # Copy the file
    shutil.copy(src_path, tgt_path)

print("Sorted Data Successfully...")
