import os
import random
import shutil

def create_subset(source_dir, dest_dir, num_samples):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Get all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Randomly sample files
    sampled_files = random.sample(files, min(num_samples, len(files)))
    
    # Copy files to the destination directory
    for file_name in sampled_files:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
    print(f"Copied {len(sampled_files)} files from {source_dir} to {dest_dir}")

# Paths to the full dataset
full_train_good = 'data/train/good'
full_train_bad = 'data/train/bad'
full_test_good = 'data/test/good'
full_test_bad = 'data/test/bad'

# Paths to the smaller dataset
subset_train_good = 'data_small/train/good'
subset_train_bad = 'data_small/train/bad'
subset_test_good = 'data_small/test/good'
subset_test_bad = 'data_small/test/bad'

# Create subsets
create_subset(full_train_good, subset_train_good, 600)
create_subset(full_train_bad, subset_train_bad, 200)
create_subset(full_test_good, subset_test_good, 150)
create_subset(full_test_bad, subset_test_bad, 50)
