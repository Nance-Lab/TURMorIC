import os
import shutil
import random
from collections import defaultdict

"""
Takes in a directory of images to separate them into training and testing data.

The image directory is passed into the function along with groups like
brain region or sex as well as treatment conditions of the slices. The
function then splits the images into a 80:20 training and testing data
without data leakage. The data is grouped by slice and treatment
conditions in new training and testing directories.

Parameters:
    base_dir: The directory of all images to be used for training and testing.
    groups: The variable groups of the images like brain region or subject sex.
    treatment_conditions: The treatment applied to the slices.

Returns:
    train_dir: The directory of training files from an 80:20 split from the base directory.
    test_dir: The directory of testing files from an 80:20 split from the base directory.
"""
# Function to organize files into training and testing folders without slice leakage
def organize_files_without_leakage(base_dir, train_dir, test_dir, groups, treatment_conditions, test_size=0.2):
    for group in groups:
        for condition in treatment_conditions:
            condition_path = os.path.join(base_dir, group, condition)
            if not os.path.exists(condition_path):
                continue
            
            print(f'processing {group} {condition} :)')

            # Group files by brain slice
            slice_files = defaultdict(list)
            for file in os.listdir(condition_path):
                if os.path.isfile(os.path.join(condition_path, file)):
                    # Extract slice_id based on the naming pattern
                    slice_id = "_".join(file.split("_")[2:3])  # Extract the third element (Slice204)
                    slice_files[slice_id].append(file)
            
            # Split slices into training and testing
            slice_ids = list(slice_files.keys())
            random.seed(42)  # For reproducibility
            random.shuffle(slice_ids)
            
            split_index = int(len(slice_ids) * (1 - test_size))
            train_slices = slice_ids[:split_index]
            test_slices = slice_ids[split_index:]
            
            # Create subdirectories for training and testing
            train_subdir = os.path.join(train_dir, group, condition)
            test_subdir = os.path.join(test_dir, group, condition)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)
            
            # Move files to the appropriate folders
            for slice_id in train_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(os.path.join(condition_path, file), os.path.join(train_subdir, file))
            
            for slice_id in test_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(os.path.join(condition_path, file), os.path.join(test_subdir, file))



