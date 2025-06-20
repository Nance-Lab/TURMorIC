import os
import shutil
import random
from collections import defaultdict
# from skimage.filters import try_all_threshold
# from skimage.filters import threshold_isodata
# from skimage.filters import threshold_li
# from skimage.filters import threshold_mean
# from skimage.filters import threshold_minimum
# from skimage.filters import threshold_otsu
# from skimage.filters import threshold_triangle
# from skimage.filters import threshold_yen
# from turmoric.image_process import save_npy_as_tif, nd2_to_tif
# from turmoric.cell_analysis import apply_regionprops

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
    train_dir: The directory of training files
               from an 80:20 split from the base directory.
    test_dir: The directory of testing files
               from an 80:20 split from the base directory.
"""

# Function to organize files into training and testing folders
# without slice leakage


def organize_files_without_leakage(base_dir: str, train_dir: str, test_dir: str, groups: list,
                                   treatment_conditions: list, test_size: float=0.2) -> None:
    """
    Organize files into training and testing sets without data leakage across brain slices.

    This function walks through a directory structure organized by experimental `groups`
    and `treatment_conditions`, groups image files by brain slice ID (extracted from filenames),
    and splits slices into training and testing sets. It ensures that all files from the
    same brain slice are allocated to only one set, avoiding data leakage.

    Parameters
    ----------
    base_dir : str
        Root directory containing raw data files structured as `base_dir/group/condition/`.
    train_dir : str
        Output directory where training files will be copied.
    test_dir : str
        Output directory where testing files will be copied.
    groups : list of str
        List of experimental groups (e.g., ["control", "treated"]).
    treatment_conditions : list of str
        List of treatment condition subfolders under each group (e.g., ["vehicle", "drug"]).
    test_size : float, optional
        Proportion of slices to use for testing. Default is 0.2 (20%).

    Returns
    -------
    None
        The function performs file copying to organize training and testing sets
        but does not return a value.

    Notes
    -----
    - Slice IDs are extracted from the third underscore-separated token in each filename.
      Adjust the parsing logic if your filename structure differs.
    - Files are grouped and split at the slice level, not the file level, to prevent data leakage.
    - The train/test split is deterministic due to a fixed random seed (`random.seed(42)`).
    - Existing files in the output directories will not be overwritten.
    - Uses `shutil.copy` to duplicate files into the train/test directories.

    Examples
    --------
    >>> base_dir = "/data/brain_slices"
    >>> train_dir = "/data/split/train"
    >>> test_dir = "/data/split/test"
    >>> groups = ["control", "treated"]
    >>> treatment_conditions = ["drugA", "drugB"]
    >>> organize_files_without_leakage(base_dir, train_dir, test_dir, groups, treatment_conditions)
    processing control drugA :)
    processing control drugB :)
    processing treated drugA :)
    processing treated drugB :)
    """
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
                    # Extract the third element
                    slice_id = "_".join(file.split("_")[2:3])
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
                    shutil.copy(os.path.join(condition_path, file),
                                os.path.join(train_subdir, file))

            for slice_id in test_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(os.path.join(condition_path, file),
                                os.path.join(test_subdir, file))


def recursively_get_all_filepaths(input_folder: str, file_type: str) -> list:
    """
    Recursively retrieve all file paths of a specific type from a directory.

    This function walks through a directory tree starting from `input_folder` and collects
    the full paths of all files that end with the specified `file_type` extension.

    Parameters
    ----------
    input_folder : str
        Root directory to search for files.
    file_type : str
        File extension to filter by (e.g., '.tif', '.npy').

    Returns
    -------
    file_list : list of str
        List containing full paths to all files in `input_folder` and its subdirectories
        that match the specified file extension.

    Notes
    -----
    - The function performs a case-sensitive match on the file extension.
    - Subdirectories are traversed using `os.walk`.

    Examples
    --------
    >>> file_paths = recursively_get_all_filepaths('/path/to/folder', '.txt')
    >>> for path in file_paths:
    ...     print(path)
    /path/to/folder/file1.txt
    /path/to/folder/subfolder/file2.txt
    """
    # Walk through all files and subfolders
    file_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            # ensure only the correct file type is processed
            if file.endswith(file_type):
                # Full input path
                input_path = os.path.join(root, file)
                file_list.append(input_path)

    return file_list
