import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from turmoric.utils import recursively_get_all_filepaths


def apply_regionprops(file, properties_list):

    # Load the binary mask
    binary_mask = np.load(file)

    # Label connected regions in the binary mask
    label_image = label(binary_mask)

    # Measure properties
    props = regionprops_table(label_image, properties=properties_list)

    # Create a DataFrame for the current file
    props_df = pd.DataFrame(props)
    props_df['filename'] = file  # Add filename column

    return props_df


def apply_regionprops_recursively(input_folder, properties_list=(
                                'area', 'bbox_area', 'centroid', 'convex_area',
                                'eccentricity', 'equivalent_diameter',
                                'euler_number', 'extent', 'filled_area',
                                'major_axis_length', 'minor_axis_length',
                                'orientation', 'perimeter', 'solidity')):
    """
    Processes all `.npy` files in the input folder (and subfolders),
    applies labeling, calculates properties, and saves results in a CSV file.

    Parameters:
    - input_folder: Path to the folder containing `.npy` files.
    - properties_list: List of properties to calculate using regionprops_table.
    """

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    all_dataframes = []  # List to store individual DataFrames

    # Recursively walk through input folder and process .npy files
    file_list = recursively_get_all_filepaths(input_folder, '.npy')
    for file in file_list:
        try:
            all_dataframes.append(apply_regionprops(file, properties_list))

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pd.concat(all_dataframes, ignore_index=True)
