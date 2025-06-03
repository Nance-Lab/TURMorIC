import pytest
import random
import os
import pandas as pd
from skimage import io
from skimage.util import img_as_ubyte
import tempfile
from typing import Callable
import numpy as np
import turmoric
from turmoric.cell_analysis import apply_regionprops
from turmoric.cell_analysis import apply_regionprops_recursively


def test_apply_regionprops(input_folder, properties_list):
    # Create a temporary directory with a dummy binary mask
    with tempfile.TemporaryDirectory() as temp_dir:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1  # Create a square region
        mask_path = os.path.join(temp_dir, "mask_01.npy")
        np.save(mask_path, mask)

        # Run the function
        props_df = apply_regionprops(mask_path, properties_list)

        # Assertions
        assert isinstance(props_df, pd.DataFrame), "Output should be a DataFrame"
        assert 'filename' in props_df.columns, "DataFrame should contain 'filename' column"
        assert len(props_df) == 1, "There should be one region in the mask"
        assert all(prop in props_df.columns for prop in properties_list), "DataFrame should contain specified properties"

    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string")
    if not isinstance(properties_list, list):
        raise TypeError("properties_list must be a list of strings")


def test_apply_regionprops_recursively(input_folder, properties_list):
    # Create a temporary directory with multiple binary masks
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(3):
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20*i:20*(i+1), 20*i:20*(i+1)] = 1  # Create square regions
            mask_path = os.path.join(temp_dir, f"mask_{i+1:02d}.npy")
            np.save(mask_path, mask)

        # Run the function
        props_df = apply_regionprops_recursively(temp_dir, properties_list)

        # Assertions
        assert isinstance(props_df, pd.DataFrame), "Output should be a DataFrame"
        assert 'filename' in props_df.columns, "DataFrame should contain 'filename' column"
        assert len(props_df) == 3, "There should be three regions in total"
        assert all(prop in props_df.columns for prop in properties_list), "DataFrame should contain specified properties"

    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string")
    if not isinstance(properties_list, list):
        raise TypeError("properties_list must be a list of strings")
    if not all(isinstance(prop, str) for prop in properties_list):
        raise TypeError("All elements in properties_list must be strings")
    if not all(prop in ['area', 'centroid', 'eccentricity'] for prop in properties_list):
        raise ValueError("properties_list contains invalid property names. Valid names are 'area', 'centroid', 'eccentricity'.")
#         assert binary.shape == (100, 100), "Output shape should match input image shape"
#         assert np.any(binary), "Output should contain some True values"
#     finally:
#         os.remove(temp_path)  # Clean up temporary file
#
#     if not isinstance(file, str):
#         raise TypeError("file must be a string representing the file path")
#     if not isinstance(channel, int) or channel < 0:
#         raise ValueError("channel must be a non-negative integer")
#     if not isinstance(properties_list, list):
#         raise TypeError("properties_list must be a list of strings") 