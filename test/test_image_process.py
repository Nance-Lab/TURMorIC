import pytest
import random
import os
from skimage import io
from skimage.util import img_as_ubyte
import tempfile
from typing import Callable
import numpy as np
import turmoric
from turmoric.image_process import load_npy_file
from turmoric.image_process import normalize_npy_data
from turmoric.image_process import nd2_to_tif

def test_load_npy_file():
    # Create a temporary directory with a dummy .npy file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample numpy array and save it
        sample_data = np.random.rand(100, 100)
        npy_path = os.path.join(temp_dir, "sample.npy")
        np.save(npy_path, sample_data)

        # Load the .npy file using the function
        loaded_data = load_npy_file(npy_path)

        # Assertions
        assert isinstance(loaded_data, np.ndarray), "Loaded data should be a numpy array"
        assert loaded_data.shape == sample_data.shape, "Loaded data shape should match saved data shape"
        assert np.array_equal(loaded_data, sample_data), "Loaded data should match saved data"

def test_normalize_npy_data():
    # Create a sample numpy array with float data
    float_data = np.random.rand(100, 100) * 1000  # Values in [0, 1000]
    normalized_float = normalize_npy_data(float_data)

    # Assertions for float data normalization
    assert normalized_float.dtype == np.uint8, "Normalized data should be of type uint8"
    assert normalized_float.min() >= 0 and normalized_float.max() <= 255, "Normalized data should be in [0, 255]"

    # Create a boolean mask
    bool_mask = np.array([[True, False], [False, True]])
    normalized_mask = normalize_npy_data(bool_mask)

    # Assertions for boolean mask normalization
    assert normalized_mask.dtype == np.uint8, "Normalized mask should be of type uint8"
    assert np.array_equal(normalized_mask, np.array([[255, 0], [0, 255]])), "Boolean mask should be converted to uint8"

def test_nd2_to_tif():
    # Create a temporary directory with a dummy .nd2 file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample numpy array and save it as .npy (simulating .nd2)
        sample_data = np.random.rand(100, 100)
        nd2_path = os.path.join(temp_dir, "sample.nd2")
        np.save(nd2_path, sample_data)  # Simulating .nd2 with .npy for testing

        # Convert the .nd2 file to .tif
        nd2_to_tif(temp_dir, "sample.nd2")

        # Check if the .tif file was created
        tif_path = os.path.join(temp_dir, "sample.tif")
        assert os.path.exists(tif_path), "The TIFF file should be created"
        # Load the TIFF file to verify its content
        tif_data = io.imread(tif_path)
        assert isinstance(tif_data, np.ndarray), "Loaded TIFF data should be a numpy array"
        assert tif_data.shape == sample_data.shape, "TIFF data shape should match original data shape"
        assert np.all(tif_data >= 0) and np.all(tif_data <= 255), "TIFF data should be in [0, 255] range"
        # Clean up the temporary .nd2 file
        os.remove(nd2_path)
        # Check if the .tif file was created
        assert os.path.exists(tif_path), "The TIFF file should be created from the .nd2 file"
        # Load the TIFF file to verify its content
        tif_data = io.imread(tif_path)
        assert isinstance(tif_data, np.ndarray), "Loaded TIFF data should be a numpy array"
        assert tif_data.shape == sample_data.shape, "TIFF data shape should match original data shape"
        assert np.all(tif_data >= 0) and np.all(tif_data <= 255), "TIFF data should be in [0, 255] range"
        # Clean up the temporary .nd2 file
        os.remove(nd2_path)
        # Check if the .tif file was created
        assert os.path.exists(tif_path), "The TIFF file should be created from the .nd2 file"
        # Load the TIFF file to verify its content
        tif_data = io.imread(tif_path)
        assert isinstance(tif_data, np.ndarray), "Loaded TIFF data should be a numpy array"
        assert tif_data.shape == sample_data.shape, "TIFF data shape should match original data shape"
        assert np.all(tif_data >= 0) and np.all(tif_data <= 255), "TIFF data should be in [0, 255] range"
        # Clean up the temporary .nd2 file
        os.remove(nd2_path)