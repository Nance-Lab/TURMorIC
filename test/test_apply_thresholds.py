import random
import pytest
import os
import shutil
from skimage import io
from skimage.util import img_as_ubyte
from skimage.io import imsave
import shutil
import tempfile
from typing import Callable
import numpy as np
import turmoric
from turmoric.apply_thresholds import apply_all_thresh
from turmoric.apply_thresholds import apply_li_threshold
from turmoric.apply_thresholds import apply_threshold_recursively


def test_apply_all_thresh(input_folder, output_folder, channel, figsize):
    # Create temporary input and output directories
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create a dummy RGB image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_path = os.path.join(input_dir, "test_image.tif")
    imsave(image_path, image)

    # Mock the file discovery function
    from turmoric import recursively_get_all_filepaths
    def mock_get_all_filepaths(folder, ext):
            return [image_path]
    turmoric.recursively_get_all_filepaths = mock_get_all_filepaths

        # Run the function
    apply_all_thresh(input_dir, output_dir, channel=1)

        # Check that output file exists
    output_files = os.listdir(output_dir)
    assert any(f.endswith("_all_thresh.tif") for f in output_files)

    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string")
    if not isinstance(output_folder, str):
        raise TypeError("output_folder must be a string")
    if not isinstance(channel, int) or channel < 0:
        raise ValueError("channel must be a non-negative integer")
    if not (isinstance(figsize, tuple) and len(figsize) == 2 and all(isinstance(x, (int, float)) for x in figsize)):
        raise TypeError("figsize must be a tuple of two numbers")

def test_apply_li_threshold(file, channel, binary_li):
    # Create a synthetic image with two intensity regions
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50:] = 200 # Bottom half is brighter

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        io.imsave(temp_file.name, img_as_ubyte(image))
        temp_path = temp_file.name

    try:
        # Apply the thresholding function
        binary = apply_li_threshold(temp_path)

        # Assertions
        assert binary.dtype == bool, "Output should be a boolean array"
        assert binary.shape == image.shape, "Output shape should match input"
        assert binary[:50].sum() == 0, "Top half should be below threshold"
        assert binary[50:].sum() > 0, "Bottom half should be above threshold"

    finally:
        # Clean up
        os.remove(temp_path)
    if not isinstance(file, str):
        raise TypeError("file must be a string")
    if not isinstance(channel, int) or channel < 0:
        raise ValueError("channel must be a non-negative integer")
    if not isinstance(binary_li, np.ndarray):
        raise TypeError("binary_li must be a numpy array")
    if binary_li.ndim != 2:
        assert isinstance(binary_li, np.ndarray) and binary_li.dtype == bool and binary_li.ndim == 2


# Dummy threshold function
def dummy_threshold(file_path):
    return np.ones((10, 10), dtype=bool)


@pytest.fixture
def temp_dirs():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    # Create dummy .tif files
    test_files = ['image1.tif', 'subdir/image2.tif']
    for file in test_files:
        full_path = os.path.join(input_dir, file)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(b'dummy data')

    yield input_dir, output_dir, test_files

    # Cleanup
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)


def test_apply_threshold(temp_dirs):
    input_dir, output_dir, test_files = temp_dirs

    apply_threshold_recursively(
        input_folder=input_dir,
        output_folder=output_dir,
        threshold_function=dummy_threshold
    )

    for file in test_files:
        expected_path = os.path.join(
            output_dir,
            os.path.splitext(file)[0] + '.npy'
        )
        assert os.path.exists(expected_path), f"{expected_path} not found"

        arr = np.load(expected_path)
        assert np.array_equal(arr, np.ones((10, 10), dtype=bool))
