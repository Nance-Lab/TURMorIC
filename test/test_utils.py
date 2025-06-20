import pytest
import random
import os
from skimage import io
from skimage.util import img_as_ubyte
import tempfile
from typing import Callable
import numpy as np
import turmoric
from turmoric.utils import organize_files_without_leakage
from turmoric.utils import recursively_get_all_filepaths


def test_organize_files_without_leakage(input_folder, output_folder):
    # Create a temporary directory with dummy files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        for i in range(5):
            file_path = os.path.join(temp_dir, f"file_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"This is file {i}")

        # Run the function
        organize_files_without_leakage(temp_dir, output_folder)

        # Check if files are moved to output folder
        output_files = os.listdir(output_folder)
        assert len(output_files) == 5, "All files should be moved to the output folder"
        for i in range(5):
            assert f"file_{i}.txt" in output_files, f"file_{i}.txt should be in the output folder"

    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string")
    if not isinstance(output_folder, str):
        raise TypeError("output_folder must be a string")


def test_recursively_get_all_filepaths(input_folder, ext):
    # Create a temporary directory with dummy files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        for i in range(5):
            file_path = os.path.join(temp_dir, f"file_{i}.{ext}")
            with open(file_path, 'w') as f:
                f.write(f"This is file {i}")

        # Run the function
        filepaths = recursively_get_all_filepaths(temp_dir, ext)

        # Check if all files are returned
        assert len(filepaths) == 5, "All files should be found"
        for i in range(5):
            assert os.path.basename(filepaths[i]) == f"file_{i}.{ext}", f"file_{i}.{ext} should be in the list"

    if not isinstance(input_folder, str):
        raise TypeError("input_folder must be a string")
    if not isinstance(ext, str):
        raise TypeError("ext must be a string")


def test_recursively_get_all_filepaths_invalid_input():
    with pytest.raises(TypeError):
        recursively_get_all_filepaths(123, "txt")  # input_folder should be a string
    with pytest.raises(TypeError):
        recursively_get_all_filepaths("/path/to/folder", 123)  # ext should be a string
    with pytest.raises(FileNotFoundError):
        recursively_get_all_filepaths("/non/existent/path", "txt")  # Non-existent path
#     """
#     Recursively retrieve all file paths of a specific type from a directory.