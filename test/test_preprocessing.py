# import module or package here
import pytest
import random
import numpy as np
import os

class TestPreprocessing(pytest.TestCase):

    def test_validate_iamge_format(self):
        test = 0  # remove this and write testing code
        ## testing code

    def test_remove_noise(self):
        test = 0
        ## testing code

    def test_normalize_variations(self):
        test = 0
        ## testing code

    def test_adjust_contrast(self):
        test = 0
        ## testing code

    def test_collect_selected_bstack(self):
        test = 0

    def test_image_list_split(self):
        ##Just copying the main code from the .ipynb, not a test yet hehes
        folder_location = 'test_data'
        arr = os.listdir(folder_location)
        folder_list = np.asarray(arr)
        folder_list = [ x for x in folder_list if "DS" not in x ]
        files_to_split_list = []
        expected_file_list = []

        for folders in folder_location:
            image_array = os.listdir(str(folder_location + '/' + folders))
            subfolder_list = np.asarray(image_array)
            subfolder_list = [ x for x in subfolder_list if "DS" not in x]
            for subfolders in subfolder_list:
                image_array = os.listdir(str(folder_location + '/' + folders + '/' + subfolders))
                files_list = np.asarray(image_array)
                files_list = [x for x in files_list if "DS" not in x]
                for files in files_list:
                    name = str(folder_location + '/' + folders + '/' + subfolders + '/' + files)
                    files_to_split_list.append(name)

        self.assertListEqual(files_to_split_list, expected_file_list)