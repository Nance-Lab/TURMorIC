# this is a python script that opens an image from a file, filters it depending on their extension, splits the image into two channels and then separate those new images on separated files depending on their chanel and type.
# The script also includes unit tests to validate the image processing functions.

import shutil, os
import numpy as np
import pandas as pd
from skimage import io #loads image as a numpy array
import matplotlib.pyplot as plt
from PIL import Image #pillow library used for loading images or saving
import unittest
import tempfile

#######################################
stain1 = 'dapi'
stain2 = 'iba'
#DAPI staining is helpful for identifying the overall number and distribution of cells within brain tissue. It provides a general nuclear marker for both neuronal and non-neuronal cells (like astrocytes, oligodendrocytes, and microglia).
#iba: IBA1 staining is essential for studying microglia morphology, activation, and distribution. Microglia play a significant role in neuroinflammation, brain development, and injury responses.

file_type_new = '.png'

# Set folder location
folder_location = r"C:\Users\sergi\Desktop\analysis_images"

#######################################
#Function 1: Filters images by their format(folder_cleaner)

def folder_cleaner(folder, image_type):
    if not folder:  # Check if the folder is empty
        return "The folder is empty"
    k = 0
    folder_array = np.array(folder)  
    for file in folder_array:
        if file.startswith('.'):  # Skip hidden files (those starting with '.')
            continue
        if image_type in str(file):  # Check if the file contains the image_type (e.g., '.jpg', '.png', etc.)
            k += 1
        else:
            # Remove files that do not match the specified image type
            folder_array = np.delete(folder_array, np.argwhere(folder_array == str(file)))
    return folder_array  # Return the filtered folder array
# List files in the folder
files_in_folder = os.listdir(folder_location)

# Call the folder_cleaner function 
cleaned_files = folder_cleaner(files_in_folder, file_type_new)

# Display the list of image names
print("Filtered images:")
for file in cleaned_files:
    print(file)  # Print only the image names

#####################################
# Function 2:  Transforms a list of filenames into a NumPy array
def list_to_array(file_list):
    return np.asarray(file_list)  # Convert the list to a NumPy array
file_array = list_to_array(cleaned_files) # Get all filtered files as an array
print("\nFiltered image files as an array:") # Display the filtered array
print(file_array)

#####################################
# Function 3: Extract and save the two channels of images. 
def extract_channels(file_list, folder_location):
    for file_name in file_list:
        file_path = os.path.join(folder_location, file_name)
        
        try:
            # Read the image
            im = io.imread(file_path)
            
            # Ensure the image has at least two channels
            if im.ndim < 3 or im.shape[0] < 2:
                raise ValueError(f"Image {file_name} does not have at least 2 channels.")
            
            # Extract first and second channels
            channel1 = im[0, :, :]
            channel2 = im[1, :, :]

            if np.all(channel1 == 0) and np.all(channel2 == 0): #checks if images are completely black as precaution
                raise ValueError(f"Image {file_name} is completely black.")

            # Convert channels to images
            channel1_img = Image.fromarray(np.uint16(channel1))
            channel2_img = Image.fromarray(np.uint16(channel2))

            # Remove file extension (e.g., image1.png → image1)
            base_name = file_name.replace(file_type_new, "")

            # Save extracted channels with modified names
            channel1_img.save(os.path.join(folder_location, f"{base_name}_{stain1}{file_type_new}"))
            channel2_img.save(os.path.join(folder_location, f"{base_name}_{stain2}{file_type_new}"))

            print(f"Extracted channels saved for {file_name}")

        except ValueError as e:
            print(f"Warning: {e}")
            raise
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Call function
extract_channels(file_array, folder_location)

#######################################
#Function 4: Creates channel folders and relocate extracted channels for each filtered image.
def move_stain_images(folder_location, stain1, stain2):  
    # Define destination folders
    stain1_folder = os.path.join(folder_location, stain1)
    stain2_folder = os.path.join(folder_location, stain2)

    # Ensure destination folders exist
    try:
        os.makedirs(stain1_folder, exist_ok=True)
        os.makedirs(stain2_folder, exist_ok=True)
        print(f"Folders created/verified: {stain1_folder}, {stain2_folder}")
    except OSError as e:
        print(f"Error: Could not create destination folders - {e}")
        return
    file_list = os.listdir(folder_location)

    # Check if there are files to move
    if not file_list:
        print("No files found to move.")
        return

    # Iterate through the file list and move files
    for tiled_image in file_list:
        source_path = os.path.join(folder_location, tiled_image)  
        
        # Check if the file exists in the original location
        if not os.path.exists(source_path):
            print(f"Error: File not found - {tiled_image}")
            continue  # Skip to the next file

        # Determine destination based on channel type.
        if tiled_image.lower().endswith("_dapi.png"):
            destination = os.path.join(stain1_folder, tiled_image)
        elif tiled_image.lower().endswith("_iba.png"):
            destination = os.path.join(stain2_folder, tiled_image)
        else:
            print(f"Skipped: {tiled_image} (Condition not met)")
            continue  # Skip files that do not match conditions
        
        # Move the file
        try:
            shutil.move(source_path, destination)
            print(f"Moved: {tiled_image} → {destination}")
        except Exception as e:
            print(f"Error moving {tiled_image}: {e}")

# Call the function 
move_stain_images(folder_location,stain1,stain2)

####################################################
#UNIT TESTS
####################################################

#Function 1

class TestFolderCleaner(unittest.TestCase):

    def test_format_types(self):
        """Ensures folder cleaner function is abble to handle different files type"""
        test_folder = [
            "image1.jpg",  # Image file (jpg)
            "image2.png",  # Image file (png)
            "image3.tif",  # Image file (tif)
            "document.txt",  # Non-image file
            "image4.pdf"]  # Non-image file
        image_formats = ['.jpg', '.png', '.tif', '.jpeg'] # Define multiple image formats to test
        
        for image_type in image_formats:
            with self.subTest(image_type=image_type):
                cleaned_files = folder_cleaner(test_folder, image_type)
                self.assertTrue(all(file.endswith(image_type) for file in cleaned_files), f"Some files do not match {image_type}: {cleaned_files}")
                
    def test_empty_folder(self):
        """Ensures folder cleaner function is able to handle empty folders"""
        test_folder2 = []  # Simulates empty folder
        test_image_type = '.png'
        result = folder_cleaner(test_folder2, test_image_type)
        self.assertEqual(result, "The folder is empty", "ok")  # Expected the "folder is empty" message
    
    def test_hidden_files(self):
        """Ensures folder cleaner function is not processing hidden files (.)"""
        test_folder3 = [".hiddenfile.png", "visible.png", ".anotherhidden.png"]  # Simulated list including hidden files
        image_formats = ['.png']
    
        for image_type in image_formats:
            cleaned_files = folder_cleaner(test_folder3, image_type)
            # Ensure that no hidden files (files starting with '.') are included
            self.assertFalse(any(file.startswith('.') for file in cleaned_files),
                         f"Hidden files were incorrectly included: {cleaned_files}")
            print(f"Filtered images for {image_type}: {cleaned_files}")

    def test_special_characters(self):
        """Ensures folder cleaner function handles files with special caracters names"""
        test_folder4 = ["phRo$$@home.jpg", "#!weird#name.png", "normA00)_image.jpeg"]
        image_formats = ['.jpg', '.png', '.jpeg']
        
        for image_type in image_formats:
            with self.subTest(image_type=image_type):
                cleaned_files = folder_cleaner(test_folder4, image_type)
                self.assertTrue(all(file.endswith(image_type) for file in cleaned_files), 
                                f"Some files do not match {image_type}: {cleaned_files}")
                print(f"Filtered images for {image_type}: {cleaned_files}")
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#Function 2

class TestListToArray(unittest.TestCase):

    def test_non_empty_list(self):
        """Test conversion of a non-empty list to a NumPy array"""
        file_list = ["file1.jpg", "file2.png", "file3.jpeg"]
        result = list_to_array(file_list)
        self.assertIsInstance(result, np.ndarray, "The result should be a NumPy array")
        self.assertEqual(result.tolist(), file_list, "The converted array does not match the input list")

    def test_empty_list(self):
        """Test conversion of an empty list to a NumPy array"""
        file_list = []
        result = list_to_array(file_list)
        self.assertIsInstance(result, np.ndarray, "The result should be a NumPy array")
        self.assertEqual(result.tolist(), file_list, "The result array for an empty list should be empty")

    def test_mixed_data_types(self):
        """Test conversion of a list with mixed data types (strings and integers)"""
        file_list = ["file1.jpg", 123, "file2.png", 456]
        result = list_to_array(file_list)
        self.assertIsInstance(result, np.ndarray, "The result should be a NumPy array")
        # Check if the result is converted to string due to mixed data types
        self.assertEqual(result.tolist(), ['file1.jpg', '123', 'file2.png', '456'], "The converted array does not match the expected list with string conversion")

    def test_array_conversion(self):
        """Test conversion of a list containing NumPy arrays to a NumPy array"""
        file_list = [np.array([1, 2]), np.array([3, 4])]
        result = list_to_array(file_list)
        self.assertIsInstance(result, np.ndarray, "The result should be a NumPy array")
        self.assertEqual(result.tolist(), [item.tolist() for item in file_list], "The arrays inside the list should be converted correctly")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#Function 3

class TestExtractChannels(unittest.TestCase):
     
    def test_missing_channels(self):
        """Ensures images without 2 channels can be processed or fails gracefully"""
        with tempfile.TemporaryDirectory() as temp_folder:
            # Simulate an image with only 1 channel
            file_name = "image_single_channel.tif"
            image = np.zeros((1, 100, 100), dtype=np.uint16)
            io.imsave(os.path.join(temp_folder, file_name), image)

            # extracting channels
            with self.assertRaises(ValueError):
                extract_channels([file_name], temp_folder)

    def test_completely_black_image(self):
        """Ensures completely black images are handled appropriately"""
        with tempfile.TemporaryDirectory() as temp_folder:
            # Simulate a completely black image with 2 channels
            file_name = "image_black.tif"
            black_image = np.zeros((2, 100, 100), dtype=np.uint16)
            io.imsave(os.path.join(temp_folder, file_name), black_image)

            # extracting channels
            with self.assertRaises(ValueError):
                extract_channels([file_name], temp_folder)

    def test_large_image_size(self):
        """Ensure that images that are too large are handled appropriately."""
        with tempfile.TemporaryDirectory() as temp_folder:
            file_name = "large_image.tif" 
            large_image = np.random.randint(1, 256, size=(2, 10000, 10000), dtype=np.uint16) # Create a very large image 
            io.imsave(os.path.join(temp_folder, file_name), large_image)
            try:
                extract_channels([file_name], temp_folder)
            except MemoryError:
              print(f"MemoryError: The image {file_name} is too large to process.")
        

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#Function 4

class TestMoveStainImages(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_folder = tempfile.mkdtemp() #creates a temporary folder to simulate as test folder
        test_files = ["image1.jpg", "image2.png", "image3.jpg", "document.txt", ".hiddenfile.png", "image4.jpeg"]
        for file_name in test_files:
            with open(os.path.join(self.test_folder, file_name), 'w') as f:
                f.write(f"This is a test file: {file_name}")
        
        self.stain1 = "dapi"
        self.stain2 = "iba"

    def test_source_folder_exists(self):
        """Ensure the source folder exists before processing."""
        self.assertTrue(os.path.exists(self.test_folder)) #verifies that folder exists

    def test_empty_source_folder(self):
        """Ensure the function handles an empty source folder or fails gracefully."""
        with tempfile.TemporaryDirectory() as empty_folder: # Create a temporary empty folder using tempfile
            self.assertEqual(len(os.listdir(empty_folder)), 0)  # Ensure the folder is empty
            # Now, test the function with the empty folder
            move_stain_images(empty_folder, self.stain1, self.stain2)

    def test_only_allowed_files(self):
        """Ensures that only files with the right extension are moved to their respective directory"""
        with tempfile.TemporaryDirectory() as temp_folder: #create a temporary folder
            # Create destination subfolders for stains
            stain1_folder = os.path.join(temp_folder, "dapi")
            stain2_folder = os.path.join(temp_folder, "iba")
            os.makedirs(stain1_folder, exist_ok=True)
            os.makedirs(stain2_folder, exist_ok=True)

            # Create files with different extensions
            file1 = os.path.join(temp_folder, "sample1_dapi.png")
            file2 = os.path.join(temp_folder, "sample2_iba.png")
            file3 = os.path.join(temp_folder, "sample3.txt")
            
            # Write content to the files
            for file in [file1, file2, file3]:
                with open(file, 'w') as f:
                    f.write("content")

            # Call the move_stain_images function
            move_stain_images(temp_folder, "dapi", "iba")
            
            # Only "sample1_dapi.png" and "sample2_iba.png" should have been moved
            self.assertTrue(os.path.exists(os.path.join(stain1_folder, "sample1_dapi.png")))
            self.assertTrue(os.path.exists(os.path.join(stain2_folder, "sample2_iba.png")))
            self.assertFalse(os.path.exists(os.path.join(stain1_folder, "sample3.txt")))
            self.assertFalse(os.path.exists(os.path.join(stain2_folder, "sample3.txt")))
    
    def test_duplicate_file_handling(self):
        """Ensure duplicate files are detected and not overwritten for both stains."""
    
        for stain in [self.stain1, self.stain2]:  # ["dapi", "iba"]
            stain_folder = os.path.join(self.test_folder, stain)
            os.makedirs(stain_folder, exist_ok=True)  # Create stain folder if needed

            # Create a duplicate file in the destination folder
            duplicate_file = os.path.join(stain_folder, f"sample1_{stain}.png")
            with open(duplicate_file, 'w') as f:
                f.write(f"duplicate test for {stain}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
