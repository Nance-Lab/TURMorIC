# this is a python script that reads an image, splits it into 4 quadrants, converts it to grayscale, and saves the processed images.
# The script also includes unit tests to validate the image processing functions.
import cv2 # OpenCV library
import numpy as np # Numpy library
import pytest # Unit testing library
import os # Operating system library

def split_image_into_quadrants(image: np.ndarray): # Function to split image into 4 quadrants
    """Splits an image into four quadrants and returns them as a list."""
    if image is None or not isinstance(image, np.ndarray): # Check if image is valid or not 
        raise ValueError("Invalid image input") # Raise an error if image is invalid
    
    height, width = image.shape[:2] # Get the height and width of the image
    mid_x, mid_y = width // 2, height // 2 # Get the mid point of the image
    
    top_left = image[:mid_y, :mid_x] # Get the top left quadrant of the image
    top_right = image[:mid_y, mid_x:] # Get the top right quadrant of the image
    bottom_left = image[mid_y:, :mid_x] # Get the bottom left quadrant of the image
    bottom_right = image[mid_y:, mid_x:] # Get the bottom right quadrant of the image
    
    return [top_left, top_right, bottom_left, bottom_right] # Return the 4 quadrants of the image

def convert_to_grayscale(image: np.ndarray): # Function to convert image to grayscale
    """Converts a color image to grayscale."""
    if image is None or not isinstance(image, np.ndarray): # Check if image is valid or not
        raise ValueError("Invalid image input")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

def process_image(image_path): # Function to process the image
    """Loads an image, splits it into quadrants, converts to grayscale, and saves outputs."""
    if not os.path.exists(image_path): # Check if the image exists or not
        raise FileNotFoundError(f"Image not found: {image_path}") # Raise an error if image is not found
    
    image = cv2.imread(image_path) # Read the image
    if image is None: # Check if the image is valid or not
        raise ValueError("Failed to load image. Check file format.")
    
    quadrants = split_image_into_quadrants(image) # Split the image into quadrants
    grayscale_image = convert_to_grayscale(image) # Convert the image to grayscale
    
    output_dir = os.path.join(os.path.dirname(image_path), "processed_output") # Create a directory to save the processed images on my desktop
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it does not exist, the name of this 
    
    for i, quadrant in enumerate(quadrants): # Save each quadrant as a separate image
        cv2.imwrite(os.path.join(output_dir, f"quadrant_{i+1}.jpg"), quadrant) # Save the quadrant image
    
    cv2.imwrite(os.path.join(output_dir, "grayscale.jpg"), grayscale_image) # Save the grayscale image
     
    return output_dir # Return the output directory
def validate_image_format(image_path):
    """Checks if the image format is valid (e.g., .jpg, .png).""" # Function to validate the image format
    valid_formats = ('.jpg', '.png', '.tiff', '.bmp', '.nd2') # List of valid image formats
    return image_path.lower().endswith(valid_formats) # Check if the image format is valid

def generate_noisy_image(): # Function to generate a noisy image
    """Generates an image with artificial noise for testing.""" # Generate a noisy image
    noisy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) # Generate random pixel values
    return noisy_image # Return the noisy image

def generate_image_with_variations(): # Function to generate an image with intensity variations
    """Generates an image with intensity variations for testing.""" # Generate an image with intensity variations
    image = np.linspace(50, 200, 100 * 100 * 3).reshape(100, 100, 3).astype(np.uint8) # Create an image with intensity variations
    return image

class TestImageProcessing(pytest.TestCase):
    def setUp(self): # Set up the test case
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    
    def test_split_dimensions(self): # Test to check the dimensions of the quadrants
        quadrants = split_image_into_quadrants(self.image) # Split the image into quadrants
        self.assertEqual(len(quadrants), 4) # Check if the number of quadrants is 4
        self.assertEqual(quadrants[0].shape[:2], (50, 50)) # Check the dimensions of the quadrants
        self.assertEqual(quadrants[1].shape[:2], (50, 50)) # Check the dimensions of the quadrants
        self.assertEqual(quadrants[2].shape[:2], (50, 50)) #  Check the dimensions of the quadrants
        self.assertEqual(quadrants[3].shape[:2], (50, 50)) # Check the dimensions of the quadrants
    
    def test_invalid_input(self): # Test to check invalid input
        with self.assertRaises(ValueError): # Check if the function raises a ValueError
            split_image_into_quadrants(None) # Pass None as input
        with self.assertRaises(ValueError):
            split_image_into_quadrants("not an image")
    
    def test_grayscale_conversion(self): # Test to check grayscale conversion
        gray_image = convert_to_grayscale(self.image) #     Convert the image to grayscale
        self.assertEqual(len(gray_image.shape), 2)  # Grayscale images have 2 dimensions
        self.assertEqual(gray_image.shape, (100, 100)) # Check the dimensions of the grayscale image
    
if __name__ == "__main__":
    image_path = "/Users/munawaraxh/cheme_546/project/Planning/example_dataset/example_images/MEF_wildtype/xy013c1.tif"
    try:
        output_directory = process_image(image_path)
        print(f"Processed images saved in: {output_directory}")
    except Exception as e:
        print(f"Error: {e}")
