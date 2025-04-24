import os
import numpy as np
from skimage import io, filters, morphology
from scipy import ndimage

def apply_li_threshold(file, size):
    """
    Applies Li thresholding to all .tif images in the input folder (and subfolders)
    and saves the binary masks in the output folder.

    Parameters:
    - input_folder: Path to the folder containing .tif images.
    - output_folder: Path to save the processed binary masks.
    - min_object_size: Minimum size of objects to retain in the binary mask.
    """
   
    # Read the image
    im = io.imread(file)

    # Assume the second channel is the microglia channel
    microglia_im = im[:, :, 1] if im.ndim == 3 else im

    # Apply Li threshold
    thresh_li = filters.threshold_li(microglia_im)
    binary_li = microglia_im > thresh_li

    # Remove small objects and fill holes
    binary_li = morphology.remove_small_objects(binary_li, min_size=size)
    binary_li = ndimage.binary_fill_holes(binary_li)

    return binary_li



def apply_li_threshold_recursively(input_folder, output_folder='./li_output/', threshold_method='li'):
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    threshold_function = threshold_helper_func(threshold_method)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Walk through all files and subfolders
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".tif"):
                # Full input path
                input_path = os.path.join(root, file)

                # Create corresponding output subfolder
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Full output path
                output_path = os.path.join(output_subfolder, file.replace(".tif", "_" + threshold_method + "_thresh.npy"))
                try:
                    binary_image = threshold_function(file)
                    np.save(output_path, binary_image)
                    
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print(f"Processing completed. Results are saved in '{output_folder}'.")

def threshold_helper_func(threshold):
    if threshold == 'li':
        return  apply_li_threshold
    else:
        raise KeyError('Threshold not available. Choose from: li')
