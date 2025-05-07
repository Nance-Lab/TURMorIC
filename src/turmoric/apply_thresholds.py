import os
import numpy as np
from skimage import io, filters, morphology
from scipy import ndimage
from turmoric.utils import recursively_get_all_filepaths
import matplotlib.pyplot as plt


def apply_all_thresh(input_folder, output_folder, channel=1, figsize=(10, 8)):

    os.makedirs(output_folder, exist_ok=True)

    file_list = recursively_get_all_filepaths(input_folder, ".tif")
    for file in file_list:
        im = io.imread(file)
        microglia_im = im[:, :, channel] if im.ndim == 3 else im
        fig, ax = filters.try_all_threshold(microglia_im,
                                            figsize=figsize,
                                            verbose=False)
        output_path = os.path.join(output_folder,
                                   os.path.basename(file).replace(
                                       '.tif', '_all_thresh.tif'))
        fig.savefig(output_path)
        plt.close(fig)
    return


def apply_li_threshold(file, channel=1):
    """
    Applies Li thresholding to all .tif images in the input folder
    (and subfolders) and saves the binary masks in the output folder.

    Parameters:
    - input_folder: Path to the folder containing .tif images.
    - output_folder: Path to save the processed binary masks.
    - min_object_size: Minimum size of objects to retain in the binary mask.
    """

    # Read the image
    im = io.imread(file)

    microglia_im = im[:, :, channel] if im.ndim == 3 else im

    # Apply Li threshold
    thresh_li = filters.threshold_li(microglia_im)
    binary_li = microglia_im > thresh_li

    return binary_li


def apply_threshold_recursively(input_folder,
                                output_folder='./thresh_output/',
                                threshold_funcion=apply_li_threshold):

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    file_list = recursively_get_all_filepaths(input_folder, ".tif")

    for file in file_list:
        try:
            binary_image = threshold_funcion(file)
            output_path = os.path.join(output_folder,
                                       file.replace('.tif', '.npy'))
            np.save(output_path, binary_image)
        except Exception as e:
            print(f"Error processing {file}: {e}")
