import os
import numpy as np
from skimage import io, filters, morphology
from scipy import ndimage
import click
from pathlib import Path
from skimage.measure import block_reduce, label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
import tifffile as tiff

def create_microglia_mask(image, threshold_method=filters.threshold_li):

    thresh_li = threshold_method(image)
    binary_li = image > thresh_li

    objects = label(binary_li)
    objects = clear_border(objects)
    large_objects = remove_small_objects(objects, min_size=50000)
    small_objects = label((objects ^ large_objects) > thresh_li)

    binary_li = ndimage.binary_fill_holes(remove_small_objects(small_objects > thresh_li, min_size=500))

    #scaled_img = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')
    #hist = np.histogram(scaled_img.flatten(), range=[0,50], bins=50)
    return binary_li


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, readable=True, path_type=Path))
@click.argument('output_folder', type=click.Path(exists=False, path_type=Path))
@click.option("-s", "--size", type=click.INT, default=71)
def apply_li_threshold(input_folder, output_folder, size=71):
    """
    Applies Li thresholding to all .tif images in the input folder (and subfolders)
    and saves the binary masks in the output folder.

    Parameters:
    - input_folder: Path to the folder containing .tif images.
    - output_folder: Path to save the processed binary masks.
    - size: Minimum size of objects to retain in the binary mask.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

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
                output_path = os.path.join(output_subfolder, file.replace(".tif", "_li_thresh.npy"))

                try:
                    # Read the image
                    #img = io.imread(input_path)

                    # Assume the second channel is the microglia channel
                    #microglia_im = img[:, :, 1] if img.ndim == 3 else img

                    # Apply Li threshold

                    img = tiff.imread(input_path)
                    
                    binary_li = create_microglia_mask(img)

                        # Save the binary mask as .npy
                    np.save(output_path, binary_li)
                        

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print(f"Processing completed. Results are saved in '{output_folder}'.")

# Example usage
if __name__ == "__main__":
    apply_li_threshold()
    