import os
import numpy as np
from skimage import io, filters, morphology
from scipy import ndimage
from turmoric.utils import recursively_get_all_filepaths
import matplotlib.pyplot as plt
from typing import Callable


def apply_all_thresh(input_folder: str, output_folder: str, channel: int=1, figsize: tuple=(10, 8)) -> None:
    """
    Apply multiple thresholding algorithms to .tif images and save comparison plots.

    This function loads all `.tif` images found recursively in the input folder.
    For each image, it extracts the specified channel (if multi-channel), applies a suite
    of thresholding methods using `skimage.filters.try_all_threshold`, and saves the resulting
    comparison figure to the output folder.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing input `.tif` images.
    output_folder : str
        Path to the folder where output thresholding comparison plots will be saved.
    channel : int, optional
        Index of the image channel to process if the image is multi-channel (default is 1).
    figsize : tuple of int, optional
        Size of the matplotlib figure for the thresholding comparison plot (default is (10, 8)).

    Returns
    -------
    None
        This function does not return anything. It saves output images to the specified folder.

    Notes
    -----
    - Requires `recursively_get_all_filepaths` function to retrieve file paths recursively.
    - Uses `skimage.io.imread` to read images and `skimage.filters.try_all_threshold` to apply
      thresholding methods.
    - Output images are saved with filenames ending in `_all_thresh.tif`.

    Examples
    --------
    >>> from my_thresholding_module import apply_all_thresh
    >>> input_dir = "data/microscopy_images"
    >>> output_dir = "results/thresholding_plots"
    >>> apply_all_thresh(input_dir, output_dir, channel=0, figsize=(12, 10))

    This will process all `.tif` images in 'data/microscopy_images', apply thresholding to channel 0,
    and save the comparison plots in 'results/thresholding_plots'.
    """
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


def apply_li_threshold(file: str, channel: int=1) -> np.ndarray[bool]:
    """
    Apply Li thresholding to an .tif image and returns a binary mask.

    This function reads an image from the specified file path, selects the specified
    channel, and applies Li's thresholding method to generate a binary image. Pixels
    with intensity values greater than the threshold are set to True.

    Parameters
    ----------
    file : str
        Path to the image file to be processed.
    channel : int, optional
        Index of the color channel to use if the image is RGB or multi-channel.
        Default is 1 (typically the green channel in RGB images).

    Returns
    -------
    binary_li : ndarray of bool
        A 2D binary image where True indicates pixels above the Li threshold.

    Notes
    -----
    Li thresholding is an iterative method that minimizes the cross-entropy
    between the foreground and background pixel distributions.

    References
    ----------
    Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding.
    Pattern Recognition, 26(4), pp.617-625.

    Examples
    --------
    >>> from skimage import io
    >>> import matplotlib.pyplot as plt
    >>> binary_mask = apply_li_threshold("microglia_image.tif", channel=1)
    >>> plt.imshow(binary_mask, cmap='gray')
    >>> plt.title("Li Thresholded Image")
    >>> plt.axis('off')
    >>> plt.show()

    """

    # Check if the file exists
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    # Check if the file is a .tif image
    if not file.lower().endswith('.tif'):
        raise ValueError(f"File {file} is not a .tif image.")
    # Check if the channel is valid
    if not isinstance(channel, int) or channel < 0:
        raise ValueError("Channel must be a non-negative integer.")
    # Check if the file is a valid image
    try:
        im = io.imread(file)
    except Exception as e:
        raise ValueError(f"Could not read image {file}: {e}")
    # Check if the image is multi-channel
    if im.ndim not in [2, 3]:
        raise ValueError(f"Image {file} is not a valid 2D or 3D image.")
    # Check if the channel is valid for the image
    if im.ndim == 3 and (channel < 0 or channel >= im.shape[2]):
        raise ValueError(f"Channel {channel} is out of bounds for image {file}.")
    # Check if the image is grayscale
    if im.ndim == 2 and channel != 0:
        raise ValueError(f"Image {file} is grayscale, channel must be 0.")
    # Check if the image is RGB
    if im.ndim == 3 and im.shape[2] != 3:
        raise ValueError(f"Image {file} is not RGB, channel must be 0, 1 or 2.")
    # Check if the image is RGBA
    if im.ndim == 3 and im.shape[2] != 4:
        raise ValueError(f"Image {file} is not RGBA, channel must be 0, 1, 2 or 3.")   
    # Read the image
    im = io.imread(file)

    microglia_im = im[:, :, channel] if im.ndim == 3 else im

    # Apply Li threshold
    thresh_li = filters.threshold_li(microglia_im)
    binary_li = microglia_im > thresh_li

    return binary_li


def apply_threshold_recursively(input_folder: str,
                                output_folder: str='./thresh_output/',
                                threshold_funcion: Callable[[str], np.ndarray]=apply_li_threshold) -> None:
     
    """
    Recursively applies a thresholding function to all `.tif` images in a directory
    and saves the resulting binary images as `.npy` files.

    Parameters
    ----------
    input_folder : str
        Path to the input directory containing `.tif` image files.
    output_folder : str, optional
        Path to the output directory where thresholded `.npy` files will be saved.
        Defaults to './thresh_output/'.
    threshold_funcion : callable, optional
        A function that takes a file path as input and returns a binary NumPy array.
        Defaults to `apply_li_threshold`.

    Returns
    -------
    None
            This function does not return a value. It saves the binary images as `.npy` files
            in the specified output directory.

    
    Raises
    ------
    Prints error messages if:
        - The input folder does not exist.
        - Any file fails to process due to an exception.

    Notes
    -----
    - The function assumes that `recursively_get_all_filepaths` is defined elsewhere and
      returns a list of `.tif` file paths.
    - The thresholding function should return a binary NumPy array.

    Examples
    --------
    >>> def dummy_threshold(file_path):
    ...     import numpy as np
    ...     return np.ones((100, 100), dtype=bool)  # Dummy binary image
    ...
    >>> apply_threshold_recursively('path/to/tif_images',
    ...                              output_folder='path/to/output',
    ...                              threshold_funcion=dummy_threshold)

    """
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