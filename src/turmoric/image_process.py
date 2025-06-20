import os
import skimage
import numpy as np
from nd2 import ND2File
import tifffile

def load_npy_file(path: str, file_name: str) -> np.ndarray:
    """
    Load a NumPy `.npy` file and apply connected component labeling.

    This function loads a binary or labeled image stored in a `.npy` file,
    applies connected component labeling using `skimage.measure.label`, and
    returns the labeled image data.

    Parameters
    ----------
    path : str
        Directory containing the `.npy` file.
    file_name : str
        Name of the `.npy` file to load.

    Returns
    -------
    image_data : ndarray
        A 2D NumPy array where connected components in the binary mask are labeled with
        unique integer values. The background is labeled as 0.

    Raises
    ------
    Exception
        If the file cannot be loaded or processed, an error message is printed
        and the exception is raised.

    Notes
    -----
    - If an error occurs while loading or processing the file, an error message is printed.
    - The function assumes the `.npy` file contains a 2D binary mask (e.g., values of 0 and 1).
    - Uses `skimage.measure.label` for connected component labeling.
Examples
    --------
    >>> import numpy as np
    >>> from skimage import measure
    >>> import os
    >>> image = np.zeros((5, 5), dtype=int)
    >>> image[1:3, 1:3] = 1
    >>> image[3:5, 3:5] = 1
    >>> np.save('example.npy', image)
    >>> labeled = load_npy_file('.', 'example.npy')
    >>> print(labeled)
    [[0 0 0 0 0]
     [0 1 1 0 0]
     [0 1 1 0 0]
     [0 0 0 2 2]
     [0 0 0 2 2]]
    """
    npy_path = os.path.join(path, file_name)

    try:
        image_data = skimage.measure.label(np.load(npy_path))
    except Exception as e:
        print(f"error processing '{npy_path}': {e}")

    return image_data

def normalize_npy_data(npy_image_data: np.ndarray) -> np.ndarray:
    """
    Normalize NumPy image data array to uint8 format.

    This function converts a NumPy array representing image data into a 
    normalized 8-bit unsigned integer format (`uint8`), suitable for image 
    processing and visualization. It handles boolean arrays by mapping 
    `True` to 255 and `False` to 0. For non-uint8 numeric arrays, it scales 
    the values to the [0, 255] range.If the input is not already `uint8`, it
    is linearly scaled to the range [0, 255].

    Parameters
    ----------
    npy_image_data : np.ndarray
        A NumPy array containing image data. Can be of dtype `bool`, 
        `float`, `int`, etc. The array must be 2D or compatible with
        image-like data.

    Returns
    -------
    normalized_image_data : np.ndarray
        A NumPy array of dtype `uint8` with values scaled to the range [0, 255].

    Notes
    -----
    - Boolean arrays are multiplied by 255 to convert to `uint8`.
    - For non-boolean arrays, the data is scaled as:
        `(array - array.min()) / (array.max() - array.min()) * 255`
    - A small epsilon (`1e-8`) is added to the denominator to prevent division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import normalize_npy_data

    # Example with float data
    >>> float_image = np.random.rand(100, 100)  # Values in [0, 1]
    >>> normalized = normalize_npy_data(float_image)
    >>> normalized.dtype
    dtype('uint8')
    >>> normalized.min(), normalized.max()
    (0, 255)

    # Example with boolean mask
    >>> bool_mask = np.array([[True, False], [False, True]])
    >>> normalized_mask = normalize_npy_data(bool_mask)
    >>> normalized_mask
    array([[255,   0],
           [  0, 255]], dtype=uint8)
    """
    if npy_image_data.dtype == np.bool_:
        # Convert boolean to uint8 (True -> 255, False -> 0)
        normalized_image_data = (npy_image_data * 255).astype(np.uint8)
    else:
        # Normalize and scale image data to uint8 if it's not already
        if npy_image_data.dtype != np.uint8:
            normalized_image_data = (
                255 * (npy_image_data - npy_image_data.min()) / (npy_image_data.ptp() + 1e-8)
            ).astype(np.uint8)

    return normalized_image_data

def save_npy_as_tif():
    # Construct the output TIFF file path
        #relative_path = os.path.relpath(dirpath, root_directory)
        # tiff_dir = os.path.join(output_dir, relative_path)
        # os.makedirs(tiff_dir, exist_ok=True)
        # tiff_path = os.path.join(tiff_dir, file.replace(".npy", ".tif"))

        # # Save as TIFF
        # #Image.fromarray(image_data).save(tiff_path)
        # print(f"Converted: {npy_path} -> {tiff_path}")
        # io.imsave(tiff_path, image_data)
    pass
    
def nd2_to_tif(path: str, file_name: str) -> None:
    """
    Convert an `.nd2` microscopy image file to a `.tif` file.

    This function reads a Nikon ND2 image file from the specified path and converts it
    to a TIFF file using `tifffile.imwrite`. The output `.tif` file is saved in the
    same directory with the same base name.

    Parameters
    ----------
    path : str
        Directory containing the `.nd2` file.
    file_name : str
        Name of the `.nd2` file to convert.

    Returns
    -------
    None
        The function saves the converted `.tif` file to disk and does not return a value.

    Notes
    -----
    - Requires `ND2File` from the `nd2` or `nd2reader` library to read ND2 files.
    - The converted `.tif` file is saved to the same directory as the input file,
      with the `.nd2` extension replaced by `.tif`.
    - Ensure that the ND2 file contains a data format compatible with TIFF output.

    Examples
    --------
    >>> from my_module import nd2_to_tif
    >>> nd2_to_tif("/data/images", "sample_image.nd2")
    # This will create a file named 'sample_image.tif' in the '/data/images' directory.
    """
    nd2_path = os.path.join(path, file_name)
    tif_path = os.path.join(path, file_name.replace(".nd2", ".tif"))
    
    with ND2File(nd2_path) as nd2_file:
        nd2_data = nd2_file.asarray()

        tifffile.imwrite(tif_path, nd2_data)

    
 
def load_tif_file():
    pass