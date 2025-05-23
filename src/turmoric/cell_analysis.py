import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from turmoric.utils import recursively_get_all_filepaths

def apply_regionprops(file: str, properties_list: list) -> pd.DataFrame:
    """
    Compute region properties from a binary mask stored in a .npy file.

    This function loads a binary mask from a NumPy `.npy` file, labels connected regions,
    computes specified region properties using `skimage.measure.regionprops_table`, and
    returns the results as a pandas DataFrame.

    Parameters
    ----------
    file : str
        Path to the `.npy` file containing a binary mask (2D NumPy array of 0s and 1s).
    properties_list : list of str
        List of region properties to compute. These should be valid property names
        accepted by `skimage.measure.regionprops_table`, such as 'area', 'centroid',
        'eccentricity', etc.

    Returns
    -------
    props_df : pandas.DataFrame
        A DataFrame containing the computed region properties for each labeled region.
        Includes an additional column `'filename'` with the source file path.

    Notes
    -----
    - The binary mask should be a 2D NumPy array saved with `np.save`.
    - Each row in the resulting DataFrame corresponds to a connected region in the mask.
    - For valid property names, see the documentation for `skimage.measure.regionprops_table`.
    - The binary mask should be a 2D array where foreground regions are marked with 1s.
    - Connected components are labeled using 8-connectivity by default.
    - This function is useful for batch processing of image masks in segmentation tasks.
    
    Examples
    --------
    >>> props_df = apply_regionprops("mask_01.npy", ["area", "centroid"])
    >>> print(props_df.head())
    """

    # Load the binary mask
    binary_mask = np.load(file)

    # Label connected regions in the binary mask
    label_image = label(binary_mask)

    # Measure properties
    props = regionprops_table(label_image, properties=properties_list)

    # Create a DataFrame for the current file
    props_df = pd.DataFrame(props)
    props_df['filename'] = file  # Add filename column

    return props_df


def apply_regionprops_recursively(input_folder: str, properties_list: tuple=(
                                'area', 'bbox_area', 'centroid', 'convex_area',
                                'eccentricity', 'equivalent_diameter',
                                'euler_number', 'extent', 'filled_area',
                                'major_axis_length', 'minor_axis_length',
                                'orientation', 'perimeter', 'solidity')) -> pd.DataFrame:
   """
    Recursively applies region properties extraction to all `.npy` files in a directory.

    This function traverses the given input folder and its subfolders to find all `.npy` files.
    Each file is assumed to contain a labeled image (e.g., from segmentation). The function uses
    `apply_regionprops` to extract specified properties from each labeled region and returns a combined
    pandas DataFrame of all results.
 
    Parameters
    ----------
    input_folder : str
        Path to the root directory containing `.npy` binary mask files.
    properties_list : tuple of str, optional
        A tuple of region properties to compute for each labeled region in the binary masks.
        Default includes a comprehensive set of geometric properties such as 'area',
        'centroid', 'eccentricity', etc. Valid property names must be accepted by
        `skimage.measure.regionprops_table`.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame containing region properties from all processed files.
        Each row corresponds to a labeled region and includes a 'filename' column indicating
        the source file.

    Raises
    ------
    FileNotFoundError
        If the input folder does not exist.
    Exception
        If an error occurs while processing a file, it is caught and printed, but processing continues.

    Notes
    -----
    - Binary masks must be stored as `.npy` files containing 2D NumPy arrays.
    - If a file cannot be processed, an error message is printed and processing continues.
    - Requires `recursively_get_all_filepaths` to collect `.npy` file paths.
    - Uses `apply_regionprops` to compute the region properties and returns a DataFrame.

    Examples
    --------
    >>> df = apply_regionprops_recursively('/path/to/npy_files')
    >>> print(df.head())

    >>> df = apply_regionprops_recursively('/data/images', properties_list=('area', 'centroid'))
    >>> df[['area', 'centroid-0', 'centroid-1']].plot.scatter(x='centroid-0', y='centroid-1', c='area', colormap='viridis')
    >>> plt.show()
    """    
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    all_dataframes = []  # List to store individual DataFrames

    # Recursively walk through input folder and process .npy files
    file_list = recursively_get_all_filepaths(input_folder, '.npy')
    for file in file_list:
        try:
            all_dataframes.append(apply_regionprops(file, properties_list))

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pd.concat(all_dataframes, ignore_index=True)
