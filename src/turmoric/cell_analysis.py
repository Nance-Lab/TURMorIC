import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from turmoric.utils import recursively_get_all_filepaths

from skimage.segmentation import clear_border

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
    binary_mask = np.load(file, allow_pickle=True)

    # Get dimensions
    height, width = binary_mask.shape
    mid_h = height // 2
    mid_w = width // 2

    # Split into 4 quadrants
    quadrants = {
        'quad1': binary_mask[0:mid_h, 0:mid_w],           # Top-left
        'quad2': binary_mask[0:mid_h, mid_w:width],       # Top-right
        'quad3': binary_mask[mid_h:height, 0:mid_w],      # Bottom-left
        'quad4': binary_mask[mid_h:height, mid_w:width]   # Bottom-right
    }

    # Collect properties from all quadrants
    all_props_dfs = []

    for quad_name, quad_mask in quadrants.items():
        # Clear objects touching the border of this quadrant
        quad_mask = clear_border(quad_mask)
        
        # Label connected regions in this quadrant
        label_image = label(quad_mask)
        
        # Measure properties
        props = regionprops_table(label_image, properties=properties_list)
        
        # Create DataFrame for this quadrant
        props_df = pd.DataFrame(props)
        props_df['filename'] = file
        props_df['quadrant'] = quad_name
        
        all_props_dfs.append(props_df)

    # Combine all quadrants into one DataFrame
    combined_props_df = pd.concat(all_props_dfs, ignore_index=True)

    return combined_props_df


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
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)

                try:
                    df = apply_regionprops(file_path, properties_list)
                    all_dataframes.append(df)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return pd.concat(all_dataframes, ignore_index=True)
