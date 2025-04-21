import os
import skimage
import numpy as np
from nd2 import ND2File
import tifffile

def load_npy_file(path, file_name):
    npy_path = os.path.join(path, file_name)

    try:
        image_data = skimage.measure.label(np.load(npy_path))
    except Exception as e:
        print(f"error processing '{npy_path}': {e}")

    return image_data

def normalize_npy_data(npy_image_data):
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
    
def nd2_to_tif(path, file_name):
    nd2_path = os.path.join(path, file_name)
    tif_path = os.path.join(path, file_name.replace(".nd2", ".tif"))
    
    with ND2File(nd2_path) as nd2_file:
        nd2_data = nd2_file.asarray()

        tifffile.imwrite(tif_path, nd2_data)

    
 
def load_tif_file():
    pass