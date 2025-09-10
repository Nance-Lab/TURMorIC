Quick Start Guide
=================

This guide will walk you through the basic TURMorIC workflow for analyzing brain slice images.

Basic Workflow
--------------

The typical TURMorIC analysis consists of these steps:

1. **File Conversion**: Convert ND2 files to TIFF format
2. **Thresholding**: Apply binary thresholding to segment cells
3. **Analysis**: Extract morphological properties
4. **Visualization**: Plot results and statistics

Step 1: Convert ND2 to TIFF
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have ND2 microscopy files, first convert them to TIFF:

.. code-block:: python

   from turmoric.image_process import nd2_to_tif
   
   # Convert a single file
   nd2_to_tif("path/to/images/", "sample.nd2")
   
   # Or use the command line script
   # python scripts/nd2_to_tif.py input_folder output_folder

Step 2: Apply Thresholding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply Li thresholding to segment microglia:

.. code-block:: python

   from turmoric.apply_thresholds import apply_li_threshold
   
   # Apply to a single image
   binary_mask = apply_li_threshold("image.tif", channel=1)
   
   # Or process entire directories
   from turmoric.apply_thresholds import apply_threshold_recursively
   
   apply_threshold_recursively(
       input_folder="path/to/images/",
       output_folder="path/to/masks/",
       threshold_function=apply_li_threshold
   )

Step 3: Extract Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze the segmented cells:

.. code-block:: python

   from turmoric.cell_analysis import apply_regionprops_recursively
   import numpy as np
   
   # Extract morphological properties
   properties = ['area', 'perimeter', 'eccentricity', 'solidity']
   df = apply_regionprops_recursively("path/to/masks/", properties)
   
   # Calculate additional metrics
   df['circularity'] = 4 * np.pi * df['area'] / df['perimeter']**2
   df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']

Step 4: Organize Data for Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organize your data to prevent data leakage:

.. code-block:: python

   from turmoric.utils import organize_files_without_leakage
   
   groups = ["control", "treatment"]
   conditions = ["condition_A", "condition_B"]
   
   organize_files_without_leakage(
       base_dir="path/to/data/",
       train_dir="path/to/training/",
       test_dir="path/to/testing/",
       groups=groups,
       treatment_conditions=conditions,
       test_size=0.2
   )

Command Line Usage
------------------

TURMorIC provides several command-line scripts for common tasks:

**Apply Li thresholding to images:**

.. code-block:: bash

   python scripts/apply_single_threshold.py input_folder output_folder

**Compare multiple thresholding methods:**

.. code-block:: bash

   python scripts/try_all_thresh.py input_folder output_folder

**Extract region properties:**

.. code-block:: bash

   python scripts/apply_regionprops.py input_folder output.csv

**Split files for training/testing:**

.. code-block:: bash

   python scripts/train_test_split.py

Example: Complete Analysis Pipeline
-----------------------------------

Here's a complete example analyzing a dataset:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from turmoric.apply_thresholds import apply_threshold_recursively, apply_li_threshold
   from turmoric.cell_analysis import apply_regionprops_recursively
   from turmoric.utils import organize_files_without_leakage
   
   # 1. Organize data
   organize_files_without_leakage(
       base_dir="raw_data/",
       train_dir="training/",
       test_dir="testing/",
       groups=["control", "treatment"],
       treatment_conditions=["drug_A", "drug_B"],
       test_size=0.2
   )
   
   # 2. Apply thresholding
   apply_threshold_recursively(
       input_folder="training/",
       output_folder="training_masks/",
       threshold_function=apply_li_threshold
   )
   
   # 3. Extract properties
   properties = ['area', 'perimeter', 'eccentricity', 'solidity']
   df = apply_regionprops_recursively("training_masks/", properties)
   
   # 4. Calculate derived metrics
   df['circularity'] = 4 * np.pi * df['area'] / df['perimeter']**2
   
   # 5. Save results
   df.to_csv("morphology_analysis.csv", index=False)
   
   print(f"Analyzed {len(df)} cells across conditions")
   print(df.groupby('treatment')['area'].describe())

What's Next?
------------

* Check out the :doc:`tutorials/index` for detailed examples
* Read the :doc:`api/modules` for complete function documentation
* See example datasets in the ``example_dataset/`` directory