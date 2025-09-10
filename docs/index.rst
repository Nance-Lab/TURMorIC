TURMorIC: Turmeric-enhanced Unsupervised Recognition of Microglia Image Characteristics
==================================================================================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/Nance-Lab/TURMorIC/blob/main/LICENSE
   :alt: License

TURMorIC is a Python package for automated analysis of brain slice images, specifically designed for microglia morphology quantification. Built upon the VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) framework, TURMorIC provides a streamlined pipeline for researchers to analyze brain tissue images with minimal manual intervention.

Key Features
------------

* **Automated Image Processing**: Convert ND2 microscopy files to standard formats
* **Advanced Thresholding**: Multiple thresholding algorithms (Li, Otsu, Mean, etc.)
* **Morphological Analysis**: Extract detailed cellular properties using regionprops
* **VAMPIRE Integration**: Train and apply VAMPIRE models for morphology classification
* **Batch Processing**: Handle large datasets efficiently
* **Reproducible Workflows**: Prevent data leakage with proper train/test splitting

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Nance-Lab/TURMorIC.git
   cd TURMorIC

   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate turmoric

   # Install the package
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from turmoric.apply_thresholds import apply_li_threshold
   from turmoric.cell_analysis import apply_regionprops

   # Apply Li thresholding to an image
   binary_mask = apply_li_threshold("path/to/image.tif", channel=1)

   # Extract morphological properties
   properties = apply_regionprops("path/to/mask.npy", ["area", "perimeter"])

🔍 **Looking for specific functions?** → :doc:`api/function_index`

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules
   
.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: About
   
   citation
   license

Most Used Functions
-------------------

New users typically start with these functions:

.. autosummary::
   :toctree: _autosummary

   ~turmoric.apply_thresholds.apply_li_threshold
   ~turmoric.cell_analysis.apply_regionprops_recursively
   ~turmoric.utils.organize_files_without_leakage
   ~turmoric.image_process.nd2_to_tif

Citation
--------

If you use TURMorIC in your research, pleaseTURMorIC: Turmeric-enhanced Unsupervised Recognition of Microglia Image Characteristics
==================================================================================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/Nance-Lab/TURMorIC/blob/main/LICENSE
   :alt: License

TURMorIC is a Python package for automated analysis of brain slice images, specifically designed for microglia morphology quantification. Built upon the VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) framework, TURMorIC provides a streamlined pipeline for researchers to analyze brain tissue images with minimal manual intervention.

Key Features
------------

* **Automated Image Processing**: Convert ND2 microscopy files to standard formats
* **Advanced Thresholding**: Multiple thresholding algorithms (Li, Otsu, Mean, etc.)
* **Morphological Analysis**: Extract detailed cellular properties using regionprops
* **VAMPIRE Integration**: Train and apply VAMPIRE models for morphology classification
* **Batch Processing**: Handle large datasets efficiently
* **Reproducible Workflows**: Prevent data leakage with proper train/test splitting

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Nance-Lab/TURMorIC.git
   cd TURMorIC

   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate turmoric

   # Install the package
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from turmoric.apply_thresholds import apply_li_threshold
   from turmoric.cell_analysis import apply_regionprops

   # Apply Li thresholding to an image
   binary_mask = apply_li_threshold("path/to/image.tif", channel=1)

   # Extract morphological properties
   properties = apply_regionprops("path/to/mask.npy", ["area", "perimeter"])

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/modules
   
.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: About
   
   citation
   license

Citation
--------

If you use TURMorIC in your research, please cite:

.. code-block:: bibtex

   @software{turmoric2025,
     title={TURMorIC: Turmeric-enhanced Unsupervised Recognition of Microglia Image Characteristics},
     author={Schimek, Nels and Phommatha, Krista and Landis, Colin and Yase, Muna and Wood, Heather and Mayta, Sergi},
     year={2025},
     url={https://github.com/Nance-Lab/TURMorIC}
   }

And the original VAMPIRE method:

.. code-block:: bibtex

   @article{phillip2021robust,
     title={A robust unsupervised machine-learning method to quantify the morphological heterogeneity of cells and nuclei},
     author={Phillip, Jude M and Han, Kyu-Sang and Chen, Wei-Chiang and Wirtz, Denis and Wu, Pei-Hsun},
     journal={Nature protocols},
     volume={16},
     number={2},
     pages={754--774},
     year={2021},
     publisher={Nature Publishing Group}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`