Installation
============

Requirements
------------

* Python 3.9 or higher
* Git for cloning the repository

Dependencies
~~~~~~~~~~~~

TURMorIC depends on several scientific Python packages:

* **Core scientific stack**: NumPy, SciPy, pandas
* **Image processing**: scikit-image, opencv-python
* **Machine learning**: scikit-learn
* **Visualization**: matplotlib, seaborn
* **File handling**: tifffile, nd2
* **VAMPIRE analysis**: vampire-analysis

Installation Methods
--------------------

Method 1: Conda Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Nance-Lab/TURMorIC.git
   cd TURMorIC

   # Create conda environment from file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate turmoric
   
   # Install the package in development mode
   pip install -e .

Method 2: pip installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Nance-Lab/TURMorIC.git
   cd TURMorIC

   # Create virtual environment
   python -m venv turmoric-env
   
   # Activate virtual environment
   # On macOS/Linux:
   source turmoric-env/bin/activate
   # On Windows:
   turmoric-env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install the package
   pip install -e .

Verification
------------

To verify your installation works correctly:

.. code-block:: python

   import turmoric
   from turmoric.apply_thresholds import apply_li_threshold
   from turmoric.cell_analysis import apply_regionprops
   
   print("TURMorIC installation successful!")

Development Installation
------------------------

For developers who want to contribute to TURMorIC:

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/TURMorIC.git
   cd TURMorIC
   
   # Install in development mode with test dependencies
   conda env create -f environment.yml
   conda activate turmoric
   pip install -e ".[dev]"
   
   # Run tests to verify everything works
   pytest

Common Issues
-------------

**Import errors**
   Make sure you've activated the correct conda environment and installed all dependencies.

**ND2 file reading issues**
   The ``nd2`` package requires specific system libraries. If you encounter issues, try:
   
   .. code-block:: bash
   
      conda install -c conda-forge nd2reader

**VAMPIRE model issues**
   The vampire-analysis package may need to be installed separately:
   
   .. code-block:: bash
   
      pip install vampire-analysis