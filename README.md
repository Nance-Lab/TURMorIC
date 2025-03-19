# Planning

![logo](docs/TURMorIC.png)

Developers: Colin, Krista, Sergi, Muna, Heather

This repository contains the test functions for the updated VAMPIRE (Visually Aided Morpho-Phenotyping Image Recognition) program. The updated program humorously titled "TURMERIC" serves to extract the functions of VAMPIRE and create callable function packages and a user interface. These will allow any user interested in performing brain slice image analysis to do so more readily.

# Inside this repository you will find:
- the MIT license
- the .yml and .toml for the software configuration
- the docs, example_dataset, notebooks, src, and test directories (detailed below)

# docs Directory 
This directory contains documents related to the initial design process for the software package including use cases, user stories, and identified components as well as the team logo and details on the GUI. Finally, there is a slide deck with a presentation given to the CHEM E 546: Software Engineering for Molecular Science and Engineering showcasing the design process of the TURMorIC project.

# example_dataset Directory
This directory includes example .tiff images and expected outputs, like shape mode distribution, for the preprocessing as well as some examples of the segmentation process as well. These serve as potential practice images for understanding the existing code pipeline.

# notebooks Directory 
This directory contains Jupyter notebooks that hold the previous code from VAMPIRE in the subdirectory "notebooks_from_existing_codebase" as the developers work on creating the TURMorIC functions. The subdirectory "temporary_notebooks_for_concept_testing" are Jupyter notebooks that the team can use to work on code before pushing to the main branch.

# src/Nosferatu Directory 
This directory contains all of the source code for the functions of the TURMorIC package. This will be where functions will be committed to the main branch. The directory also holds the GUI components directory and the __init__.py file.
#  src/GUI Directory

# test Directory 
This directory holds all of the test functions for each of the components. This includes functions to test if the image directory exists, cleaning the file and folder paths, adjusting contrast among other preprocessing steps.

# **Installation Instructions **
To get started with TURMERIC, follow these steps:

Prerequisites:
Python version 3.8 or higher.
Git for cloning the repository.
1. Clone the repository
Start by cloning the repository to your local machine:

bash
Copy
git clone https://github.com/YourUsername/TURMERIC.git
cd TURMERIC
2. Create a virtual environment (optional but recommended)
It's a good practice to create a virtual environment to isolate dependencies:

bash
Copy
python -m venv env
Activate the virtual environment:

On macOS/Linux:
bash
Copy
source env/bin/activate
On Windows:
bash
Copy
.\env\Scripts\activate
3. Install dependencies
Once the environment is set up, install the required dependencies.

bash
Copy
conda env create -f environment.yml
Activate the Conda environment:

bash
Copy
conda activate TURMERIC

This will install all necessary Python packages for the project.

## **Usage Instructions**
After setting up the environment, you can use the TURMERIC package to perform brain slice image analysis.

# Run the GUI:
The user interface for image processing and model building is located in the src/GUI directory. You can run the main GUI file from there. Currently its functions include loading an image, applying a thresholding binary filter and saving the image.
bash
Copy
python src/GUI_components/MainWindow.py

# Run Tests:
To verify if everything is working correctly, run the tests in the test/ directory.
bash
Copy
pytest

Example Datasets
The example_dataset/ directory contains raw sample .tiff images for you to practice preprocessing images, segmentation images and example outputs from the original VAMPIRE software

# Citation:
Phillip, J. M.; Han, K.-S.; Chen, W.-C.; Wirtz, D.; Wu, P.-H. A Robust Unsupervised Machine-Learning Method to Quantify the Morphological Heterogeneity of Cells and Nuclei. Nature protocols 2021, 16 (2), 754–774. https://doi.org/10.1038/s41596-020-00432-x.
‌
