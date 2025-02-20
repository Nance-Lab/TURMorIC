## Colin

-Documentation and markdown to describe how code works  
-Code markdowns with non-Python specific explanations  
-Presentation-worthy outputs  
-Retrievable errors associated with analysis functions  
-Accompanying documentation on how to analyze output  
-Easy-to-execute functions  
-Functions to parse out morphology of astrocytes and oligodendrocytes  
-Training and testing data  
-Test functions for novel codes  
-Limited hard-coded parameters for tuning purposes  
-Functions that are generalized to cells  
-Specific functions for different types of cells like brain cells vs. gut cells  


## Heather
GUI:
MainWindow(QWidget):
    Description: controls the interface displays and passes commands to the backend
    Important parameters and roles:
        General_Layout->creates layout for pages
        Page_Layout->creates layout for buttons
        Page_Stack->creates stack of pages
        Page_Index->keeps track of current page
        Page_Buttons->keeps track of buttons on current page
        Page_Controls->keeps track of controls on current page
        Page_Controls_Layout->creates layout for controls
        
        Connects to CentralNode class to pass commands to backend
        Stores button names, default values, positions, and page orders
        Displays controls and images
    Functions:
    - Generate_Page: generates a page for the GUI which may contain unique buttons or controls
    - Generate_Controls: generates buttons, dropdowns, and sliders for the GUI
    - Update_Controls: updates the values of the controls on the GUI and the page
    - Update_Image: updates the image displayed on the GUI
    - Update_Parameters: updates the parameter shown on the GUI
CentralNode(QObject):
    Description: controls the backend of the GUI
    Important parameters and roles:
       Connects to MainWindow class to receive commands from the GUI
       Connects to ImageNode class to receive images from the backend
       Connects to FunctionNode class to pass commands to the backend
       Stores the current state of the GUI and parameters
       Controls the worker threads for image processing 
    Functions:
    - Update_Parameters: updates the parameters of the backend
    - Update_Image: updates the image displayed on the GUI
    - Update_Controls: updates the values of the controls on the backend
    - Stores and sets file paths for saving parameters and images
ImageNode(QThread):
    Description: controls the image display of the GUI
    Important parameters and roles:
       Connects to CentralNode class to receive images and data from the backend
       Updates the image displayed on the GUI based on current parameters and control_index
    Functions:
       Sends image signal to central node to update the image
FunctionNode(QObject):
    Description: controls the way that functions are called and executed (ie async, lazy, ect)
    Important parameters and roles:
       Connects to CentralNode class to receive commands from the GUI
       Stores the current state of the functions and parameters
    Functions:
        - Execute_Function: executes the function based on the current parameters and control_index
Exec_Function(QThread):
    Description: executes the function in a separate thread
    Important parameters and roles:
        Connects to FunctionNode class to receive commands from the GUI
        Calls Support_function.py file to execute functions based on current parameters and control_index
    Functions:
        - run: executes the function in a separate thread
        - Update_Function: updates the function based on the current parameters and control_index
        - Function wrappers: functions that wrap the actual function to be executed from Support_function.py

Pages of GUI:
    - FirstPage: ask if you want to upload parameters for batch process or go through the parameter selection process
    - SecondPageA: Filters and thresholding
    - ThirdPageA: Aligns boundaries to remove rotational variance 
    - FourthPageA: Principal Component Analysis
    - FifthPageA: Clustering, and class assignment

    - SecondPageB: batch process: selects file location and uploads parameters
    - ThirdPageB: shows estimated time remaining for processing

    - FinalPage: displays results and asks if you want to save results and/or parameters

Support_function.py:
•	Description: Contains the core functions for VAMPIRE analysis.
•	Important Parameters and Roles:
o	Handles data processing tasks such as filtering, thresholding, PCA, and clustering.
o	Interacts with the mainbody.py to execute the analysis pipeline.
o	Contains functions for data manipulation and analysis.

Added functions:
Workflow:
    Description: Splits images into quadrants, selects training and testing image sets, and prepares data for VAMPIRE analysis. Designed for Phuong's BEV treatment data.
    Important Parameters and Roles:
        Splits segmented microglia images into four quadrants for each image in a specified folder.
        Selects training and testing datasets, organizing images into directories for model building and testing.
        Renames and organizes images for VAMPIRE analysis according to a specific naming convention.
        Creates CSV files to store model-building and application information.
        Invokes VAMPIRE analysis at the end, passing relevant paths to the model.

Coloring Segmented Images According to Shape Mode:
    Description: Colors segmented images based on shape modes with custom colors, visualizing the output, and saving the recolored images.
    Important Parameters and Roles:
        Handles image segmentation and labeling using shape modes from CSV.
        Applies specific colors to different shape modes (Blue, Orange, Green, Red, Purple).
        Automates labeling for multiple images based on provided file paths.
        Saves the recolored images at specified output paths.

Thresholding:
    Description: Tests and applies various thresholding techniques from Sci-kit Image on images.
    Important Parameters and Roles:
        Applies multiple thresholding methods (Li, Otsu, Mean, etc.) to images.
        Uses a custom function for folder filtering and prepares image lists for threshold testing.
        Saves the thresholded images and moves them to a dedicated folder.
        Performs analysis to determine the optimal threshold (Li threshold) for microglia identification based on visual inspection.
        Removes small objects below a size threshold (based on microglia size) using Sci-kit's morphology functions.

Microglia OGD Cell Analysis:
    Description: Quantifies cell features of already segmented microglia images using multiple image analysis techniques.
    Important Parameters and Roles:
        Reads pre-processed microglia images (from the li_thresh folder).
        Applies image segmentation using skimage.measure.label and calculates multiple region properties (area, centroid, eccentricity, etc.) for each labeled region.
        Calculates additional features like circularity and aspect ratio for microglia cells.
        Saves the results in a CSV file, including the properties of all segmented cells.
        Generates a summary CSV file with average properties for each image.

Current components of codebase:
vampire.py:
•	Description: Launches the interface for the VAMPIRE GUI.
•	Important Parameters and Roles:
o	Sets up the graphical user interface for the VAMPIRE tool.
o	Implements functions for user input and interacts with other modules (e.g., loading CSV files, selecting output folders).
o	Controls the workflow through functions such as Model, getcsv, and getdir.
________________________________________
mainbody.py:
•	Description: Handles the core VAMPIRE analysis process, including:
1.	Registration: Aligns boundaries to remove rotational variance.
2.	PCA (Principal Component Analysis): Reduces data dimensions for better analysis.
3.	Clustering: Groups data based on similarities.
•	Functions:
o	mainbody: Executes the complete pipeline based on model building or application.
o	collect_seleced_bstack: Collects and prepares the boundary stack for further processing.
________________________________________
collect_selected_bstack.py:
•	Description: Reads the boundaries of cells or nuclei from a CSV file and prepares them for analysis.
•	Functions:
o	collect_seleced_bstack: Gathers boundary data from specified locations (either to build a model or apply an existing one).
________________________________________
bdreg.py:
•	Description: Registers the boundaries of cells or nuclei, ensuring alignment across images.
•	Functions:
o	bdreg: Registers the boundaries by resampling and applying Singular Value Decomposition (SVD).
________________________________________
pca_bdreg.py:
•	Description: Applies PCA to registered boundaries to reduce dimensions and capture the most relevant features.
•	Functions:
o	pca_bdreg: Handles the PCA transformation of boundary data.
________________________________________
PCA_custom.py:
•	Description: Custom implementation of PCA to process boundary data.
•	Functions:
o	PCA_custom: Returns the transformed data and regenerates the original data from the PCA transformation.
________________________________________
clusterSM.py:
•	Description: Applies clustering (K-means) to boundary data after PCA processing to categorize cells or nuclei.
•	Functions:
o	clusterSM: Performs K-means clustering and assigns cluster labels to each cell or nucleus.
________________________________________
update_csv.py:
•	Description: Generates and updates the VAMPIRE datasheet with cluster labels after analysis.
•	Functions:
o	update_csv: Writes the results, including cluster assignments, into a CSV file.
________________________________________
Helper Functions:
•	getdir: Opens a file dialog to choose a directory.
•	getcsv: Opens a file dialog to select a CSV file.
•	Model: Handles model building or application, depending on the user’s choice.
•	vampire(): Initializes and runs the GUI loop.

## Krista

## Muna

## Sergi
Components: 

1.  Image Preprocessing Functions:

    What it does: This component formats, filters, and enhances images based on file type and quality, preparing them for further analysis. It handles variations in image 
    formats, pixel qualities, and noise levels, ensuring that the images are standardized and optimized for tasks like segmentation and classification.

    Inputs: Image files in various formats (e.g., TIFF, PNG, JPEG, etc.). Image metadata (e.g., pixel dimensions, quality, stain type).Optional user-defined parameters for 
    customization (e.g., noise threshold, contrast levels).

    Outputs: Preprocessed, standardized images ready for segmentation or analysis on a separated file directory. Error or warning logs indicating any issues with image 
    quality or unsupported formats. A report or visual output demonstrating the before-and-after effects of preprocessing, such as improved contrast or reduced noise.

    How it uses other components: Function for Multi-Stain Image Segmentation/Identification and Classification (the preprocessed images are passed to the segmentation 
    function afterwards). Proper preprocessing helps ensure that these images meet the quality standards required for effective segmentation. Quantification Metrics (after 
    preprocessing, the images are used for quantification tasks (e.g., counting cells or measuring areas), ensuring that the data fed into these metrics is standarized).
  
2. Function for Multi-Stain Image Segmentation/identification and classification:

   What it does: A function that can segment and process images with different stain types, specifically tailored for microglia and oligodendrocytes in WSI brain images.

   Inputs: Whole Slide Images (WSI) of brain tissue with two distinct stains (for microglia and oligodendrocytes). Image metadata (such as stain type, resolution, and other 
   relevant conditions).
   
   Outputs:Segmented regions identifying microglia and oligodendrocytes in the images. Classification results of segmented regions, indicating whether the cell is         
   microglia, oligodendrocyte, or other cell types. A visual output with highlighted regions corresponding to the identified cells. A CSV or similar file containing the 
   count of identified cell types along with their spatial information.

   How it uses other components: Image Preprocessing Functions (Before segmentation, the function uses image preprocessing components to enhance the quality and 
   consistency). Interactive GUI (It interacts with an interactive graphical user interface (GUI) to provide real-time feedback, allowing users to adjust 
   parameters (e.g., thresholds or stain sensitivity) as needed for refinement).
   
4. Interactive GUI for Image Segmentation and Analysis:

   What it does: This component provides a simple and intuitive graphical user interface (GUI) that allows users to interact with images without needing to     
   dive into the code. It enables users to view images, adjust segmentation thresholds, and customize settings (e.g., contrast, resolution) in real-time.

   Inputs: Preprocessed Image files for segmentation and analysis (e.g., brain tissue slices), User inputs for threshold adjustments and visualization preferences (e.g., 
   adjusting stain sensitivity, zooming in on areas of interest).

   Outputs: Interactive display of images with real-time segmentation overlays. Visual feedback on changes made to thresholds, stain sensitivity, or resolution adjustments.
   Output files containing the adjusted parameters and segmented images ready for further analysis or export).
   
   How it uses other components: Image Preprocessing Functions (The GUI interfaces with image preprocessing functions to allow users to view and adjust preprocessed images, 
   it ensures that the images presented for segmentation are enhanced and standardized). Function for Image Segmentation/Identification and Classification (the GUI provides 
   real-time feedback on the segmentation process. Quantification Metrics (After segmentation, the GUI can display real-time quantification metrics (e.g., cell 
   count, area measurement) for the brain cells, allowingusers to assess and modify parameters for optimal results.

6. Documentation for Non-Programmers/all level users:

   What it does: This component provides non-technical documentation aimed at users with no programming experience. It explains the tool’s outputs and guides users on how 
   to interpret them, ensuring that individuals like Kristin and Tania can effectively use the tool for their research without needing to understand the underlying code. 
   The documentation includes step-by-step instructions, explanations of key concepts, and visual aids, and instructions on how to cite this tool when used by users.

   Inputs: Descriptions of common image analysis tasks (e.g., cell segmentation, threshold adjustment).

   Outputs: Comprehensive guides detailing how to interpret segmentation results, understanding segmentation overlays, and reviewing performance metrics. Visual aids (e.g., 
   annotated images, flowcharts) to demonstrate common tasks, making it easier for non-technical users to follow along. Glossary of terms to help users understand key 
   concepts.

   How it uses other components: Interactive GUI for Image Segmentation and Analysis (The documentation provides guidance on how to use the GUI, including how to adjust 
   settings, interact with images, and interpret the results). Image Preprocessing Functions (the documentation explains how the preprocessing functions work and why they 
   are important. Function for Image Segmentation/Identification and Classification (it provides clear explanations of the segmentation process, helping users understand 
   how the tool differentiates between cells and how they can interpret the segmented images. Quantification Metrics: The documentation explains the quantification metrics 
   output by the tool.

7. Quantification Metrics:

   What it does: Function that automatically quantify microglia and oligodendrocytes based on their segmentation, including cell shape and counts, area, and morphology 
   measurements. 

   Inputs: Segmented images containing microglia and oligodendrocytes (e.g., from previous segmentation steps).

   Output: Cell count (The number of microglia and oligodendrocytes identified in the segmented images), Area measurements; the area (in pixels or physical units) occupied 
   by the microglia and oligodendrocytes. Morphological data: Key morphological features of the cells (e.g., cell perimeter, aspect ratio, shape descriptors like 
   circularity, elongation).

   How it uses other components: Image Preprocessing Functions (the quantification functions rely on preprocessed images to ensure that the data being analyzed is of high 
   quality, with noise removed and contrast enhanced for accurate measurements). Function for Image Segmentation/Identification and Classification (The quantification 
   functions work directly with the results from the segmentation functions to count and measure the identified microglia and oligodendrocytes). Interactive GUI for Image 
   Segmentation and Analysis (users can adjust segmentation parameters using the GUI, and the quantification metrics update in real-time to reflect the changes made).
   Documentation for Non-Programmers (the documentation helps users understand the quantification outputs by explaining the meaning of the different metrics, e.g. what the 
   cell count represents or how to interpret the morphological data)
   
8. Demo Data Set for new users/ Training Data examples:

    What it does: This component provides a curated set of brain tissue images from a compilation of experiments, along with a pre-existing training dataset for testing and 
    validating algorithms. This dataset includes both raw and processed images, annotations, and ground truth labels for different cell types and injury stages to 
    facilitate the testing and validation of segmentation and classification algorithms.

    Inputs: Raw image data from demo experiments (e.g., brain tissue slices from different injury stages), Pre-processed data with annotations or labeled cells (e.g.,     
    microglia, oligodendrocytes). 

    Outputs: Segmented images showing cells and other relevant structures, with labels indicating the classification of different cell types (e.g., microglia, 
    oligodendrocytes).Performance metrics (e.g., accuracy, precision, recall) of segmentation algorithms on the provided data. A test report summarizing the metrics and 
    providing a brief analysis on those. Visual outputs showing comparative results of the original versus segmented images.

    How it uses other components: Image Preprocessing Functions: The demo dataset leverages preprocessing functions to standardize and enhance images before they are used 
    in train and testing or real-world analysis. Function for Multi-Stain Image Segmentation/Identification and Classification (the dataset is used to test and validate 
    segmentation algorithms tailored to different brain cell types, including microglia and oligodendrocytes. It provides a testbed for assessing how well these functions 
    can classify cells in user-provided images) Quantification Metrics: The example data set is analyzed using quantification tools to assess the number and type of cells 
    identified in the WSI images.
