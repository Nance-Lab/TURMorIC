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
