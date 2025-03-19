## GUI Breakdown
### **MainWindow(QWidget):**
Description:Controls the interface displays and passes commands to the backend
-Connects to CentralNode class to pass commands to backend
-Stores button names, default values, positions, and page orders
Use Case:
-Allows the user to interact with the system easily by providing an intuitive, user-friendly interface with clearly defined buttons and controls.
-Users want an error-free experience, and the GUI automatically handles input validation and error checking to ensure that the system behaves predictably.
-Users desire convenience, and the GUI can remember their settings across different sessions, ensuring they don’t have to re-enter their preferences every time.
Displays controls and images
-Important parameters and roles:
-General_Layout: Creates layout for pages.
-Page_Layout: Creates layout for buttons.
-Page_Stack: Creates a stack of pages.
-Page_Index: Keeps track of the current page.
-Page_Buttons: Keeps track of buttons on the current page.
-Page_Controls: Keeps track of controls on the current page.
-Page_Controls_Layout: Creates layout for controls.
Functions:
-Generate_Page: Generates a page for the GUI which may contain unique buttons or controls.
-Generate_Controls: Generates buttons, dropdowns, and sliders for the GUI.
-Update_Controls: Updates the values of the controls on the GUI and the page.
-Update_Image: Updates the image displayed on the GUI.
-Update_Parameters: Updates the parameters shown on the GUI.
Pages of GUI:
-FirstPage: Ask if you want to upload parameters for a batch process or go through the parameter selection process.
-SecondPageA: Filters and thresholding.
-ThirdPageA: Aligns boundaries to remove rotational variance.
-FourthPageA: Principal Component Analysis (PCA).
-FifthPageA: Clustering and class assignment.
-SecondPageB: Batch process: Selects file location and uploads parameters.
-ThirdPageB: Shows estimated time remaining for processing.
-FinalPage: Displays results and asks if you want to save results and/or parameters.
_________________________________________________________________
## **CentralNode(QObject):**
Description: Controls the backend of the GUI.
-Connects to MainWindow class to receive commands from the GUI.
-Connects to ImageHandler class to receive images from the backend.
-Connects to FunctionHandler class to pass commands to the backend.
-Stores the current state of the GUI and parameters.
-Controls the worker threads for image processing.

Use Case:
-Manages the backend communication between the GUI and the various processing modules (image handling, functions execution) to ensure seamless interaction between user commands and backend processing.
-Users want to easily manage their data and parameters, and the CentralNode ensures that all settings and inputs are tracked across the system.
-Allows the user to work efficiently, as it manages the threading and execution of background tasks without freezing the GUI, allowing continuous interaction with the system.

Important parameters and roles:
-Connects to MainWindow: Receives commands from the MainWindow to manage tasks like image loading, applying filters, or running models.
-Connects to ImageHandler: Facilitates communication between the frontend and the image processing thread.
-Connects to FunctionHandler: Initiates functions based on user input and manages function execution asynchronously.
-Stores current state of parameters: Manages the internal state of the GUI, such as selected filters, image data, and function parameters.
-Manages worker threads: Controls the background workers for image manipulation, model building, and other tasks.

Functions:
-Update_Parameters: Updates parameters on the backend and passes them to the relevant worker thread.
-Update_Image: Emits signals to update the image displayed in the GUI.
-Update_Controls: Updates the backend state of controls based on user interaction.
-File management: Stores and sets file paths for saving parameters and images.
_______________________________________________________
## ** ImageHandler(QThread):** 
Description:
-Handles image processing functions and sends processed image data to CentralNode.
-Performs image manipulation tasks in a separate thread, such as applying filters and performing transformations.
-Ensures GUI remains responsive by offloading intensive tasks to the background.

Use Case:
-Allows users to manipulate images without freezing the interface, enabling real-time interaction with the system while complex image processing is happening in the background.
-Users expect quick results, and the ImageHandler ensures that images are processed efficiently in the background without affecting the GUI’s responsiveness.
-Users want the system to be intuitive, and the image processing happens automatically based on the user’s inputs, with no need for them to manage the backend.

Important parameters and roles:
-Connects to CentralNode: Receives image data from the backend and processes it.
-Sends processed image signals: Sends the processed image back to CentralNode to update the GUI display.
Functions:
-apply_filter_to_image: Applies a filter to an image (e.g., thresholding, Sobel edge detection) based on the user’s selection.
-emit_updated_image: Converts the processed image to a QImage and emits it for display in the GUI.
____________________________________________________________
## **FunctionHandler(QThread):**
Description: 
-Handles the execution of background functions (such as PCA or clustering) that do not directly affect the image but modify other data.
-Executes tasks asynchronously to keep the GUI responsive.

Use Case:
-Helps users run computationally heavy functions (like PCA or clustering) without interrupting their workflow, as these tasks are executed in the background.
-Users want to perform complex tasks like dimensionality reduction or data classification without having to wait for the interface to respond, and FunctionHandler enables this by offloading the work to a separate thread.

Important parameters and roles:
-Connects to CentralNode: Receives function-related commands from the GUI and processes them in the background.
-Stores current state of function parameters: Manages the parameters of functions like PCA, clustering, etc.

Functions:
-run: Executes the function in the background, ensuring that the GUI stays responsive.
-select_function: Determines the function to execute based on the current control index (e.g., PCA ).
__________________________________________________________________________
## **ModelHandler(QThread):**
Description: 
-Handles the process of building or applying a model in a background thread to avoid blocking the GUI.
-Works with FunctionHandler to execute model-related operations such as clustering or PCA.

Use Case:
-Helps users build or apply models without freezing the UI, allowing them to perform complex modeling tasks (like clustering) while continuing to interact with other parts of the application.
-Users expect models to be built quickly and without errors, and the ModelHandler ensures the task is performed reliably and in the background, so the user doesn’t need to wait.

Important parameters and roles:
-Connects to CentralNode: Sends progress updates and status changes during model building or application.
-Uses output path and clusters: Handles output paths and the number of clusters for certain models.

Functions:
-run: Starts the model-building or application process in a separate thread.
-mainbody: Contains core logic for the model-building or applying process, including progress updates and status messaging.
-update_status: Sends status updates (e.g., "Modeling initiated…", "Modeling completed") to the GUI.
-update_progress: Sends percentage-based progress updates to indicate task completion.
-apply_model: Applies the model to the data if build_model is False and updates the GUI accordingly.

  
