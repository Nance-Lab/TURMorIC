## GUI Breakdown
### **MainWindow(QWidget):**
Controls the interface displays and passes commands to the backend
Connects to CentralNode class to pass commands to backend
Stores button names, default values, positions, and page orders
Displays controls and images
    
Important parameters and roles:
- General_Layout->creates layout for pages
- Page_Layout->creates layout for buttons
- Page_Stack->creates stack of pages
- Page_Index->keeps track of current page
- Page_Buttons->keeps track of buttons on current page
- Page_Controls->keeps track of controls on current page
- Page_Controls_Layout->creates layout for controls

Functions:
- Generate_Page: generates a page for the GUI which may contain unique buttons or controls
- Generate_Controls: generates buttons, dropdowns, and sliders for the GUI
- Update_Controls: updates the values of the controls on the GUI and the page
- Update_Image: updates the image displayed on the GUI
- Update_Parameters: updates the parameter shown on the 

Pages of GUI:
- FirstPage: ask if you want to upload parameters for batch process or go through the parameter selection process
- SecondPageA: Filters and thresholding
- ThirdPageA: Aligns boundaries to remove rotational variance 
- FourthPageA: Principal Component Analysis
- FifthPageA: Clustering, and class assignment
- SecondPageB: batch process: selects file location and uploads parameters
- ThirdPageB: shows estimated time remaining for processing
- FinalPage: displays results and asks if you want to save results and/or parameters

### **CentralNode(QObject):**
Description: controls the backend of the GUI

Important parameters and roles:
- Connects to MainWindow class to receive commands from the GUI
- Connects to ImageNode class to receive images from the backend
- Connects to FunctionNode class to pass commands to the backend
- Stores the current state of the GUI and parameters
- Controls the worker threads for image processing 

Functions:
- Update_Parameters: updates the parameters of the backend
- Update_Image: updates the image displayed on the GUI
- Update_Controls: updates the values of the controls on the backend
- Stores and sets file paths for saving parameters and images

### **ImageNode(QThread):**
Description: controls the image display of the GUI

Important parameters and roles:
- Connects to CentralNode class to receive images and data from the backend
- Updates the image displayed on the GUI based on current parameters and control_index

Functions:
- Sends image signal to central node to update the image

### **FunctionNode(QObject):**
Description: controls the way that functions are called and executed (ie async, lazy, ect)

Important parameters and roles:
- Connects to CentralNode class to receive commands from the GUI
- Stores the current state of the functions and parameters
Functions:
- Execute_Function: executes the function based on the current parameters and control_index

### **Exec_Function(QThread):**
Description: executes the function in a separate thread
Important parameters and roles:
- Connects to FunctionNode class to receive commands from the GUI
- Calls Support_function.py file to execute functions based on current parameters and control_index
Functions:
- run: executes the function in a separate thread
- Update_Function: updates the function based on the current parameters and control_index
- Function wrappers: functions that wrap the actual function to be executed from Support_function.py