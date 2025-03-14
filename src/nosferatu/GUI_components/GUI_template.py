import sys
from PyQt6.QtWidgets import (QApplication, QWidget,
        QVBoxLayout, QPushButton, QLabel, QLineEdit,
          QProgressBar, QFileDialog, QHBoxLayout, 
          QGridLayout, QStackedWidget,QIntValidator)
from PyQt6.QtCore import Qt, pyqtSignal,pyqtSlot, QThread, QObject
from PyQt6.QtGui import QImage, QPixmap
from datetime import datetime
import pickle
/home/hwood/Planning/scr/GUI_components/GUI_template

class CentralNode(QObject):
    """
    The CentralNode is responsible for facilitating communication between the main window and QThread-based worker classes.

    This class sends information to the worker threads, and receives updates from them to forward back to the main window.
    It serves as an intermediary between the GUI and the backend processing threads.

    Attributes:
    """
    def __init__(self):
        super(CentralNode, self).__init__()
        self.controller_thread = QThread()
        self.controller = ThreadController()
        self.controller.moveToThread(self.controller_thread)
        self.controller_thread.start()
        self.main_window = MainWindow()
        self.current_control=0 # current set of controls
        self.stuff = None
        
        # Instantiate workers
        self.image_handler = ImageHandler("path/to/image.jpg")
        self.build_model_worker = BuildModel(data="some data")

        # Connections to MainWindow
        self.image_handler.update_image.connect(self.main_window.displayImage)
        self.build_model_worker.progress_changed.connect(self.main_window.update_progress)
        self.build_model_worker.status_updated.connect(self.main_window.update_status)

    def update_controls(self):
        self.current_control+=1

    def send_data_to_worker(self, data):
        """
        Sends data to worker thread(s) for processing.

        Args:
            data (any): Data to be sent to the worker thread.
        """
        pass

    def receive_data_from_worker(self, data):
        """
        Receives processed data from worker thread(s) and forwards it to the main window.

        Args:
            data (any): Processed data received from the worker thread.
        """
        pass

    def update_status(self, new_status):
        self.update_status_signal.emit(new_status)

    def start_processing(self, process):
        """
        Start a process.
        """
        self.controller.start_worker(process)

    def stop_process(self, process):
        """
        Stop a process.
        """
        self.controller.stop_worker(process)

class ThreadController(QObject):
    """
    The ThreadController manages commands from the CentralNode to start or stop QThread worker classes.

    This class is designed to start and stop worker threads based on requests from the CentralNode.
    It ensures that tasks are run in the background without blocking the main GUI thread.

    Attributes:
        None
    """
    def __init__(self):
        super().__init__()

    def start_worker(self, worker):
        """
        Starts a given worker thread.

        Args:
            worker (QThread): The worker thread to be started.
        """
        worker.start()

    def stop_worker(self, worker):
        """
        Stops a given worker thread.

        Args:
            worker (QThread): The worker thread to be stopped.
        """
        worker.requestInterruption()
        worker.wait()

class ImageHandler(QThread):
    """
    Takes in parameters from CentralNode, modifies them, and sends them to the main window for display.

    This class performs image manipulation tasks in a separate thread. It receives parameters from the CentralNode, 
    processes images, and sends the results back to the main window to be displayed.

    Signals:
        update_image (QImage): Emitted to send the processed image back to the main window for display.
    
    Attributes:
        image_data (QImage): The image data to be processed.
        modify_type (str): The type of modification (e.g., "rotate", "filter", etc.).
    """
    update_image = pyqtSignal(QImage)

    def __init__(self, image_data, modify_type):
        super().__init__()
        self.image_data = image_data
        self.modify_type = modify_type

    def run(self):
        """
        Runs the image manipulation process in a separate thread.
        Based on the `modify_type`, it performs different image transformations.
        """
        if self.modify_type == "rotate":
            self.process_rotate()
        elif self.modify_type == "filter":
            self.process_filter()
        # Add more processing types as needed

    def process_filter(self): # Sample function to be replaced with actual filters
        """
        Applies a filter to the image (e.g., grayscale or blur).
        """
        # Example: apply a grayscale filter
        gray_image = self.image_data.convertToFormat(QImage.Format_Grayscale8)
        self.update_image.emit(gray_image)

class BuildModel(QThread):
    """
    Worker class for performing background tasks in a separate thread.
    This class inherits from `QThread` and is designed to handle long-running operations
    (such as building or applying a model) without blocking the main GUI thread.

    The worker communicates progress and status updates via the `progress_changed` and
    `status_updated` signals.

    Signals:
        progress_changed (int): Emitted to indicate progress (percentage) of the task.
        status_updated (str): Emitted to update the status message during task execution.

    Attributes:
        build_model (bool): Flag indicating whether the worker is building a model (True) 
                             or applying a model (False).
        csv (str): Path to the CSV file containing image set information.
        entries (dict): Dictionary containing user input fields, such as output paths and model settings.
        outpth (str, optional): Output path for saving results, defaults to `None`.
        clnum (int, optional): The number of clusters for clustering the data, defaults to `None`.
    """
    progress_changed = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self, build_model, csv, entries, outpth=None, clnum=None):
        """
        Initializes the Worker object with the specified parameters for background processing.
        
        Args:
            build_model (bool): A flag indicating whether to build a model (`True`) or apply a model (`False`).
            csv (str): Path to the CSV file containing image set information.
            entries (dict): Dictionary containing user input fields such as output folder paths, model settings, etc.
            outpth (str, optional): Output path for saving results, defaults to `None`.
            clnum (int, optional): The number of clusters to use for clustering, defaults to `None`.
        """
        super().__init__()
        self.build_model = build_model
        self.csv = csv
        self.entries = entries
        self.outpth = outpth
        self.clnum = clnum

    def run(self):
        """
        Runs the model-building or applying process in a separate thread.
        Depending on the value of `build_model`, it either builds or applies the model.
        """
        self.mainbody()

    def mainbody(self):
        """
        Core logic for the model-building or applying process.
        Updates progress and status during the process.
        """
        progress = 50
        experimental = True
        realtimedate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        N = int(self.entries['Number of coordinates'].text())

        if self.build_model:
            self.status_updated.emit("Modeling initiated...")
            self.progress_changed.emit(progress + 15)
            progress += 20
            self.progress_changed.emit(progress)
            progress += 25
            self.progress_changed.emit(progress)
            self.status_updated.emit('Modeling completed.')
        else:
            self.status_updated.emit("Applying model...")
            progress += 100
            self.progress_changed.emit(progress)
            self.status_updated.emit("Model applied successfully.")

class MainWindow(QWidget):
    """
    The main window of the PyQt6 GUI application.

    This class sets up the main user interface (UI) for the application, including the layout and controls.
    It handles user interaction and connects buttons to specific functionalities such as loading CSV files, 
    selecting output folders, and building models. It also manages the progress bar and status label.

    Attributes:
        number_of_pages (int): The number of pages in the GUI (used for stacked widgets).
        current_control (int): The index of the current page being displayed in the GUI.
        controlStack (QStackedWidget): Stack widget used to manage different pages.
        Image_Display1 (QLabel): Label for displaying images.
        load_csv_button (QPushButton): Button to load a CSV file.
        select_folder_button (QPushButton): Button to select an output folder.
        build_model_button (QPushButton): Button to trigger model building.
        progress_bar (QProgressBar): Progress bar to indicate task progress.
        status_label (QLabel): Label to display the current status message.
        next_button (QPushButton): Button to navigate to the next page when the task is complete.
    """

    def __init__(self):
        """
        Initializes the MainWindow and sets up the UI.

        This method sets the window title, size, and initializes the layout for the GUI.
        It also creates a stacked widget to manage multiple pages and prepares the initial layout.

        It does not take any parameters and sets up default values for essential attributes such as:
            - `number_of_pages`: The number of pages in the GUI.
            - `current_control`: The current page index (default is 0).
            - `controlStack`: A stacked widget to manage different pages.
        """
        super().__init__()
        self.setWindowTitle("PyQt6 GUI Template")
        self.setGeometry(0, 0, 1000, 800)
        self.number_of_pages = 3  # Number of pages in the GUI 
        self.current_control = 0  # Current page of the GUI
        self.controlStack = QStackedWidget(self)
        self.default_values={'n_clusters':10}
        # initializing empty values
        self.Central_Connection=None, self.Image
        self.init_ui()

    def init_ui(self):
        # Main Layout Setup
        outerLayout = QGridLayout()
        displayLayout = QHBoxLayout()
        sidebuttonLayout = QVBoxLayout()

        # Universal widgets
        ################### 

        # Displays images
        self.Image_Display1 = QLabel(self)
        self.Image_Display1.setFixedSize(640, 480)
        displayLayout.addWidget(self.Image_Display1)

        # Page navigation Button
        self.next_button = self.make_button(self,'Next Page', self.update_control_stack)
        self.next_button.setVisible(False)

        
        # Page specific widgets
        #######################
        self.page1 = QWidget()
        self.page2 = QWidget()
        self.page3 = QWidget()  

        # Add pages to Navigation 
        self.controlStack.addWidget(self.page1)
        self.controlStack.addWidget(self.page2)
        self.controlStack.addWidget(self.page3)

        # Initiate page layouts
        self.page1_layout = QVBoxLayout()
        self.page2_layout = QVBoxLayout()
        self.page3_layout = QVBoxLayout()

        # page1
        self.page1_layout.addWidget(QLabel("Page 1 "))
        self.load_csv_button = self.make_button(self,'Load CSV', self.load_csv)
        self.select_folder_button = self.make_button(self, 'Select Folder', self.select_folder)
        sidebuttonLayout.addWidget(self.select_folder_button)


        self.build_model_button = self.make_button(self,'Build Model', self.build_model)
        sidebuttonLayout.addWidget(self.build_model_button)
        sidebuttonLayout.addWidget(self.load_csv_button)  
        self.page1_layout.addWidget(self.next_button)
        self.page1.setLayout(self.page1_layout)

        # page2
        self.page2_layout.addWidget(QLabel("Page 2 Content Here"))
        self.page2.setLayout(self.page2_layout)

        # page 3: Model Building
        self.clusters_input = QLineEdit(self)
        self.clusters_input.setPlaceholderText("Enter Number of Clusters")
        self.clusters_input.setValidator(QIntValidator(self))  # Only allows integer input
        self.clusters_input.setText(self.default_values["n_clusters"]) 
        self.clusters_input.setPlaceholderText("Enter Number of Clusters")
        self.clusters_input.setValidator(QIntValidator(self))

        # Adds layouts to the main layout
        outerLayout.addWidget(self.controlStack, 0, 0)
        outerLayout.addLayout(displayLayout, 0, 0)
        outerLayout.addLayout(sidebuttonLayout, 0, 1)
        self.setLayout(outerLayout)

        # Initializes connection to CentralNode
        self.Central_Connection= CentralNode()
        self.Central_Connection.updateImage.connect(lambda image: self.displayImage(image))

    # UI helper functions
    #####################
    def update_control_stack(self):
        """Updates current set of pages and controls"""
        self.current_control+=1

    def make_button(self,name, action):
        button = QPushButton(name, self)
        button.clicked.connect(action)
        return button

    def progress_bar(self, Layout):
         # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        Layout.addWidget(self.progress_bar)
        self.status_label = QLabel('Status: Ready', self)
        Layout.addWidget(self.status_label)

    def image_display_visable(self):
        if self.Image_Display1.isVisible:
            self.Image_Display1.setVisible(False)
        else:
            self.Image_Display1.setVisible(True)

    # Functions that communicate with other classes
    ###############################################
    @ pyqtSlot(QImage)
    def displayImage(self, Image):
        self.Image_Display1.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def load_csv(self):
        """Initializes file selection window
        """
        file, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file:
            self.entries['Image sets to build'].setText(file)

    @ pyqtSlot()
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.entries['Model output folder'].setText(folder)
    @ pyqtSlot()
    def update_controls(self):
        self.Central_Connection.update_controls()

    @ pyqtSlot()
    def build_model(self):
        """ To do: add communication to central node"""
        build_model = True
        csv = self.entries['Image sets to build'].text()
        entries = self.entries
        outpth = self.entries['Model output folder'].text() 
        self.worker = build_model
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.start()

    @ pyqtSlot()
    def update_progress(self, progress):
        """
        Updates the progress bar with the current progress value.

        This method receives the current progress value (an integer) from the worker thread and updates the progress bar
        accordingly.

        Args:
            progress (int): The current progress value (0 to 100).
        """
        self.progress_bar.setValue(progress)

    @ pyqtSlot()
    def update_status(self, status):
        """
        Updates the status label with a new status message.

        This method receives a status message from the worker thread and updates the status label to reflect the current 
        state of the process (e.g., "Modeling initiated", "Modeling completed").

        Args:
            status (str): The new status message to display."""
        self.status_label.setText(f"Status: {status}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
