from PyQt6.QtCore import Qt, pyqtSignal,pyqtSlot, QThread, QObject
from PyQt6.QtGui import QImage, QPixmap

import sys
import pickle

import ImageHandler
import FunctionHandler
import ModelHandler


class CentralNode(QObject):
    """
    The CentralNode is responsible for facilitating communication between the main window and QThread-based worker classes.

    This class sends information to the worker threads, and receives updates from them to forward back to the main window.
    It serves as an intermediary between the GUI and the backend processing threads.

    Attributes:
    """
    def __init__(self, main_window):
        super(CentralNode, self).__init__()
        self.main_window = main_window
        self.current_control=0 # current set of controls
        self.stuff = None
        
        # Instantiate workers
        self.image_handler = ImageHandler(data="some data")
        self.function_handler=FunctionHandler(data="some data")
        self.model_handler = ModelHandler(data="some data")

        # Connections to MainWindow
        self.function_handler.update_image.connect()
        self.image_handler.update_image.connect(self.main_window.displayImage)
        self.function_handler.update_data.connect()
        self.build_model.progress_changed.connect(self.main_window.update_progress)
        self.build_model.status_updated.connect(self.main_window.update_status)
        self.function_handler.update_model.connect()

    def update_controls(self):
        self.current_control+=1

    def update_status(self, new_status):
        self.update_status_signal.emit(new_status)

    def start_processing(self, process, **param):
        """
        Start a process.
        """
        process.start(process)

    def stop_process(self, process):
        """
        Stop a process.
        """
        process.requestInterruption()
        process.wait()