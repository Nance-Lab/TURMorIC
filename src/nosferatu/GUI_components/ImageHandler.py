from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap


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