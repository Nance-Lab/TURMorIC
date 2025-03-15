
from PyQt6.QtCore import pyqtSignal, QThread

class FunctionHandler(QThread):
    def __init__(self, FunctionHandler):
        """ doc string"""
        super().__init__()

    def run(self):
        print("executing function")