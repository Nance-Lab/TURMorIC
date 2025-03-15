from PyQt6.QtCore import pyqtSignal, QThread
from datetime import datetime


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

    def __init__(self, build_model, csv, entries, outpth=None,n_clusters=None):
        """
        Initializes the Worker object with the specified parameters for background processing.
        
        Args:
            build_model (bool): A flag indicating whether to build a model (`True`) or apply a model (`False`).
            csv (str): Path to the CSV file containing image set information.
            entries (dict): Dictionary containing user input fields such as output folder paths, model settings, etc.
            outpth (str, optional): Output path for saving results, defaults to `None`.
            n_clusters(int, optional): The number of clusters to use for clustering, defaults to `None`.
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

    def apply_clustering(self):
        print('')