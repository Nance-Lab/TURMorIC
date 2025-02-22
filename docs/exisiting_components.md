Current components of codebase:

vampire.py:
- Description: Launches the interface for the VAMPIRE GUI.
- Important Parameters and Roles:
    - Sets up the graphical user interface for the VAMPIRE tool.
    - Implements functions for user input and interacts with other modules (e.g., loading CSV files, selecting output folders).
    - Controls the workflow through functions such as Model, getcsv, and getdir.

---
mainbody.py:
- Description: Handles the core VAMPIRE analysis process, including:
    - Registration: Aligns boundaries to remove rotational variance.
    - PCA (Principal Component Analysis): Reduces data dimensions for better analysis.
    - Clustering: Groups data based on similarities.
- Functions:
    - mainbody: Executes the complete pipeline based on model building or application.
    - collect_seleced_bstack: Collects and prepares the boundary stack for further processing.
---
collect_selected_bstack.py:
- Description: Reads the boundaries of cells or nuclei from a CSV file and prepares them for analysis.
- Functions:
    - collect_seleced_bstack: Gathers boundary data from specified locations (either to build a model or apply an existing one).
---
bdreg.py:
- Description: Registers the boundaries of cells or nuclei, ensuring alignment across images.
- Functions:
    - bdreg: Registers the boundaries by resampling and applying Singular Value Decomposition (SVD).
________________________________________
pca_bdreg.py:
- Description: Applies PCA to registered boundaries to reduce dimensions and capture the most relevant features.
- Functions:
    - pca_bdreg: Handles the PCA transformation of boundary data.
________________________________________
PCA_custom.py:
- Description: Custom implementation of PCA to process boundary data.
- Functions:
    - PCA_custom: Returns the transformed data and regenerates the original data from the PCA transformation.
________________________________________
clusterSM.py:
- Description: Applies clustering (K-means) to boundary data after PCA processing to categorize cells or nuclei.
- Functions:
    - clusterSM: Performs K-means clustering and assigns cluster labels to each cell or nucleus.
________________________________________
update_csv.py:
- Description: Generates and updates the VAMPIRE datasheet with cluster labels after analysis.
- Functions:
    - update_csv: Writes the results, including cluster assignments, into a CSV file.
________________________________________
Helper Functions:
- getdir: Opens a file dialog to choose a directory.
- getcsv: Opens a file dialog to select a CSV file.
- Model: Handles model building or application, depending on the user’s choice.
- vampire(): Initializes and runs the GUI loop.