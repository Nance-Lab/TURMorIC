# GUI
### **What it does**  
The **Interactive GUI Component** provides a **graphical interface** that allows users to visualize, adjust, and process images **without coding expertise**. This component is essential for **Maya, Jaden, and Ethan**, who prefer an intuitive, step-by-step workflow.
[More details on GUI implementation here](gui_details.md)

### **Inputs**  
- **Uploaded raw or preprocessed images**.  
- **Segmentation overlays** from the **Segmentation & Quantification Component**.  
- **User inputs** (sliders, dropdowns, buttons) for fine-tuning thresholding and segmentation settings.  

### **Outputs**  
- **Real-time visualization** of segmentation overlays.  
- **Exported user-defined settings**, enabling reproducibility.  
- **Feedback alerts**, notifying users of potential image quality or segmentation issues.  

### **How it uses other components**  
- Works with **Adaptive Thresholding Component** to interactively adjust segmentation.  
- Allows real-time refinement of **Segmentation & Quantification Component** outputs.  

### **Side effects**  
- **User bias may affect segmentation accuracy** when manually adjusting settings.  
- **Real-time image processing can be slow** for large datasets, requiring optimization.  

# **Image Preprocessing Component**  

### **What it does**  
The **Image Preprocessing Component** prepares raw microscopy images for segmentation and analysis. It ensures consistency in image format, removes background noise, adjusts contrast, and normalizes variations in staining intensity to facilitate accurate segmentation. Since users like **Maya and Jaden** have limited experience with image processing, this component must function with minimal user input while providing automated enhancements.

### ** Important Parameters and Roles**
- Splits segmented microglia images into four quadrants for each image in a specified folder.
- Selects training and testing datasets, organizing images into directories for model building and testing.
- Colors segmented images based on shape modes with custom colors. 
- Handles image segmentation and labeling using shape modes from CSV.
- Automates labeling for multiple images based on provided file paths.

### **Inputs**  
- A **folder** containing **raw microscopy images** in **.nd2** format or **a single .nd2 image**.   
- **Metadata**, including **resolution, stain type, exposure time, and microscope settings**, to be factored into preprocessing.  
- **User-defined parameters** (optional), such as **contrast adjustment levels, noise reduction settings, and intensity normalization preferences**.  

### **Outputs**  
- **Preprocessed images** optimized for segmentation, saved in a designated output directory.  
- **Log file** documenting preprocessing steps and identifying any issues (e.g., overexposure, poor contrast).  
- **CSV file** to store model-building and application information.
- **Visual comparisons** allowing users to assess preprocessing improvements before segmentation.  

### **How it uses other components**  
- Passes enhanced images to the **Segmentation & Quantification Component** for further processing.  
- Interacts with the **Adaptive Thresholding Component** to refine intensity adjustments dynamically.  
- **Integrates with the Interactive GUI**, providing users with previews and manual adjustment options.  

### **Side effects**  
- **Potential alteration of cellular structures** if filters are applied too aggressively.  
- **Increased storage usage** due to the creation of multiple image versions.  
- **Over-processing risk**, possibly introducing artificial enhancements that could skew analysis results.
- **Renames and organizes images** for VAMPIRE analysis according to a specific naming convention.
- **Saves the recolored images** at specified output paths.
- **Sends colored image files** to the GUI


## **Adaptive Thresholding Component**  

### **What it does**  
The **Adaptive Thresholding Component** dynamically adjusts segmentation parameters based on staining intensity variations. This is particularly useful for **Jaden**, who experiences inconsistent thresholding due to variations in staining quality.

### **Important Parameters and Roles**
- Applies multiple thresholding methods (Li, Otsu, Mean, etc.) to images.
- Uses a custom function for folder filtering and prepares image lists for threshold testing.
- Saves the thresholded images and moves them to a dedicated folder.
- Performs analysis to determine the optimal threshold (Li threshold) for microglia identification based on visual inspection.
- Removes small objects below a size threshold (based on microglia size) using Sci-kit's morphology functions.

### **Inputs**  
- **Image data** from the **Image Preprocessing Component**.  
- **User-defined or auto-calculated threshold values**, affecting segmentation sensitivity.  
- **Reference datasets** (if available) containing **annotated images** to enhance thresholding accuracy.  

### **Outputs**  
- **Optimized segmentation masks**, ensuring consistency across images.  
- **Suggested threshold values**, which users can refine manually.  
- **Heatmaps visualizing staining intensity variations**, aiding in sample consistency analysis.  

### **How it uses other components**  
- Works with the **Segmentation & Quantification Component** to enhance segmentation accuracy.  
- Provides **real-time feedback** in the **Interactive GUI**, enabling users to fine-tune thresholding.  
- **Applies to multiple images** using the **Batch Processing Component**, ensuring uniform processing.  

### **Side effects**  
- **Overcorrection risks**, where weakly stained structures may be misclassified as background.  
- **Automatically generated values may not always align with user expectations**, requiring manual validation.  

## Segmentation & Quantification Component**  

### **What it does**  
The **Segmentation & Quantification Component** detects and outlines **microglia and glial cells** within images, extracting morphological features such as **cell shape, size, clustering behavior, and complexity**. This component ensures high segmentation accuracy while remaining user-friendly, especially for **Maya, Amina, and Ethan**, who require automated processing.

### **Important Parameters and Roles**
- Applies image segmentation using skimage.measure.label and calculates multiple region properties (area, centroid, eccentricity, etc.) for each labeled region.
- Calculates additional features like circularity and aspect ratio for microglia cells.

### **Inputs**  
- **Preprocessed images** from the **Image Preprocessing Component**.  
- **Segmentation parameters** from the **Adaptive Thresholding Component**
- **Ground truth data** (if available) for validation and segmentation accuracy improvements.  

### **Outputs**  
- **Segmented images** in **.tiff** with labeled cellular structures.  
- **.csv files containing quantitative datasets**, including:  
  - **Cell count**  
  - **Shape descriptors** (circularity, elongation, convexity)  
  - **Clustering and spatial distribution statistics**  
- **Segmentation overlays** that allow users to visually assess analysis results.  

### **How it uses other components**  
- Works with **Image Preprocessing Component** to ensure quality segmentation.  
- Uses **Adaptive Thresholding Component** recommendations to refine segmentation.  
- Passes extracted features to the **Data Export API** for storage and computational analysis.  
- **Integrates with the Interactive GUI**, enabling real-time parameter adjustments.  

### **Side effects**  
- **Incorrect threshold settings** may result in over-segmentation or under-segmentation.  
- **Processing large images may cause computational delays**, necessitating GPU acceleration.  
- **User modifications in the GUI** could introduce segmentation variability, affecting reproducibility.

## **Batch Processing Component**  

### **What it does**  
The **Batch Processing Component** enables researchers to process multiple images simultaneously, ensuring consistency across experiments. This is essential for **Amina and Olivia**, who require large-scale data analysis.

### **Inputs**  
- **Folder of images** requiring batch segmentation.  
- **Predefined segmentation and thresholding parameters** for consistency.  

### **Outputs**  
- **Batch-processed segmented images** stored in a structured directory.  
- **Statistical summaries** for large-scale comparative studies.  

### **How it uses other components**  
- Uses **Segmentation & Quantification Component** to apply segmentation across multiple images.  
- Works with **Data Export API** for structured results storage. 

## **Data Export API Component**  

### **What it does**  
The **Data Export API Component** saves analysis results in structured formats (**CSV, JSON**) for computational analysis and machine learning applications. This is essential for **Olivia**, who needs structured data for predictive modeling.

### **Inputs**  
- **Extracted morphological features** from the **Segmentation & Quantification Component**.  
- **User-defined export preferences** (format, fields to include).  

### **Outputs**  
- **Structured datasets** (CSV, JSON) ready for statistical or machine learning analysis. 

### **How it uses other components**  
- Works with **Batch Processing Component** to export large datasets efficiently.  
- **Supports integration with machine learning pipelines** for advanced analysis.  

### **Side effects (moar deets)**  
- **Outputs are formatted correctly** to allow for additional processing.
- **Incorrectly formatted outputs** may require additional processing.
