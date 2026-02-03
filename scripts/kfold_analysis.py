import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
import numpy as np
import pandas as pd
import vampire
import logging
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from turmoric.utils import recursively_get_all_filepaths
from vampire_plotting import *


from turmoric import vampire_model

logger = logging.getLogger(__name__)


def extract_condition_from_path(file_path: str) -> str:
    """
    Extract experimental condition from file path.

    This function assumes directory structure like:
    /base_path/group/condition/file.tif

    Parameters
    ----------
    file_path : str
        Full path to image file

    Returns
    -------
    str
        Condition label (e.g., "group_condition")

    Examples
    --------
    >>> path = "/data/Blast_45_angle_5/Left_HC/image_001.tif"
    >>> extract_condition_from_path(path)
    'Blast_45_angle_5_Left_HC'
    """
    path = Path(file_path)
    # Get parent directories (condition and group)
    # Assumes structure: .../group/condition/file.tif
    condition = path.parent.name  # e.g., "Left_HC"
    group = path.parent.parent.name  # e.g., "Blast_45_angle_5"
    return f"{group}_{condition}"


def split_images_into_kfolds(
    data_path: str,
    n_folds: int = 5,
    file_extension: str = ".tif",
    random_state: int = 42,
    stratify: bool = True,
    condition_extractor: Optional[Callable[[str], str]] = None,
) -> List[Tuple[List[str], List[str]]]:
    """
    Split image files from a directory into K folds for cross-validation.

    This function discovers all image files in the specified directory (recursively),
    then splits them into K folds. When stratify=True, it ensures balanced
    representation of experimental conditions across folds.

    Parameters
    ----------
    data_path : str
        Path to directory containing image files (searched recursively)
    n_folds : int, optional
        Number of folds for cross-validation (default: 5)
    file_extension : str, optional
        File extension to search for (e.g., '.tif', '.npy') (default: '.tif')
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    stratify : bool, optional
        If True, perform stratified k-fold to balance conditions across folds (default: True)
    condition_extractor : Callable[[str], str], optional
        Function to extract condition label from file path. If None, uses
        extract_condition_from_path which assumes structure: .../group/condition/file.ext

    Returns
    -------
    fold_splits : List[Tuple[List[str], List[str]]]
        List of tuples, where each tuple contains:
        - List[str]: training file paths for this fold
        - List[str]: validation file paths for this fold
        Length of list equals n_folds.

    Raises
    ------
    FileNotFoundError
        If data_path does not exist
    ValueError
        If no files with specified extension are found, or if stratification
        is impossible due to insufficient samples per condition

    Examples
    --------
    >>> # Stratified split (recommended)
    >>> fold_splits = split_images_into_kfolds('/data/images', n_folds=5, stratify=True)
    >>> train_files, val_files = fold_splits[0]  # First fold
    >>> print(f"Fold 1: {len(train_files)} train, {len(val_files)} val")

    >>> # Non-stratified split
    >>> fold_splits = split_images_into_kfolds('/data/images', n_folds=5, stratify=False)

    >>> # Custom condition extractor
    >>> def my_extractor(path):
    ...     # Extract condition from filename instead of directory
    ...     return Path(path).stem.split('_')[0]
    >>> fold_splits = split_images_into_kfolds(
    ...     '/data/images',
    ...     condition_extractor=my_extractor
    ... )

    >>> # Iterate through all folds
    >>> for fold_idx, (train, val) in enumerate(fold_splits):
    ...     print(f"Fold {fold_idx + 1}: {len(train)} train, {len(val)} val")

    Notes
    -----
    - Files are split at the file level (not grouped by slice or sample)
    - Shuffle is enabled by default for randomization
    - Each file appears in exactly one validation set across all folds
    - Each file appears in (n_folds - 1) training sets
    - Stratified splitting ensures balanced condition representation but requires
      at least n_folds samples per condition
    """
    # Validate data path
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Get all image files
    logger.info(f"Searching for {file_extension} files in {data_path}")
    image_files = recursively_get_all_filepaths(str(data_path), file_extension)

    if len(image_files) == 0:
        raise ValueError(
            f"No files with extension {file_extension} found in {data_path}"
        )

    logger.info(f"Found {len(image_files)} files")

    # Validate n_folds
    if n_folds > len(image_files):
        raise ValueError(
            f"n_folds ({n_folds}) cannot be greater than number of files ({len(image_files)})"
        )

    # Convert to numpy array for indexing
    image_array = np.array(image_files)

    if stratify:
        # Extract conditions from file paths
        if condition_extractor is None:
            condition_extractor = extract_condition_from_path

        logger.info("Extracting experimental conditions from file paths...")
        conditions = np.array([condition_extractor(f) for f in image_files])

        # Log condition distribution
        condition_counts = Counter(conditions)
        logger.info(f"Found {len(condition_counts)} unique conditions:")
        for condition, count in sorted(condition_counts.items()):
            logger.info(f"  {condition}: {count} files")

        # Check if stratification is possible
        min_samples = min(condition_counts.values())
        if min_samples < n_folds:
            logger.warning(
                f"Minimum samples per condition ({min_samples}) is less than n_folds ({n_folds}). "
                f"Stratified splitting may not be perfectly balanced. "
                f"Consider reducing n_folds or setting stratify=False."
            )

        # Create stratified k-fold splits
        logger.info(
            f"Creating {n_folds} stratified folds with random_state={random_state}"
        )
        skfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

        fold_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            skfold.split(image_array, conditions)
        ):
            train_files = image_array[train_idx].tolist()
            val_files = image_array[val_idx].tolist()

            # Log condition distribution for this fold
            train_conditions = conditions[train_idx]
            val_conditions = conditions[val_idx]
            train_dist = Counter(train_conditions)
            val_dist = Counter(val_conditions)

            logger.info(
                f"Fold {fold_idx + 1}/{n_folds}: "
                f"{len(train_files)} train, {len(val_files)} val files"
            )
            logger.debug(f"  Train distribution: {dict(train_dist)}")
            logger.debug(f"  Val distribution: {dict(val_dist)}")

            fold_splits.append((train_files, val_files))

    else:
        # Create regular k-fold splits (no stratification)
        logger.info(
            f"Creating {n_folds} non-stratified folds with random_state={random_state}"
        )

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(image_array)):
            train_files = image_array[train_idx].tolist()
            val_files = image_array[val_idx].tolist()

            logger.info(
                f"Fold {fold_idx + 1}/{n_folds}: "
                f"{len(train_files)} train files, {len(val_files)} val files"
            )

            fold_splits.append((train_files, val_files))

    logger.info(f"Successfully created {n_folds} folds")
    return fold_splits


def train_and_predict_vampire(
    train_files: List[str],
    val_files: List[str],
    output_dir: str,
    fold_name: str = "fold",
    model_name: str = "vampire",
    num_points: int = 50,
    num_clusters: int = 5,
) -> Tuple[Path, pd.DataFrame]:
    """
    Train a VAMPIRE model on training files and predict on validation files.

    This function:
    1. Copies train/val files to temporary directories
    2. Extracts features from training images
    3. Trains a VAMPIRE model
    4. Applies the model to validation images
    5. Returns paths to model and predictions CSV

    Parameters
    ----------
    train_files : List[str]
        List of training image file paths
    val_files : List[str]
        List of validation image file paths
    output_dir : str
        Directory to save outputs (model, predictions, copied files).
        Must be a writable location, e.g., '/Users/username/output/fold_1'
        or './output/fold_1'
    fold_name : str, optional
        Name identifier for this fold (default: 'fold')
    model_name : str, optional
        Name identifier for the model (default: 'vampire')
    num_points : int, optional
        Number of points for VAMPIRE model (default: 50)
    num_clusters : int, optional
        Number of clusters for VAMPIRE model (default: 5)

    Returns
    -------
    model_path : Path
        Path to the trained model pickle file
    predictions_df : pd.DataFrame
        DataFrame containing predictions for validation set with columns:
        - filename: original file path
        - shape_mode: predicted shape mode (cluster assignment)
        - additional VAMPIRE output columns

    Raises
    ------
    FileNotFoundError
        If train_files or val_files contain non-existent paths
    OSError
        If output_dir cannot be created (e.g., read-only file system)
    Exception
        If feature extraction, model training, or prediction fails

    Examples
    --------
    >>> from pathlib import Path
    >>> fold_splits = split_images_into_kfolds('/data/images', n_folds=5)
    >>> train_files, val_files = fold_splits[0]
    >>>
    >>> # Use an absolute path in your home directory or current working directory
    >>> output_base = Path.home() / 'vampire_output'  # e.g., /Users/username/vampire_output
    >>> # or use current directory: output_base = Path.cwd() / 'vampire_output'
    >>>
    >>> model_path, predictions = train_and_predict_vampire(
    ...     train_files, val_files,
    ...     output_dir=str(output_base / 'fold_1'),
    ...     fold_name='fold_1'
    ... )
    >>> print(f"Model saved to: {model_path}")
    >>> print(f"Predictions shape: {predictions.shape}")

    Notes
    -----
    - Creates subdirectories: train/, val/, and results/ under output_dir
    - Files are copied (not moved) to preserve originals
    - VAMPIRE model and predictions are saved in output_dir
    - The function uses vampire.extraction.extract_properties() for feature extraction
    - Model training uses vampire.quickstart.fit_models()
    - Predictions use vampire.quickstart.transform_datasets()
    - Ensure output_dir is a writable location (not /output or other system directories)
    """
    import shutil
    import pandas as pd
    import vampire
    from pathlib import Path

    output_dir = Path(output_dir).resolve()

    # Try to create the directory and provide helpful error message if it fails
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create output directory '{output_dir}': {e}\n"
            f"Please use a writable location such as:\n"
            f"  - Path.home() / 'vampire_output' (e.g., /Users/username/vampire_output)\n"
            f"  - Path.cwd() / 'vampire_output' (current working directory)\n"
            f"  - An absolute path in your home directory"
        ) from e

    # Create subdirectories for train and val
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    logger.info(
        f"Processing {fold_name}: {len(train_files)} train, {len(val_files)} val files"
    )

    # Step 1: Copy training files to train directory
    logger.info("Copying training files...")
    for src_file in train_files:
        src_path = Path(src_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Training file not found: {src_file}")

        dst_path = train_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    logger.info(f"Copied {len(train_files)} training files to {train_dir}")

    # Step 2: Copy validation files to val directory
    logger.info("Copying validation files...")
    for src_file in val_files:
        src_path = Path(src_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Validation file not found: {src_file}")

        dst_path = val_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    logger.info(f"Copied {len(val_files)} validation files to {val_dir}")

    # Step 3: Extract features from training images
    logger.info("Extracting features from training images...")
    try:
        vampire.extraction.extract_properties(str(train_dir))
        logger.info("Feature extraction completed")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

    # Step 4: Train VAMPIRE model
    logger.info(f"Training VAMPIRE model: {model_name}")

    build_info_df = pd.DataFrame(
        {
            "img_set_path": [str(train_dir)],
            "output_path": [str(output_dir)],
            "model_name": [model_name],
            "num_points": [num_points],
            "num_clusters": [num_clusters],
            "num_pc": [np.nan],
        }
    )

    try:
        vampire.quickstart.fit_models(build_info_df)
        logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    # Step 5: Find the trained model file
    model_pattern = f"model_{model_name}_({num_points}_{num_clusters}_*)__.pickle"
    model_files = list(output_dir.glob(model_pattern))

    if not model_files:
        raise FileNotFoundError(
            f"No model file found matching pattern: {model_pattern}"
        )

    model_path = model_files[0]
    logger.info(f"Model saved to: {model_path}")

    # Step 6: Apply model to validation set
    logger.info("Applying model to validation images...")

    apply_info_df = pd.DataFrame(
        {
            "img_set_path": [str(val_dir)],
            "model_path": [str(model_path)],
            "output_path": [str(output_dir)],
            "img_set_name": [fold_name],
        }
    )

    try:
        vampire.quickstart.transform_datasets(apply_info_df)
        logger.info("Model application completed")
    except Exception as e:
        logger.error(f"Model application failed: {e}")
        raise

    # Step 7: Load and return predictions CSV
    # VAMPIRE typically saves predictions as: {img_set_name}_predictions.csv or similar
    # You may need to adjust this based on VAMPIRE's actual output format
    predictions_pattern = f"apply-properties_{model_name}_on_{fold_name}_*.csv"
    prediction_files = list(output_dir.glob(predictions_pattern))

    if not prediction_files:
        # Try alternative patterns
        prediction_files = list(output_dir.glob("*predictions*.csv"))

    if not prediction_files:
        logger.warning("No predictions CSV found, checking val_dir...")
        prediction_files = list(val_dir.glob("*.csv"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No predictions CSV found in {output_dir} or {val_dir}. "
            f"Searched for patterns: {predictions_pattern}, *predictions*.csv"
        )

    # Use the most recently modified CSV file
    predictions_csv = max(prediction_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading predictions from: {predictions_csv}")

    predictions_df = pd.read_csv(predictions_csv)

    # NEW - warns if cluster_id is missing
    if "cluster_id" not in predictions_df.columns:
        logger.warning(
            f"'cluster_id' column not found in predictions.Available columns: {list(predictions_df.columns)}"
        )
    else:
        logger.info(
            f"Found cluster_id column with {predictions_df['cluster_id'].nunique()} unique values"
        )

    # Add original file paths to predictions if not already present
    if "original_path" not in predictions_df.columns:
        predictions_df["fold"] = fold_name

    logger.info(f"Loaded {len(predictions_df)} predictions")

    return model_path, predictions_df

def align_vampire_models_by_contours(
    model_paths: List[str],
    reference_model_idx: int = 0
) -> Tuple[List, Dict[int, Dict[int, int]]]:
    """
    Align VAMPIRE models across folds using contour centroids (Hungarian algorithm).
    
    This function aligns cluster labels across multiple VAMPIRE models by matching
    their mean cluster contours. This is the proper way to align VAMPIRE models
    since clustering is done on contour shape space, not morphological features.
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to VAMPIRE model pickle files, one per fold
    reference_model_idx : int, optional
        Index of the model to use as reference (default: 0)
        
    Returns
    -------
    aligned_models : List[vampire.model.Vampire]
        List of loaded VAMPIRE models (note: models themselves aren't modified,
        only the mapping is returned)
    cluster_mappings : Dict[int, Dict[int, int]]
        Dictionary mapping fold_idx -> {original_cluster_id: aligned_cluster_id}
        
    Examples
    --------
    >>> model_paths = ['/output/fold_1/model.pickle', '/output/fold_2/model.pickle']
    >>> models, mappings = align_vampire_models_by_contours(model_paths)
    >>> print(f"Fold 1 mapping: {mappings[1]}")
    {0: 2, 1: 0, 2: 1, 3: 4, 4: 3}
    
    Notes
    -----
    - All models must have the same n_clusters and n_points
    - Uses Euclidean distance between mean_cluster_contours
    - Reference model's labels remain unchanged (identity mapping)
    - Apply the mappings to your predictions DataFrame to align cluster_id
    """
    from scipy.optimize import linear_sum_assignment
    
    logger.info(f"Aligning {len(model_paths)} VAMPIRE models using contour centroids...")
    
    # Load all models
    models = [load_vampire_model(path) for path in model_paths]
    
    # Validate all models have same parameters
    n_clusters = models[0].n_clusters
    n_points = models[0].n_points
    
    for i, model in enumerate(models[1:], 1):
        if model.n_clusters != n_clusters:
            raise ValueError(
                f"Model {i} has {model.n_clusters} clusters, "
                f"but reference has {n_clusters}"
            )
        if model.n_points != n_points:
            raise ValueError(
                f"Model {i} has {model.n_points} points, "
                f"but reference has {n_points}"
            )
    
    # Get reference centroids (mean cluster contours)
    reference_centroids = models[reference_model_idx].mean_cluster_contours
    logger.info(f"Using model {reference_model_idx} as reference")
    logger.debug(f"Reference centroids shape: {reference_centroids.shape}")
    
    # Compute mappings for each model
    cluster_mappings = {}
    
    for fold_idx, model in enumerate(models):
        if fold_idx == reference_model_idx:
            # Reference model has identity mapping
            mapping = {i: i for i in range(n_clusters)}
            logger.info(f"Fold {fold_idx}: Reference fold, no alignment needed")
        else:
            # Get current model's centroids
            current_centroids = model.mean_cluster_contours
            
            # Compute cost matrix (Euclidean distances between all pairs)
            cost_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    cost_matrix[i, j] = np.linalg.norm(
                        reference_centroids[i] - current_centroids[j]
                    )
            
            # Apply Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Create mapping: original cluster ID -> aligned cluster ID
            mapping = {col_ind[i]: i for i in range(n_clusters)}
            
            logger.info(f"Fold {fold_idx}: Cluster mapping: {mapping}")
            
            # Log the quality of matching (total distance)
            total_distance = cost_matrix[row_ind, col_ind].sum()
            logger.debug(f"Fold {fold_idx}: Total alignment distance: {total_distance:.4f}")
        
        cluster_mappings[fold_idx] = mapping
    
    logger.info("Model alignment completed")
    
    return models, cluster_mappings


def apply_alignment_to_predictions(
    predictions_df: pd.DataFrame,
    fold_idx: int,
    cluster_mapping: Dict[int, int]
) -> pd.DataFrame:
    """
    Apply cluster alignment mapping to predictions DataFrame.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with 'cluster_id' column
    fold_idx : int
        Fold index (for reference in output)
    cluster_mapping : Dict[int, int]
        Mapping from original to aligned cluster IDs
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aligned cluster_id and original saved
        
    Examples
    --------
    >>> aligned_df = apply_alignment_to_predictions(
    ...     predictions_df=fold_preds,
    ...     fold_idx=1,
    ...     cluster_mapping={0: 2, 1: 0, 2: 1, 3: 4, 4: 3}
    ... )
    """
    aligned_df = predictions_df.copy()
    
    # Save original cluster ID
    aligned_df['cluster_id_original'] = aligned_df['cluster_id']
    
    # Apply mapping
    aligned_df['cluster_id'] = aligned_df['cluster_id_original'].map(cluster_mapping)
    
    logger.info(
        f"Fold {fold_idx}: Aligned {len(aligned_df)} predictions "
        f"({aligned_df['cluster_id'].nunique()} unique clusters)"
    )
    
    return aligned_df


def align_kfold_predictions_with_contours(
    all_predictions: List[pd.DataFrame],
    model_paths: List[str],
    reference_fold_idx: int = 0
) -> List[pd.DataFrame]:
    """
    Align k-fold predictions using VAMPIRE contour-based alignment.
    
    This is the recommended way to align VAMPIRE k-fold results, as it uses
    the actual contour centroids that define the clusters.
    
    Parameters
    ----------
    all_predictions : List[pd.DataFrame]
        List of prediction DataFrames from k-fold CV
    model_paths : List[str]
        List of paths to corresponding VAMPIRE model files
    reference_fold_idx : int, optional
        Index of reference fold (default: 0)
        
    Returns
    -------
    aligned_predictions : List[pd.DataFrame]
        List of DataFrames with aligned cluster_id values
        
    Examples
    --------
    >>> # After k-fold training
    >>> model_paths = ['/output/fold_1/model.pickle', ...]
    >>> aligned_preds = align_kfold_predictions_with_contours(
    ...     all_predictions,
    ...     model_paths,
    ...     reference_fold_idx=0
    ... )
    >>> combined = pd.concat(aligned_preds, ignore_index=True)
    """
    # Get alignment mappings from contour centroids
    models, cluster_mappings = align_vampire_models_by_contours(
        model_paths,
        reference_model_idx=reference_fold_idx
    )
    
    # Apply mappings to predictions
    aligned_predictions = []
    
    for fold_idx, (preds_df, mapping) in enumerate(zip(all_predictions, cluster_mappings.values())):
        aligned_df = apply_alignment_to_predictions(
            preds_df,
            fold_idx,
            mapping
        )
        aligned_predictions.append(aligned_df)
    
    # Verify alignment
    logger.info("\nAlignment verification:")
    for fold_idx, aligned_df in enumerate(aligned_predictions):
        if 'cluster_id_original' in aligned_df.columns:
            logger.info(f"Fold {fold_idx}:")
            logger.info(f"  Original clusters: {sorted(aligned_df['cluster_id_original'].unique())}")
            logger.info(f"  Aligned clusters: {sorted(aligned_df['cluster_id'].unique())}")
    
    return aligned_predictions


def compare_contours_overlay(
    model_paths: List[str],
    output_path: str,
    fold_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 3),
    contour_scale: float = 3,
    colors: Optional[List] = None,
    alpha: float = 0.4,
    lw: float = 2
) -> None:
    """
    Overlay mean contours from multiple folds to show variation.
    
    This function overlays all folds' contours for each cluster in a single
    row, making it easy to see consistency/variation across folds.
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to VAMPIRE model pickle files
    output_path : str
        Path to save the plot
    fold_names : List[str], optional
        Names for each fold
    figsize : Tuple[int, int], optional
        Figure size (default: (14, 3))
    contour_scale : float, optional
        Scale of contours (default: 3)
    colors : List, optional
        Base colors for clusters. Each fold gets a shade of the base color
    alpha : float, optional
        Transparency (default: 0.4)
    lw : float, optional
        Line width (default: 2)
        
    Examples
    --------
    >>> compare_contours_overlay(model_paths, 'contours_overlay.png')
    
    Notes
    -----
    Low variation = consistent shapes across folds (good!)
    High variation = unstable clustering (may need more data or clusters)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    
    n_folds = len(model_paths)
    
    if fold_names is None:
        fold_names = [f'Fold {i+1}' for i in range(n_folds)]
    
    # Load models
    logger.info(f"Loading {n_folds} models for overlay comparison...")
    models = [load_vampire_model(path) for path in model_paths]
    
    num_clusters = models[0].n_clusters
    num_points = models[0].n_points
    
    # Create figure
    fig, axes = plt.subplots(1, num_clusters, figsize=figsize)
    if num_clusters == 1:
        axes = [axes]
    
    # Set base colors for clusters
    if colors is None:
        colors = [plt.get_cmap('twilight')(i) 
                 for i in np.linspace(0.1, 0.9, num_clusters)]
    
    # Generate fold colors (variations of base color)
    fold_color_variations = plt.cm.viridis(np.linspace(0.2, 0.8, n_folds))
    
    for cluster_id in range(num_clusters):
        ax = axes[cluster_id]
        
        # Plot all folds for this cluster
        for fold_idx, model in enumerate(models):
            x = model.mean_cluster_contours[cluster_id, :num_points]
            y = model.mean_cluster_contours[cluster_id, num_points:]
            
            # Form closed shape
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            # Center and scale
            x_centered = (x - x.mean()) * contour_scale
            y_centered = (y - y.mean()) * contour_scale
            
            # Plot with fold-specific color
            ax.plot(x_centered, y_centered,
                   color=fold_color_variations[fold_idx],
                   alpha=alpha, lw=lw,
                   label=fold_names[fold_idx])
        
        ax.set_title(f'Shape Mode {cluster_id}', fontsize=11, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        
        # Add legend only to first subplot
        if cluster_id == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    plt.suptitle('Shape Consistency Across Folds (Overlaid)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved overlay comparison to {output_path}")
    plt.close()


def compare_alignment_effect(
    model_paths: List[str],
    output_path: str,
    reference_fold_idx: int = 0,
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Compare contour overlay before and after alignment.
    
    Creates a side-by-side comparison showing how the Hungarian algorithm
    improves shape consistency across folds.
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to VAMPIRE model pickle files
    output_path : str
        Path to save the comparison plot
    reference_fold_idx : int, optional
        Index of reference fold (default: 0)
    figsize : Tuple[int, int], optional
        Figure size (default: (16, 6))
        
    Examples
    --------
    >>> compare_alignment_effect(
    ...     model_paths=model_paths,
    ...     output_path='alignment_before_after.png'
    ... )
    
    Notes
    -----
    - Left panel shows original cluster assignments (before alignment)
    - Right panel shows after applying Hungarian algorithm alignment
    - Lower variation in right panel = successful alignment
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    
    # Load models and compute alignment
    models, cluster_mappings = align_vampire_models_by_contours(
        model_paths,
        reference_model_idx=reference_fold_idx
    )
    
    n_folds = len(models)
    num_clusters = models[0].n_clusters
    num_points = models[0].n_points
    
    # Generate fold colors
    fold_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_folds))
    
    # Create figure with two rows
    fig, axes = plt.subplots(2, num_clusters, figsize=figsize)
    if num_clusters == 1:
        axes = axes.reshape(2, 1)
    
    contour_scale = 3
    alpha = 0.4
    lw = 2
    
    # Top row: BEFORE alignment (original cluster IDs)
    for cluster_id in range(num_clusters):
        ax = axes[0, cluster_id]
        
        for fold_idx, model in enumerate(models):
            x = model.mean_cluster_contours[cluster_id, :num_points]
            y = model.mean_cluster_contours[cluster_id, num_points:]
            
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            x_centered = (x - x.mean()) * contour_scale
            y_centered = (y - y.mean()) * contour_scale
            
            ax.plot(x_centered, y_centered,
                   color=fold_colors[fold_idx],
                   alpha=alpha, lw=lw,
                   label=f'Fold {fold_idx+1}' if cluster_id == 0 else None)
        
        if cluster_id == 0:
            ax.set_ylabel('Before\nAlignment', fontsize=11, fontweight='bold')
        ax.set_title(f'SM {cluster_id}', fontsize=10, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        
        if cluster_id == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Bottom row: AFTER alignment (aligned cluster IDs)
    for aligned_cluster_id in range(num_clusters):
        ax = axes[1, aligned_cluster_id]
        
        for fold_idx, model in enumerate(models):
            # Get the mapping for this fold
            mapping = cluster_mappings[fold_idx]
            
            # Find which original cluster ID maps to this aligned cluster ID
            original_cluster_id = None
            for orig, aligned in mapping.items():
                if aligned == aligned_cluster_id:
                    original_cluster_id = orig
                    break
            
            if original_cluster_id is None:
                continue
            
            x = model.mean_cluster_contours[original_cluster_id, :num_points]
            y = model.mean_cluster_contours[original_cluster_id, num_points:]
            
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            x_centered = (x - x.mean()) * contour_scale
            y_centered = (y - y.mean()) * contour_scale
            
            ax.plot(x_centered, y_centered,
                   color=fold_colors[fold_idx],
                   alpha=alpha, lw=lw)
        
        if aligned_cluster_id == 0:
            ax.set_ylabel('After\nAlignment', fontsize=11, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
    
    plt.suptitle('Effect of Contour-Based Alignment on Shape Consistency', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved alignment effect comparison to {output_path}")
    plt.close()


def compute_shape_consistency_metrics(
    model_paths: List[str],
    reference_fold_idx: int = 0
) -> pd.DataFrame:
    """
    Compute quantitative metrics for shape consistency across folds.
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to VAMPIRE model pickle files
    reference_fold_idx : int, optional
        Index of reference fold (default: 0)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with consistency metrics for each shape mode:
        - shape_mode: cluster ID
        - mean_pairwise_distance_before: avg distance before alignment
        - mean_pairwise_distance_after: avg distance after alignment
        - std_distance_before: std of distances before alignment
        - std_distance_after: std of distances after alignment
        - improvement: reduction in mean distance due to alignment
        
    Examples
    --------
    >>> metrics = compute_shape_consistency_metrics(model_paths)
    >>> print(metrics)
       shape_mode  mean_dist_before  mean_dist_after  improvement
    0           0              2.45             0.32        86.9%
    1           1              3.12             0.54        82.7%
    ...
    """
    # Load models and compute alignment
    models, cluster_mappings = align_vampire_models_by_contours(
        model_paths,
        reference_model_idx=reference_fold_idx
    )
    
    n_folds = len(models)
    num_clusters = models[0].n_clusters
    
    metrics = []
    
    for cluster_id in range(num_clusters):
        # Collect contours BEFORE alignment (using original cluster IDs)
        contours_before = []
        for model in models:
            contours_before.append(model.mean_cluster_contours[cluster_id])
        
        # Collect contours AFTER alignment (using mapped cluster IDs)
        contours_after = []
        for fold_idx, model in enumerate(models):
            mapping = cluster_mappings[fold_idx]
            # Find original cluster ID that maps to this aligned cluster ID
            original_id = [orig for orig, aligned in mapping.items() if aligned == cluster_id][0]
            contours_after.append(model.mean_cluster_contours[original_id])
        
        # Compute pairwise distances BEFORE alignment
        distances_before = []
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                dist = np.linalg.norm(contours_before[i] - contours_before[j])
                distances_before.append(dist)
        
        # Compute pairwise distances AFTER alignment
        distances_after = []
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                dist = np.linalg.norm(contours_after[i] - contours_after[j])
                distances_after.append(dist)
        
        # Compute statistics
        mean_dist_before = np.mean(distances_before)
        mean_dist_after = np.mean(distances_after)
        std_dist_before = np.std(distances_before)
        std_dist_after = np.std(distances_after)
        improvement = ((mean_dist_before - mean_dist_after) / mean_dist_before) * 100
        
        metrics.append({
            'shape_mode': cluster_id,
            'mean_pairwise_distance_before': mean_dist_before,
            'mean_pairwise_distance_after': mean_dist_after,
            'std_distance_before': std_dist_before,
            'std_distance_after': std_dist_after,
            'improvement_percent': improvement
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    logger.info("\nShape Consistency Metrics:")
    logger.info(f"\n{metrics_df.to_string(index=False)}")
    
    return metrics_df


if __name__ == "__main__":

    data_path = (
        "/Users/nelsschimek/Documents/nancelab/Data/gaby_data/tifs/converted_tiffs"
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    trainer = vampire_model.VampireModelTrainer(
        base_path=data_path,
        groups=["converted_tiffs"],
        treatments=["Acet", "Amin", "HC", "Meth"],
    )

    fold_splits = split_images_into_kfolds(data_path, n_folds=5, stratify=True)

    # Create output directory in a writable location
    output_base = Path(
        "/Users/nelsschimek/Documents/nancelab/Data/gaby_data/vampire_kfold_output"
    )

    all_predictions = []
    model_paths = []
    
    for fold_idx, (train_files, val_files) in enumerate(fold_splits):
        model_path, predictions_df = train_and_predict_vampire(
            train_files=train_files,
            val_files=val_files,
            output_dir=str(output_base / f'fold_{fold_idx+1}'),
            fold_name=f'fold_{fold_idx+1}',
            model_name='vampire_kfold',
            num_points=50,
            num_clusters=5
        )
        
        all_predictions.append(predictions_df)
        model_paths.append(str(model_path))
        
        print(f"Fold {fold_idx+1} complete: {model_path}")
    
    # Step 2: Align using CONTOUR centroids (not morphological features!)
    print(f"\n{'='*60}")
    print("Aligning shape modes using contour centroids...")
    print(f"{'='*60}")
    
    aligned_predictions = align_kfold_predictions_with_contours(
        all_predictions=all_predictions,
        model_paths=model_paths,
        reference_fold_idx=0
    )
    
    # Step 3: Combine and save
    combined_aligned = pd.concat(aligned_predictions, ignore_index=True)
    combined_unaligned = pd.concat(all_predictions, ignore_index=True)
    
    output_csv_aligned = output_base / 'all_fold_predictions_aligned_by_contours.csv'
    output_csv_unaligned = output_base / 'all_fold_predictions_unaligned.csv'
    
    combined_aligned.to_csv(output_csv_aligned, index=False)
    combined_unaligned.to_csv(output_csv_unaligned, index=False)
    
    print(f"\n✓ Aligned predictions saved to: {output_csv_aligned}")
    
    # Step 4: Visualize
    plots_dir = output_base / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Compare contours across folds
    compare_contours_across_folds(
        model_paths=model_paths,
        output_path=plots_dir / 'contours_comparison_grid.png'
    )
    
    compare_contours_overlay(
        model_paths=model_paths,
        output_path=plots_dir / 'contours_overlay.png'
    )
    
    print(f"✓ Comparison plots saved to: {plots_dir}")

    # 1. Visual comparison: before vs after alignment
    compare_alignment_effect(
        model_paths=model_paths,
        output_path=plots_dir / 'alignment_before_after.png',
        reference_fold_idx=0
    )

    # 2. Quantitative metrics
    metrics_df = compute_shape_consistency_metrics(
        model_paths=model_paths,
        reference_fold_idx=0
    )

    # Save metrics
    metrics_df.to_csv(plots_dir / 'shape_consistency_metrics.csv', index=False)

    print("\nShape Consistency Summary:")
    print(metrics_df.to_string(index=False))