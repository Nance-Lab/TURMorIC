import os
import shutil
import random
from collections import defaultdict
# from skimage.filters import try_all_threshold
# from skimage.filters import threshold_isodata
# from skimage.filters import threshold_li
# from skimage.filters import threshold_mean
# from skimage.filters import threshold_minimum
# from skimage.filters import threshold_otsu
# from skimage.filters import threshold_triangle
# from skimage.filters import threshold_yen
# from turmoric.image_process import save_npy_as_tif, nd2_to_tif
# from turmoric.cell_analysis import apply_regionprops

"""
Takes in a directory of images to separate them into training and testing data.

The image directory is passed into the function along with groups like
brain region or sex as well as treatment conditions of the slices. The
function then splits the images into a 80:20 training and testing data
without data leakage. The data is grouped by slice and treatment
conditions in new training and testing directories.

Parameters:
    base_dir: The directory of all images to be used for training and testing.
    groups: The variable groups of the images like brain region or subject sex.
    treatment_conditions: The treatment applied to the slices.

Returns:
    train_dir: The directory of training files
               from an 80:20 split from the base directory.
    test_dir: The directory of testing files
               from an 80:20 split from the base directory.
"""

# Function to organize files into training and testing folders
# without slice leakage


def organize_files_without_leakage(base_dir: str, train_dir: str, test_dir: str, groups: list,
                                   treatment_conditions: list, test_size: float=0.2) -> None:
    """
    Organize files into training and testing sets without data leakage across brain slices.

    This function walks through a directory structure organized by experimental `groups`
    and `treatment_conditions`, groups image files by brain slice ID (extracted from filenames),
    and splits slices into training and testing sets. It ensures that all files from the
    same brain slice are allocated to only one set, avoiding data leakage.

    Parameters
    ----------
    base_dir : str
        Root directory containing raw data files structured as `base_dir/group/condition/`.
    train_dir : str
        Output directory where training files will be copied.
    test_dir : str
        Output directory where testing files will be copied.
    groups : list of str
        List of experimental groups (e.g., ["control", "treated"]).
    treatment_conditions : list of str
        List of treatment condition subfolders under each group (e.g., ["vehicle", "drug"]).
    test_size : float, optional
        Proportion of slices to use for testing. Default is 0.2 (20%).

    Returns
    -------
    None
        The function performs file copying to organize training and testing sets
        but does not return a value.

    Notes
    -----
    - Slice IDs are extracted from the third underscore-separated token in each filename.
      Adjust the parsing logic if your filename structure differs.
    - Files are grouped and split at the slice level, not the file level, to prevent data leakage.
    - The train/test split is deterministic due to a fixed random seed (`random.seed(42)`).
    - Existing files in the output directories will not be overwritten.
    - Uses `shutil.copy` to duplicate files into the train/test directories.

    Examples
    --------
    >>> base_dir = "/data/brain_slices"
    >>> train_dir = "/data/split/train"
    >>> test_dir = "/data/split/test"
    >>> groups = ["control", "treated"]
    >>> treatment_conditions = ["drugA", "drugB"]
    >>> organize_files_without_leakage(base_dir, train_dir, test_dir, groups, treatment_conditions)
    processing control drugA :)
    processing control drugB :)
    processing treated drugA :)
    processing treated drugB :)
    """
    for group in groups:
        for condition in treatment_conditions:
            condition_path = os.path.join(base_dir, group, condition)
            if not os.path.exists(condition_path):
                continue

            print(f'processing {group} {condition} :)')

            # Group files by brain slice
            slice_files = defaultdict(list)
            for file in os.listdir(condition_path):
                if os.path.isfile(os.path.join(condition_path, file)):
                    # Extract slice_id based on the naming pattern
                    # Extract the third element
                    slice_id = "_".join(file.split("_")[2:3])
                    slice_files[slice_id].append(file)

            # Split slices into training and testing
            slice_ids = list(slice_files.keys())
            random.seed(42)  # For reproducibility
            random.shuffle(slice_ids)

            split_index = int(len(slice_ids) * (1 - test_size))
            train_slices = slice_ids[:split_index]
            test_slices = slice_ids[split_index:]

            # Create subdirectories for training and testing
            train_subdir = os.path.join(train_dir, group, condition)
            test_subdir = os.path.join(test_dir, group, condition)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            # Move files to the appropriate folders
            for slice_id in train_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(os.path.join(condition_path, file),
                                os.path.join(train_subdir, file))

            for slice_id in test_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(os.path.join(condition_path, file),
                                os.path.join(test_subdir, file))


def recursively_get_all_filepaths(input_folder: str, file_type: str) -> list:
    """
    Recursively retrieve all file paths of a specific type from a directory.

    This function walks through a directory tree starting from `input_folder` and collects
    the full paths of all files that end with the specified `file_type` extension.

    Parameters
    ----------
    input_folder : str
        Root directory to search for files.
    file_type : str
        File extension to filter by (e.g., '.tif', '.npy').

    Returns
    -------
    file_list : list of str
        List containing full paths to all files in `input_folder` and its subdirectories
        that match the specified file extension.

    Notes
    -----
    - The function performs a case-sensitive match on the file extension.
    - Subdirectories are traversed using `os.walk`.

    Examples
    --------
    >>> file_paths = recursively_get_all_filepaths('/path/to/folder', '.txt')
    >>> for path in file_paths:
    ...     print(path)
    /path/to/folder/file1.txt
    /path/to/folder/subfolder/file2.txt
    """
    # Walk through all files and subfolders
    file_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            # ensure only the correct file type is processed
            if file.endswith(file_type):
                # Full input path
                input_path = os.path.join(root, file)
                file_list.append(input_path)

    return file_list


"""
General-purpose file and directory utilities for TURMorIC.

Includes:
- File discovery and organisation helpers
- K-fold splitting strategies for image datasets
- Condition extraction helpers for structured directory layouts
"""

import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ── Condition extractors ──────────────────────────────────────────────────────

def extract_condition_from_path(file_path: str) -> str:
    """
    Extract a ``group_condition`` label from a two-level directory structure.

    Assumes the file lives at ``…/group/condition/filename.ext``.

    Parameters
    ----------
    file_path : str
        Full path to an image file.

    Returns
    -------
    str
        Label of the form ``"<group>_<condition>"``,
        e.g. ``"Blast_45_angle_5_Left_HC"``.

    Examples
    --------
    >>> extract_condition_from_path("/data/Blast_45_angle_5/Left_HC/img.tif")
    'Blast_45_angle_5_Left_HC'
    """
    path = Path(file_path)
    condition = path.parent.name
    group = path.parent.parent.name
    return f"{group}_{condition}"


def extract_age_region_condition(file_path: str) -> str:
    """
    Extract an ``age_region`` label for datasets organised by age and brain region.

    Assumes the file lives at ``…/age/region/filename.ext``.
    Designed for the female rat dataset where experimental variables are
    age (two levels) and brain region (seven levels).

    Parameters
    ----------
    file_path : str
        Full path to an image file.

    Returns
    -------
    str
        Label of the form ``"<age>_<region>"``,
        e.g. ``"young_cortex"`` or ``"aged_corpus_callosum"``.

    Examples
    --------
    >>> extract_age_region_condition("/data/rat/female/young/cortex/img.tif")
    'young_cortex'
    """
    path = Path(file_path)
    region = path.parent.name
    age = path.parent.parent.name
    return f"{age}_{region}"


# ── File discovery ────────────────────────────────────────────────────────────

def recursively_get_all_filepaths(input_folder: str, file_type: str) -> list:
    """
    Recursively retrieve all file paths of a specific type from a directory.

    Parameters
    ----------
    input_folder : str
        Root directory to search.
    file_type : str
        File extension to match, including the leading dot (e.g. ``'.tif'``).

    Returns
    -------
    list of str
        Full paths of all matching files found under ``input_folder``.

    Examples
    --------
    >>> paths = recursively_get_all_filepaths('/data/images', '.tif')
    >>> len(paths)
    42
    """
    file_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(file_type):
                file_list.append(os.path.join(root, file))
    return file_list


# ── Train / test split (slice-level, no leakage) ──────────────────────────────

def organize_files_without_leakage(
    base_dir: str,
    train_dir: str,
    test_dir: str,
    groups: list,
    treatment_conditions: list,
    test_size: float = 0.2,
) -> None:
    """
    Organise files into training and testing sets without data leakage across
    brain slices.

    Files from the same brain slice (identified by the third
    underscore-separated token in the filename) are kept together in either
    the train or the test set.

    Parameters
    ----------
    base_dir : str
        Root directory containing raw data structured as
        ``base_dir/group/condition/``.
    train_dir : str
        Destination directory for training files.
    test_dir : str
        Destination directory for testing files.
    groups : list of str
        Experimental group names (first directory level under ``base_dir``).
    treatment_conditions : list of str
        Treatment condition names (second directory level).
    test_size : float
        Proportion of slices to reserve for testing. Default ``0.2``.

    Notes
    -----
    Uses a fixed random seed (``42``) for reproducibility.
    """
    for group in groups:
        for condition in treatment_conditions:
            condition_path = os.path.join(base_dir, group, condition)
            if not os.path.exists(condition_path):
                continue

            print(f"processing {group} {condition} :)")

            slice_files: dict = defaultdict(list)
            for file in os.listdir(condition_path):
                if os.path.isfile(os.path.join(condition_path, file)):
                    slice_id = "_".join(file.split("_")[2:3])
                    slice_files[slice_id].append(file)

            slice_ids = list(slice_files.keys())
            random.seed(42)
            random.shuffle(slice_ids)

            split_index = int(len(slice_ids) * (1 - test_size))
            train_slices = slice_ids[:split_index]
            test_slices = slice_ids[split_index:]

            train_subdir = os.path.join(train_dir, group, condition)
            test_subdir = os.path.join(test_dir, group, condition)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)

            for slice_id in train_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(
                        os.path.join(condition_path, file),
                        os.path.join(train_subdir, file),
                    )

            for slice_id in test_slices:
                for file in slice_files[slice_id]:
                    shutil.copy(
                        os.path.join(condition_path, file),
                        os.path.join(test_subdir, file),
                    )


# ── K-fold splitting ──────────────────────────────────────────────────────────

def split_images_into_kfolds_balanced(
    data_path: str,
    n_folds: int = 5,
    file_extension: str = ".tif",
    random_state: int = 42,
    condition_extractor: Optional[Callable[[str], str]] = None,
    samples_per_condition: Optional[int] = None,
) -> Tuple[List[Tuple[List[str], List[str]]], Dict[str, int]]:
    """
    Split images into K folds with exactly equal samples per condition.

    Ensures perfectly balanced representation of experimental conditions
    across folds by randomly sub-sampling each condition to the same size.
    Files excluded from the sub-sample are reported in ``usage_stats``.

    This is the recommended splitting function for cross-species or
    multi-region analyses where equal representation is scientifically
    important.

    Parameters
    ----------
    data_path : str
        Root directory containing image files (searched recursively).
    n_folds : int
        Number of cross-validation folds. Default ``5``.
    file_extension : str
        File extension to search for. Default ``'.tif'``.
    random_state : int
        Random seed for reproducibility. Default ``42``.
    condition_extractor : callable, optional
        Function ``(file_path: str) -> str`` that returns a condition label.
        If ``None``, uses :func:`extract_age_region_condition`, which assumes
        a ``…/age/region/file.ext`` directory structure.

        For other layouts supply a custom extractor, e.g.::

            def my_extractor(path):
                return Path(path).parent.parent.name   # two levels up

    samples_per_condition : int, optional
        Total number of samples to use per condition across all folds.
        Must be divisible by ``n_folds``.
        If ``None``, automatically set to
        ``(min_samples_per_condition // n_folds) * n_folds`` to maximise
        data usage while maintaining balance.

    Returns
    -------
    fold_splits : list of tuple
        ``[(train_files, val_files), ...]`` — one tuple per fold.
    usage_stats : dict
        Summary statistics:

        * ``total_files`` — total files discovered
        * ``files_used`` — files included after balancing
        * ``files_excluded`` — files dropped for balance
        * ``samples_per_condition_per_fold`` — samples per condition per fold
        * ``conditions`` — number of unique conditions
        * ``samples_per_condition_total`` — total per condition across folds

    Raises
    ------
    FileNotFoundError
        If ``data_path`` does not exist.
    ValueError
        If no files are found, if ``n_folds`` exceeds file count, if any
        condition has fewer samples than ``n_folds``, or if
        ``samples_per_condition`` is not divisible by ``n_folds``.

    Examples
    --------
    >>> # Female rat data: age × region stratification
    >>> fold_splits, stats = split_images_into_kfolds_balanced(
    ...     data_path='/data/rat/female',
    ...     n_folds=5,
    ...     condition_extractor=extract_age_region_condition,
    ... )
    >>> print(f"Using {stats['files_used']}/{stats['total_files']} files")
    >>> train_files, val_files = fold_splits[0]

    Notes
    -----
    - Splitting is performed at the file level (not at the brain-slice level).
      If slice-level splitting is required use
      :func:`organize_files_without_leakage` instead.
    - Each condition contributes exactly ``samples_per_condition_per_fold``
      files to every validation fold.
    - The split is fully deterministic given ``random_state``.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    logger.info(f"Searching for {file_extension} files in {data_path}")
    image_files = recursively_get_all_filepaths(str(data_path), file_extension)

    if not image_files:
        raise ValueError(
            f"No files with extension '{file_extension}' found in {data_path}"
        )
    logger.info(f"Found {len(image_files)} total files")

    if n_folds > len(image_files):
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed number of files ({len(image_files)})"
        )

    # Extract conditions
    if condition_extractor is None:
        condition_extractor = extract_age_region_condition

    logger.info("Extracting conditions from file paths...")
    conditions = np.array([condition_extractor(f) for f in image_files])
    image_array = np.array(image_files)

    condition_counts = Counter(conditions)
    logger.info(f"Found {len(condition_counts)} unique conditions:")
    for cond, count in sorted(condition_counts.items()):
        logger.info(f"  {cond}: {count} files")

    min_samples = min(condition_counts.values())
    if min_samples < n_folds:
        raise ValueError(
            f"Condition with fewest samples has {min_samples} files, "
            f"but n_folds={n_folds}. "
            f"Reduce n_folds or merge sparse conditions."
        )

    # Determine samples per condition
    if samples_per_condition is None:
        samples_per_fold = min_samples // n_folds
        samples_per_condition = samples_per_fold * n_folds
        logger.info(
            f"Auto-selected: {samples_per_condition} samples/condition "
            f"({samples_per_fold}/fold)"
        )
    else:
        if samples_per_condition % n_folds != 0:
            raise ValueError(
                f"samples_per_condition ({samples_per_condition}) must be "
                f"divisible by n_folds ({n_folds})"
            )
        samples_per_fold = samples_per_condition // n_folds
        if min_samples < samples_per_condition:
            raise ValueError(
                f"Requested {samples_per_condition} samples/condition but "
                f"minimum available is {min_samples}"
            )
        logger.info(
            f"Using {samples_per_condition} samples/condition "
            f"({samples_per_fold}/fold)"
        )

    # Sub-sample each condition to equal size
    rng = np.random.default_rng(random_state)
    balanced_files: List[str] = []
    balanced_conditions: List[str] = []

    for condition in sorted(condition_counts.keys()):
        mask = conditions == condition
        cond_files = image_array[mask]

        chosen = rng.choice(len(cond_files), size=samples_per_condition, replace=False)
        balanced_files.extend(cond_files[chosen])
        balanced_conditions.extend([condition] * samples_per_condition)

        excluded = len(cond_files) - samples_per_condition
        logger.info(
            f"  {condition}: using {samples_per_condition} "
            f"of {len(cond_files)} ({excluded} excluded)"
        )

    balanced_files_arr = np.array(balanced_files)
    balanced_conditions_arr = np.array(balanced_conditions)

    # Stratified k-fold
    skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_splits: List[Tuple[List[str], List[str]]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skfold.split(balanced_files_arr, balanced_conditions_arr)
    ):
        train_files = balanced_files_arr[train_idx].tolist()
        val_files = balanced_files_arr[val_idx].tolist()

        val_dist = Counter(balanced_conditions_arr[val_idx])
        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"{len(train_files)} train, {len(val_files)} val"
        )
        logger.debug(f"  Val distribution: {dict(val_dist)}")

        fold_splits.append((train_files, val_files))

    total_files = len(image_files)
    files_used = len(balanced_files)
    usage_stats: Dict[str, int] = {
        "total_files": total_files,
        "files_used": files_used,
        "files_excluded": total_files - files_used,
        "samples_per_condition_per_fold": samples_per_fold,
        "conditions": len(condition_counts),
        "samples_per_condition_total": samples_per_condition,
    }

    logger.info("=" * 60)
    logger.info("Data usage summary")
    logger.info(f"  Total files       : {total_files}")
    logger.info(
        f"  Files used        : {files_used} ({files_used / total_files * 100:.1f}%)"
    )
    logger.info(f"  Files excluded    : {total_files - files_used}")
    logger.info(f"  Conditions        : {len(condition_counts)}")
    logger.info(f"  Samples/condition : {samples_per_condition} total, {samples_per_fold}/fold")
    logger.info("=" * 60)

    return fold_splits, usage_stats