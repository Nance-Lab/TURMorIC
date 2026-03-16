"""
VAMPIRE model training, alignment, and cross-validation utilities.

This module provides:
- VampireModelTrainer: class-based interface for single train/apply workflows
- train_and_predict_vampire: functional interface for one k-fold iteration
- align_vampire_models_by_contours: Hungarian-algorithm alignment using contour centroids
- apply_alignment_to_predictions: apply a fold's cluster mapping to its predictions DataFrame
- align_kfold_predictions_with_contours: end-to-end alignment across all folds
- compute_shape_consistency_metrics: quantitative before/after alignment metrics
- compare_contours_overlay: overlay mean contours across folds per shape mode
- compare_alignment_effect: side-by-side before/after alignment visualisation
"""

import logging
import pickle
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vampire
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_vampire_model(model_path: str):
    """
    Load a VAMPIRE model from a pickle file.

    Parameters
    ----------
    model_path : str
        Path to the model ``.pickle`` file produced by
        ``vampire.quickstart.fit_models``.

    Returns
    -------
    model
        Deserialised VAMPIRE model object.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _validate_model_contours(model, model_label: str) -> np.ndarray:
    """
    Return ``mean_cluster_contours`` from a VAMPIRE model, raising a clear
    error if the attribute is missing or malformed.

    Parameters
    ----------
    model : vampire model object
    model_label : str
        Human-readable identifier used in error messages.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_clusters, n_points * 2)``.

    Raises
    ------
    AttributeError
        If the model does not expose ``mean_cluster_contours``.
    ValueError
        If the array has an unexpected shape.
    """
    if not hasattr(model, "mean_cluster_contours"):
        available = [a for a in dir(model) if not a.startswith("_")]
        raise AttributeError(
            f"Model '{model_label}' has no attribute 'mean_cluster_contours'.\n"
            f"Available attributes: {available}\n"
            f"Check the VAMPIRE version or pickle file integrity."
        )
    contours = model.mean_cluster_contours
    if contours.ndim != 2:
        raise ValueError(
            f"Model '{model_label}': mean_cluster_contours has unexpected shape "
            f"{contours.shape}. Expected (n_clusters, n_points * 2)."
        )
    return contours


# ── Class-based interface (single train/apply) ────────────────────────────────

class VampireModelTrainer:
    """
    Class-based interface for training and applying a single VAMPIRE model.

    Intended for workflows where a fixed train/test split already exists on
    disk. For k-fold cross-validation use :func:`train_and_predict_vampire`
    together with :func:`align_kfold_predictions_with_contours`.

    Parameters
    ----------
    base_path : str
        Root directory containing data.
    treatments : list of str
        Treatment condition subdirectory names.
    groups : list of str
        Experimental group subdirectory names.
    """

    def __init__(self, base_path: str, treatments: List[str], groups: List[str]):
        self.base_path = Path(base_path)
        self.treatments = treatments
        self.groups = groups
        self.model_path: Optional[Path] = None

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

    def extract_features(self, image_set_path: Path) -> None:
        """Extract contour features from images using VAMPIRE."""
        logger.info(f"Extracting features from: {image_set_path}")
        try:
            vampire.extraction.extract_properties(str(image_set_path))
            logger.info("Feature extraction completed")
        except Exception as exc:
            logger.error(f"Feature extraction failed: {exc}")
            raise

    def train_model(
        self,
        image_set_path: Path,
        model_name: str = "li",
        num_points: int = 50,
        num_clusters: int = 5,
        num_pc: int = 10,
    ) -> Path:
        """
        Train a VAMPIRE model on extracted features.

        Parameters
        ----------
        image_set_path : Path
            Directory containing training images (features already extracted).
        model_name : str
            Name tag for the saved model file.
        num_points : int
            Number of contour points.
        num_clusters : int
            Number of shape-mode clusters.
        num_pc : int
            Number of principal components. **Must be a fixed integer** to
            ensure consistent dimensionality across folds. Avoid ``np.nan``.

        Returns
        -------
        Path
            Path to the saved model pickle file.
        """
        logger.info(f"Training VAMPIRE model: {model_name}")

        build_info_df = pd.DataFrame(
            {
                "img_set_path": [str(image_set_path)],
                "output_path": [str(image_set_path)],
                "model_name": [model_name],
                "num_points": [num_points],
                "num_clusters": [num_clusters],
                "num_pc": [num_pc],
            }
        )

        try:
            vampire.quickstart.fit_models(build_info_df)
        except Exception as exc:
            logger.error(f"Model training failed: {exc}")
            raise

        model_pattern = f"model_{model_name}_({num_points}_{num_clusters}_*)__.pickle"
        model_files = list(image_set_path.glob(model_pattern))

        if not model_files:
            raise FileNotFoundError(
                f"No model file matching '{model_pattern}' in {image_set_path}"
            )

        self.model_path = model_files[0]
        logger.info(f"Model saved to: {self.model_path}")
        return self.model_path

    def create_apply_dataframe(
        self, test_base_path: Path, model_path: Path
    ) -> pd.DataFrame:
        """Build the apply-info DataFrame used by ``vampire.quickstart.transform_datasets``."""
        apply_data = []

        for group in self.groups:
            for treatment in self.treatments:
                img_set_path = test_base_path / group / treatment
                if img_set_path.exists():
                    apply_data.append(
                        {
                            "img_set_path": str(img_set_path),
                            "model_path": str(model_path),
                            "output_path": str(img_set_path),
                            "img_set_name": treatment,
                        }
                    )
                else:
                    logger.warning(f"Path does not exist, skipping: {img_set_path}")

        if not apply_data:
            raise ValueError("No valid test datasets found")

        return pd.DataFrame(apply_data)

    def apply_model(self, test_base_path: Path) -> None:
        """Apply the trained model to test datasets."""
        if self.model_path is None:
            raise ValueError("Model must be trained before applying.")

        logger.info(f"Applying model to test datasets in: {test_base_path}")
        apply_info_df = self.create_apply_dataframe(test_base_path, self.model_path)
        logger.info(f"Applying to {len(apply_info_df)} dataset(s)")

        try:
            from unittest.mock import patch
            with patch('vampire.plot.plot_distribution_contour_dendrogram'):
                vampire.quickstart.transform_datasets(apply_info_df)
        except Exception as exc:
            logger.error(f"Model application failed: {exc}")
            raise

    def run_full_pipeline(
        self,
        training_subpath: str = "training/vampire_data",
        testing_subpath: str = "testing/vampire_data",
    ) -> None:
        """Run feature extraction → training → application in sequence."""
        train_path = self.base_path / training_subpath
        test_path = self.base_path / testing_subpath

        logger.info(f"Training path : {train_path}")
        logger.info(f"Testing path  : {test_path}")

        self.extract_features(train_path)
        self.train_model(train_path)
        self.apply_model(test_path)

        logger.info("Pipeline completed successfully")


# ── Functional k-fold interface ───────────────────────────────────────────────

def train_and_predict_vampire(
    train_files: List[str],
    val_files: List[str],
    output_dir: str,
    fold_name: str = "fold",
    model_name: str = "vampire",
    num_points: int = 50,
    num_clusters: int = 5,
    num_pc: int = 10,
) -> Tuple[Path, pd.DataFrame]:
    """
    Train a VAMPIRE model on training files and predict on validation files.

    Parameters
    ----------
    train_files : list of str
        Training image file paths.
    val_files : list of str
        Validation image file paths.
    output_dir : str
        Writable directory for outputs (model pickle, copied images, CSVs).
        Subdirectories ``train/`` and ``val/`` are created automatically.
    fold_name : str
        Identifier for this fold, used in output filenames.
    model_name : str
        Name tag embedded in the model pickle filename.
    num_points : int
        Number of contour points for VAMPIRE.
    num_clusters : int
        Number of shape-mode clusters.
    num_pc : int
        Number of principal components. **Must be a fixed integer** — do not
        pass ``np.nan`` here, as this causes inconsistent PCA dimensionality
        across folds which breaks contour-based alignment.

    Returns
    -------
    model_path : Path
        Path to the trained model pickle.
    predictions_df : pd.DataFrame
        VAMPIRE predictions for the validation set, with an added ``fold``
        column.

    Raises
    ------
    FileNotFoundError
        If any source file is missing, or if no model/predictions file is
        produced by VAMPIRE.
    OSError
        If ``output_dir`` cannot be created.
    """
    output_dir = Path(output_dir).resolve()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Cannot create output directory '{output_dir}': {exc}\n"
            "Use a writable path such as Path.home() / 'vampire_output'."
        ) from exc

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    logger.info(f"{fold_name}: {len(train_files)} train, {len(val_files)} val files")

    # Copy files
    for src in train_files:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Training file not found: {src}")
        shutil.copy2(src_path, train_dir / src_path.name)

    for src in val_files:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Validation file not found: {src}")
        shutil.copy2(src_path, val_dir / src_path.name)

    # Feature extraction
    logger.info("Extracting features...")
    try:
        vampire.extraction.extract_properties(str(train_dir))
    except Exception as exc:
        logger.error(f"Feature extraction failed: {exc}")
        raise

    # Model training
    logger.info(f"Training model (num_pc={num_pc})...")
    build_info_df = pd.DataFrame(
        {
            "img_set_path": [str(train_dir)],
            "output_path": [str(output_dir)],
            "model_name": [model_name],
            "num_points": [num_points],
            "num_clusters": [num_clusters],
            "num_pc": [num_pc],          # fixed integer — never np.nan
        }
    )

    try:
        vampire.quickstart.fit_models(build_info_df)
    except Exception as exc:
        logger.error(f"Model training failed: {exc}")
        raise

    # Locate model file
    model_pattern = f"model_{model_name}_({num_points}_{num_clusters}_*)__.pickle"
    model_files = list(output_dir.glob(model_pattern))
    if not model_files:
        raise FileNotFoundError(
            f"No model file matching '{model_pattern}' found in {output_dir}"
        )
    model_path = model_files[0]
    logger.info(f"Model saved to: {model_path}")

    # Apply to validation set
    logger.info("Applying model to validation set...")
    apply_info_df = pd.DataFrame(
        {
            "img_set_path": [str(val_dir)],
            "model_path": [str(model_path)],
            "output_path": [str(output_dir)],
            "img_set_name": [fold_name],
        }
    )

    try:
        from unittest.mock import patch, MagicMock
        mock_plot = MagicMock()
        mock_plot.plot_distribution_contour_dendrogram.return_value = (MagicMock(), MagicMock())
        with patch('vampire.quickstart.plot', mock_plot):
            vampire.quickstart.transform_datasets(apply_info_df)
    except Exception as exc:
        logger.error(f"Model application failed: {exc}")
        raise

    # Locate predictions CSV — try specific pattern first, then fallback
    pred_pattern = f"apply-properties_{model_name}_on_{fold_name}_*.csv"

    # Search in fold directory and one level down (VAMPIRE writes to fold subdir)
    pred_files = list(output_dir.glob(pred_pattern))

    if not pred_files:
        pred_files = list(output_dir.glob(f"*/{pred_pattern}"))

    if not pred_files:
        pred_files = list(output_dir.glob("apply-properties_*.csv"))

    if not pred_files:
        pred_files = list(output_dir.glob("*/apply-properties_*.csv"))

    if not pred_files:
        raise FileNotFoundError(
            f"No predictions CSV found in {output_dir} or its subdirectories.\n"
            f"Searched for: '{pred_pattern}' and 'apply-properties_*.csv'\n"
            f"Files present: {list(output_dir.rglob('*.csv'))}"
        )

    # If multiple CSVs somehow exist, take the one whose name contains fold_name
    named = [p for p in pred_files if fold_name in p.name]
    predictions_csv = named[0] if named else pred_files[0]
    logger.info(f"Loading predictions from: {predictions_csv}")

    predictions_df = pd.read_csv(predictions_csv)

    if "cluster_id" not in predictions_df.columns:
        logger.warning(
            f"'cluster_id' column not found. "
            f"Available columns: {list(predictions_df.columns)}"
        )
    else:
        logger.info(
            f"cluster_id: {predictions_df['cluster_id'].nunique()} unique values"
        )

    predictions_df["fold"] = fold_name
    logger.info(f"Loaded {len(predictions_df)} predictions")

    return model_path, predictions_df


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_vampire_models_by_contours(
    model_paths: List[str],
    reference_model_idx: int = 0,
) -> Tuple[List, Dict[int, Dict[int, int]]]:
    """
    Align VAMPIRE cluster labels across folds using contour centroids.

    Uses the Hungarian algorithm on pairwise Euclidean distances between
    ``mean_cluster_contours`` arrays. This is the correct alignment space
    because VAMPIRE defines clusters in contour shape space, not in
    morphological feature space.

    Parameters
    ----------
    model_paths : list of str
        Paths to VAMPIRE model pickle files, one per fold.
    reference_model_idx : int
        Index of the model whose labels are used as the reference.
        All other folds are remapped to match this fold.

    Returns
    -------
    models : list
        Loaded VAMPIRE model objects (unmodified).
    cluster_mappings : dict of {int: dict}
        ``{fold_idx: {original_cluster_id: aligned_cluster_id}}``.
        The reference fold receives an identity mapping.

    Raises
    ------
    AttributeError
        If any model lacks ``mean_cluster_contours``.
    ValueError
        If models have inconsistent ``n_clusters`` or ``n_points``.

    Notes
    -----
    Models are not mutated — only the mapping dictionary is returned.
    Apply mappings with :func:`apply_alignment_to_predictions`.
    """
    logger.info(
        f"Aligning {len(model_paths)} VAMPIRE models "
        f"(reference: fold {reference_model_idx})..."
    )

    models = [load_vampire_model(p) for p in model_paths]

    # Validate consistency
    ref_n_clusters = models[0].n_clusters
    ref_n_points = models[0].n_points

    for i, model in enumerate(models[1:], 1):
        if model.n_clusters != ref_n_clusters:
            raise ValueError(
                f"Model {i} has {model.n_clusters} clusters; "
                f"reference has {ref_n_clusters}."
            )
        if model.n_points != ref_n_points:
            raise ValueError(
                f"Model {i} has {model.n_points} points; "
                f"reference has {ref_n_points}."
            )

    # Validate and retrieve reference contours
    reference_contours = _validate_model_contours(
        models[reference_model_idx], f"fold_{reference_model_idx}"
    )
    logger.info(f"Reference contours shape: {reference_contours.shape}")

    cluster_mappings: Dict[int, Dict[int, int]] = {}

    for fold_idx, model in enumerate(models):
        if fold_idx == reference_model_idx:
            cluster_mappings[fold_idx] = {i: i for i in range(ref_n_clusters)}
            logger.info(f"Fold {fold_idx}: reference fold — identity mapping")
            continue

        current_contours = _validate_model_contours(model, f"fold_{fold_idx}")

        # cdist expects (n_a, features) and (n_b, features)
        # reference is rows, current fold is columns
        # cost_matrix[i, j] = distance from reference cluster i to fold cluster j
        cost_matrix = cdist(reference_contours, current_contours, metric="euclidean")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # col_ind[i] is the fold cluster that best matches reference cluster row_ind[i]
        # mapping: fold's original label -> aligned (reference) label
        mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}

        mean_cost = cost_matrix[row_ind, col_ind].mean()
        logger.info(
            f"Fold {fold_idx}: mapping={mapping}  |  mean alignment cost={mean_cost:.4f}"
        )

        cluster_mappings[fold_idx] = mapping

    logger.info("Alignment complete.")
    return models, cluster_mappings


def apply_alignment_to_predictions(
    predictions_df: pd.DataFrame,
    fold_idx: int,
    cluster_mapping: Dict[int, int],
) -> pd.DataFrame:
    """
    Apply a cluster alignment mapping to a predictions DataFrame.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain a ``cluster_id`` column with integer labels.
    fold_idx : int
        Fold index used only for logging.
    cluster_mapping : dict of {int: int}
        ``{original_cluster_id: aligned_cluster_id}`` for this fold.

    Returns
    -------
    pd.DataFrame
        Copy of ``predictions_df`` with:
        - ``cluster_id_original``: preserves the pre-alignment label
        - ``cluster_id``: remapped aligned label
    """
    aligned_df = predictions_df.copy()
    aligned_df["cluster_id_original"] = aligned_df["cluster_id"]
    aligned_df["cluster_id"] = aligned_df["cluster_id_original"].map(cluster_mapping)

    n_unmapped = aligned_df["cluster_id"].isna().sum()
    if n_unmapped > 0:
        logger.warning(
            f"Fold {fold_idx}: {n_unmapped} rows could not be mapped "
            f"(cluster IDs not in mapping dict). Check for label mismatches."
        )

    logger.info(
        f"Fold {fold_idx}: aligned {len(aligned_df)} predictions "
        f"({aligned_df['cluster_id'].nunique()} unique clusters after alignment)"
    )
    return aligned_df


def align_kfold_predictions_with_contours(
    all_predictions: List[pd.DataFrame],
    model_paths: List[str],
    reference_fold_idx: int = 0,
) -> List[pd.DataFrame]:
    """
    End-to-end alignment of k-fold predictions using contour centroids.

    Parameters
    ----------
    all_predictions : list of pd.DataFrame
        One predictions DataFrame per fold (output of
        :func:`train_and_predict_vampire`).
    model_paths : list of str
        Paths to the corresponding model pickle files.
    reference_fold_idx : int
        Fold whose cluster labels are used as the reference.

    Returns
    -------
    list of pd.DataFrame
        Aligned predictions, one per fold. Each DataFrame has both
        ``cluster_id`` (aligned) and ``cluster_id_original`` columns.

    Examples
    --------
    >>> aligned = align_kfold_predictions_with_contours(
    ...     all_predictions, model_paths, reference_fold_idx=0
    ... )
    >>> combined = pd.concat(aligned, ignore_index=True)
    """
    _, cluster_mappings = align_vampire_models_by_contours(
        model_paths, reference_model_idx=reference_fold_idx
    )

    aligned_predictions = []
    for fold_idx, (preds_df, mapping) in enumerate(
        zip(all_predictions, cluster_mappings.values())
    ):
        aligned_df = apply_alignment_to_predictions(preds_df, fold_idx, mapping)
        aligned_predictions.append(aligned_df)

    # Summary verification
    logger.info("\nAlignment verification:")
    for fold_idx, aligned_df in enumerate(aligned_predictions):
        orig = sorted(aligned_df["cluster_id_original"].unique())
        aligned = sorted(aligned_df["cluster_id"].dropna().unique())
        logger.info(f"  Fold {fold_idx}: {orig} -> {aligned}")

    return aligned_predictions


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_shape_consistency_metrics(
    model_paths: List[str],
    reference_fold_idx: int = 0,
) -> pd.DataFrame:
    """
    Quantify shape consistency across folds before and after alignment.

    For each shape mode, computes pairwise Euclidean distances between mean
    contours across all fold pairs, before and after applying the Hungarian
    alignment. A large ``improvement_percent`` indicates the raw cluster
    labels were inconsistent and alignment was necessary.

    Parameters
    ----------
    model_paths : list of str
        Paths to VAMPIRE model pickle files.
    reference_fold_idx : int
        Reference fold for alignment.

    Returns
    -------
    pd.DataFrame
        One row per shape mode with columns:
        ``shape_mode``, ``mean_dist_before``, ``mean_dist_after``,
        ``std_dist_before``, ``std_dist_after``, ``improvement_percent``.
    """
    models, cluster_mappings = align_vampire_models_by_contours(
        model_paths, reference_model_idx=reference_fold_idx
    )

    n_folds = len(models)
    num_clusters = models[0].n_clusters
    metrics = []

    for cluster_id in range(num_clusters):
        # Contours BEFORE alignment — raw cluster index
        contours_before = [
            model.mean_cluster_contours[cluster_id] for model in models
        ]

        # Contours AFTER alignment — follow the mapping back to original index
        contours_after = []
        for fold_idx, model in enumerate(models):
            mapping = cluster_mappings[fold_idx]
            # Invert: find which original cluster maps to this aligned cluster_id
            original_id = next(
                orig for orig, aligned in mapping.items() if aligned == cluster_id
            )
            contours_after.append(model.mean_cluster_contours[original_id])

        def _pairwise_distances(contour_list):
            dists = []
            for i in range(n_folds):
                for j in range(i + 1, n_folds):
                    dists.append(
                        np.linalg.norm(contour_list[i] - contour_list[j])
                    )
            return np.array(dists)

        d_before = _pairwise_distances(contours_before)
        d_after = _pairwise_distances(contours_after)

        improvement = (
            (d_before.mean() - d_after.mean()) / d_before.mean() * 100
            if d_before.mean() > 0
            else 0.0
        )

        metrics.append(
            {
                "shape_mode": cluster_id,
                "mean_dist_before": d_before.mean(),
                "mean_dist_after": d_after.mean(),
                "std_dist_before": d_before.std(),
                "std_dist_after": d_after.std(),
                "improvement_percent": improvement,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    logger.info(f"\nShape consistency metrics:\n{metrics_df.to_string(index=False)}")
    return metrics_df


# ── Visualisation ─────────────────────────────────────────────────────────────

def compare_contours_overlay(
    model_paths: List[str],
    output_path: str,
    fold_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 3),
    contour_scale: float = 3.0,
    alpha: float = 0.4,
    lw: float = 2.0,
) -> None:
    """
    Overlay mean contours from all folds for each shape mode.

    Each subplot shows one shape mode with all folds overlaid. Low variation
    across folds indicates stable, reproducible clusters.

    Parameters
    ----------
    model_paths : list of str
        Paths to VAMPIRE model pickle files.
    output_path : str
        Path to save the figure (PNG). A PDF copy is also saved.
    fold_names : list of str, optional
        Display names for each fold. Defaults to ``['Fold 1', 'Fold 2', ...]``.
    figsize : tuple of int
        Matplotlib figure size.
    contour_scale : float
        Scaling factor applied to centred contour coordinates.
    alpha : float
        Line transparency.
    lw : float
        Line width.
    """
    n_folds = len(model_paths)
    fold_names = fold_names or [f"Fold {i + 1}" for i in range(n_folds)]

    models = [load_vampire_model(p) for p in model_paths]
    num_clusters = models[0].n_clusters
    num_points = models[0].n_points

    fold_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_folds))

    fig, axes = plt.subplots(1, num_clusters, figsize=figsize)
    if num_clusters == 1:
        axes = [axes]

    for cluster_id in range(num_clusters):
        ax = axes[cluster_id]

        for fold_idx, model in enumerate(models):
            x = model.mean_cluster_contours[cluster_id, :num_points]
            y = model.mean_cluster_contours[cluster_id, num_points:]

            x = np.append(x, x[0])
            y = np.append(y, y[0])

            x_c = (x - x.mean()) * contour_scale
            y_c = (y - y.mean()) * contour_scale

            ax.plot(
                x_c, y_c,
                color=fold_colors[fold_idx],
                alpha=alpha, lw=lw,
                label=fold_names[fold_idx] if cluster_id == 0 else None,
            )

        ax.set_title(f"Shape Mode {cluster_id}", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.suptitle(
        "Shape Consistency Across Folds (Overlaid)\n"
        "Low variation = stable clusters",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved overlay comparison to {output_path}")


def compare_alignment_effect(
    model_paths: List[str],
    output_path: str,
    reference_fold_idx: int = 0,
    figsize: Tuple[int, int] = (16, 6),
    contour_scale: float = 3.0,
    alpha: float = 0.4,
    lw: float = 2.0,
) -> None:
    """
    Side-by-side comparison of contour consistency before and after alignment.

    Top row: raw cluster labels. Bottom row: Hungarian-aligned labels.
    If alignment is working, the bottom row should show substantially lower
    shape variation across folds per column.

    Parameters
    ----------
    model_paths : list of str
        Paths to VAMPIRE model pickle files.
    output_path : str
        Path to save the figure (PNG). A PDF copy is also saved.
    reference_fold_idx : int
        Reference fold for alignment.
    figsize : tuple of int
        Matplotlib figure size.
    contour_scale : float
        Scaling factor applied to centred contour coordinates.
    alpha : float
        Line transparency.
    lw : float
        Line width.
    """
    models, cluster_mappings = align_vampire_models_by_contours(
        model_paths, reference_model_idx=reference_fold_idx
    )

    n_folds = len(models)
    num_clusters = models[0].n_clusters
    num_points = models[0].n_points
    fold_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_folds))

    fig, axes = plt.subplots(2, num_clusters, figsize=figsize)
    if num_clusters == 1:
        axes = axes.reshape(2, 1)

    def _plot_contour(ax, model, cluster_id, fold_idx, label=None):
        x = model.mean_cluster_contours[cluster_id, :num_points]
        y = model.mean_cluster_contours[cluster_id, num_points:]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        x_c = (x - x.mean()) * contour_scale
        y_c = (y - y.mean()) * contour_scale
        ax.plot(x_c, y_c, color=fold_colors[fold_idx],
                alpha=alpha, lw=lw, label=label)
        ax.set_aspect("equal")
        ax.axis("off")

    # Top row — before alignment (raw cluster IDs)
    for cluster_id in range(num_clusters):
        ax = axes[0, cluster_id]
        for fold_idx, model in enumerate(models):
            label = f"Fold {fold_idx + 1}" if cluster_id == 0 else None
            _plot_contour(ax, model, cluster_id, fold_idx, label)
        ax.set_title(f"SM {cluster_id}", fontsize=10, fontweight="bold")

    axes[0, 0].set_ylabel("Before\nAlignment", fontsize=11, fontweight="bold")
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Bottom row — after alignment (mapped cluster IDs)
    for aligned_id in range(num_clusters):
        ax = axes[1, aligned_id]
        for fold_idx, model in enumerate(models):
            mapping = cluster_mappings[fold_idx]
            original_id = next(
                orig for orig, aligned in mapping.items() if aligned == aligned_id
            )
            _plot_contour(ax, model, original_id, fold_idx)

    axes[1, 0].set_ylabel("After\nAlignment", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Effect of Contour-Based Alignment on Shape Consistency",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved alignment effect comparison to {output_path}")