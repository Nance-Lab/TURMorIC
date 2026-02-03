#!/usr/bin/env python
# coding: utf-8

"""
Plotting utilities for VAMPIRE k-fold cross-validation analysis.

This module provides visualization functions for analyzing VAMPIRE model
results from k-fold cross-validation, including shape mode distributions,
feature comparisons, and fold-wise analysis.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


# Configure logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_cluster_distribution(
    predictions_df: pd.DataFrame,
    output_path: str,
    title: str = "Shape Mode Distribution",
    figsize: Tuple[int, int] = (10, 6),
    by_fold: bool = False
) -> None:
    """
    Plot distribution of cluster assignments (shape modes).
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame containing 'cluster_id' column
    output_path : str
        Path to save the plot
    title : str, optional
        Plot title (default: "Shape Mode Distribution")
    figsize : Tuple[int, int], optional
        Figure size (default: (10, 6))
    by_fold : bool, optional
        If True and 'fold' column exists, show distribution per fold (default: False)
        
    Examples
    --------
    >>> plot_cluster_distribution(combined_df, 'cluster_dist.png')
    >>> plot_cluster_distribution(combined_df, 'cluster_by_fold.png', by_fold=True)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if by_fold and 'fold' in predictions_df.columns:
        # Stacked bar chart by fold
        pivot_data = predictions_df.groupby(['fold', 'cluster_id']).size().unstack(fill_value=0)
        pivot_data.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax.legend(title='Shape Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        # Simple bar chart of overall distribution
        cluster_counts = predictions_df['cluster_id'].value_counts().sort_index()
        bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Shape Mode (Cluster ID)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax.set_xticks(cluster_counts.index)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved cluster distribution plot to {output_path}")
    plt.close()


def plot_feature_distributions_by_cluster(
    predictions_df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: str,
    max_features_per_plot: int = 6
) -> None:
    """
    Plot feature distributions for each cluster (shape mode).
    
    Creates violin plots showing how morphological features differ across clusters.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame with 'cluster_id' and feature columns
    feature_columns : List[str]
        List of feature column names to plot
    output_dir : str
        Directory to save plots
    max_features_per_plot : int, optional
        Maximum features per figure (default: 6)
        
    Examples
    --------
    >>> features = ['area', 'perimeter', 'eccentricity', 'solidity']
    >>> plot_feature_distributions_by_cluster(df, features, 'plots/')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split features into chunks
    n_plots = int(np.ceil(len(feature_columns) / max_features_per_plot))
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * max_features_per_plot
        end_idx = min(start_idx + max_features_per_plot, len(feature_columns))
        features_subset = feature_columns[start_idx:end_idx]
        
        n_features = len(features_subset)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Create subplots
        if n_features == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            axes = [ax]
        else:
            fig, axes_array = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            # Always flatten to 1D array for consistent indexing
            if isinstance(axes_array, np.ndarray):
                axes = axes_array.flatten()
            else:
                axes = [axes_array]
        
        for idx, feature in enumerate(features_subset):
            ax = axes[idx]
            
            # Create violin plot
            sns.violinplot(
                data=predictions_df,
                x='cluster_id',
                y=feature,
                ax=ax,
                inner='box'
            )
            
            ax.set_xlabel('Shape Mode', fontsize=10, fontweight='bold')
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_title(f'{feature.replace("_", " ").title()} by Shape Mode', 
                        fontsize=11, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = output_dir / f'feature_distributions_{plot_idx+1}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        logger.info(f"Saved feature distribution plot to {output_path}")
        plt.close()


def plot_fold_comparison(
    all_predictions: List[pd.DataFrame],
    output_path: str,
    metric: str = 'cluster_id',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Compare a metric across all folds.
    
    Parameters
    ----------
    all_predictions : List[pd.DataFrame]
        List of prediction DataFrames, one per fold
    output_path : str
        Path to save the plot
    metric : str, optional
        Column name to compare (default: 'cluster_id')
    figsize : Tuple[int, int], optional
        Figure size (default: (12, 6))
        
    Examples
    --------
    >>> plot_fold_comparison(aligned_predictions, 'fold_comparison.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    fold_data = []
    for fold_idx, df in enumerate(all_predictions):
        if metric == 'cluster_id':
            # Count distribution
            counts = df['cluster_id'].value_counts().sort_index()
            fold_data.append(counts)
        else:
            # Extract metric values
            fold_data.append(df[metric].values)
    
    if metric == 'cluster_id':
        # Heatmap of cluster counts
        data_matrix = pd.DataFrame(fold_data)
        data_matrix.index = [f'Fold {i+1}' for i in range(len(fold_data))]
        data_matrix.columns = [f'Cluster {i}' for i in data_matrix.columns]
        
        sns.heatmap(data_matrix, annot=True, fmt='g', cmap='YlOrRd', 
                   cbar_kws={'label': 'Cell Count'}, ax=ax)
        ax.set_title('Cluster Distribution Across Folds', fontsize=14, fontweight='bold')
    else:
        # Box plot comparison
        plot_data = pd.DataFrame({
            f'Fold {i+1}': data 
            for i, data in enumerate(fold_data)
        })
        plot_data.boxplot(ax=ax)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Across Folds', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved fold comparison plot to {output_path}")
    plt.close()


def plot_alignment_comparison(
    predictions_unaligned: pd.DataFrame,
    predictions_aligned: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Compare cluster distributions before and after alignment.
    
    Parameters
    ----------
    predictions_unaligned : pd.DataFrame
        Predictions before alignment
    predictions_aligned : pd.DataFrame
        Predictions after alignment
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int], optional
        Figure size (default: (14, 6))
        
    Examples
    --------
    >>> plot_alignment_comparison(unaligned_df, aligned_df, 'alignment_effect.png')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Before alignment
    if 'fold' in predictions_unaligned.columns:
        pivot_unaligned = predictions_unaligned.groupby(['fold', 'cluster_id']).size().unstack(fill_value=0)
        pivot_unaligned.plot(kind='bar', stacked=True, ax=ax1, width=0.8)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax1.set_title('Before Alignment', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fold', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Cell Count', fontsize=10, fontweight='bold')
    
    # After alignment
    if 'fold' in predictions_aligned.columns:
        pivot_aligned = predictions_aligned.groupby(['fold', 'cluster_id']).size().unstack(fill_value=0)
        pivot_aligned.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title('After Alignment', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fold', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Cell Count', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved alignment comparison plot to {output_path}")
    plt.close()


def plot_pca_clusters(
    predictions_df: pd.DataFrame,
    feature_columns: List[str],
    output_path: str,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot PCA visualization of clusters.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with cluster_id and feature columns
    feature_columns : List[str]
        Feature columns to use for PCA
    output_path : str
        Path to save the plot
    n_components : int, optional
        Number of PCA components (default: 2)
    figsize : Tuple[int, int], optional
        Figure size (default: (10, 8))
        
    Examples
    --------
    >>> features = ['area', 'perimeter', 'eccentricity', 'solidity']
    >>> plot_pca_clusters(df, features, 'pca_plot.png')
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = predictions_df[feature_columns].dropna()
    y = predictions_df.loc[X.index, 'cluster_id']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        X_pca[:, 0], 
        X_pca[:, 1],
        c=y,
        cmap='tab10',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=50
    )
    
    # Add cluster centroids
    for cluster_id in sorted(y.unique()):
        cluster_mask = y == cluster_id
        centroid = X_pca[cluster_mask].mean(axis=0)
        ax.scatter(
            centroid[0], centroid[1],
            c='red', marker='X', s=200,
            edgecolors='black', linewidth=2,
            zorder=10
        )
        ax.text(
            centroid[0], centroid[1],
            str(cluster_id),
            fontsize=12, fontweight='bold',
            ha='center', va='center'
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('PCA Visualization of Shape Modes', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Shape Mode', ax=ax)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved PCA plot to {output_path}")
    plt.close()


def plot_dendrogram_with_shapes(
    predictions_df: pd.DataFrame,
    feature_columns: List[str],
    output_path: str,
    method: str = 'ward',
    metric: str = 'euclidean',
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot hierarchical clustering dendrogram of shape modes.
    
    Creates a dendrogram showing the hierarchical relationships between
    different shape modes (clusters) based on their morphological features.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with cluster_id and feature columns
    feature_columns : List[str]
        Morphological feature columns to use for clustering
    output_path : str
        Path to save the plot
    method : str, optional
        Linkage method for hierarchical clustering (default: 'ward')
        Options: 'ward', 'single', 'complete', 'average'
    metric : str, optional
        Distance metric (default: 'euclidean')
    figsize : Tuple[int, int], optional
        Figure size (default: (12, 8))
        
    Examples
    --------
    >>> features = ['area', 'perimeter', 'eccentricity', 'solidity']
    >>> plot_dendrogram_with_shapes(df, features, 'dendrogram.png')
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    
    # Compute cluster centroids
    cluster_centroids = predictions_df.groupby('cluster_id')[feature_columns].mean()
    cluster_ids = cluster_centroids.index.tolist()
    
    # Standardize features
    scaler = StandardScaler()
    centroids_scaled = scaler.fit_transform(cluster_centroids)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(centroids_scaled, method=method, metric=metric)
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(
        linkage_matrix,
        labels=[f'SM {i}' for i in cluster_ids],
        ax=ax,
        color_threshold=0,
        above_threshold_color='black'
    )
    
    ax.set_xlabel('Shape Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering of Shape Modes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved dendrogram to {output_path}")
    plt.close()


def plot_mean_shape_contours(
    image_dir: str,
    predictions_df: pd.DataFrame,
    output_path: str,
    n_cols: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    max_cells_per_cluster: int = 50
) -> None:
    """
    Plot mean shape contours for each cluster.
    
    For each shape mode, this function finds example cells, extracts their
    contours, and displays the mean shape alongside example cells.
    
    Parameters
    ----------
    image_dir : str
        Directory containing the original labeled images (.tif or .npy files)
    predictions_df : pd.DataFrame
        Predictions with cluster_id and filename columns
    output_path : str
        Path to save the plot
    n_cols : int, optional
        Number of columns in the subplot grid (default: 5)
    figsize : Tuple[int, int], optional
        Figure size. If None, calculated based on n_cols (default: None)
    max_cells_per_cluster : int, optional
        Maximum number of cells to average per cluster (default: 50)
        
    Examples
    --------
    >>> plot_mean_shape_contours(
    ...     '/path/to/images',
    ...     predictions_df,
    ...     'mean_shapes.png'
    ... )
    
    Notes
    -----
    - Requires image files to be accessible
    - Works with both .tif and .npy labeled images
    - Normalizes cell sizes for comparison
    """
    from skimage import io, measure
    from skimage.transform import resize
    import numpy as np
    
    image_dir = Path(image_dir)
    n_clusters = predictions_df['cluster_id'].nunique()
    
    # Calculate figure size
    n_rows = int(np.ceil(n_clusters / n_cols))
    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    logger.info(f"Extracting mean shapes for {n_clusters} clusters...")
    
    for cluster_idx, cluster_id in enumerate(sorted(predictions_df['cluster_id'].unique())):
        ax = axes[cluster_idx]
        
        # Get cells from this cluster
        cluster_cells = predictions_df[predictions_df['cluster_id'] == cluster_id]
        
        # Sample if too many cells
        if len(cluster_cells) > max_cells_per_cluster:
            cluster_cells = cluster_cells.sample(n=max_cells_per_cluster, random_state=42)
        
        logger.info(f"Processing cluster {cluster_id}: {len(cluster_cells)} cells")
        
        # Collect cell contours
        cell_masks = []
        
        for idx, row in cluster_cells.iterrows():
            try:
                # Try to find the image file
                filename = Path(row['filename']) if 'filename' in row else None
                if filename is None or not filename.exists():
                    # Try to construct path
                    if 'filename' in row:
                        possible_paths = [
                            image_dir / Path(row['filename']).name,
                            Path(row['filename'])
                        ]
                        for p in possible_paths:
                            if p.exists():
                                filename = p
                                break
                
                if filename is None or not filename.exists():
                    continue
                
                # Load image
                if filename.suffix == '.npy':
                    img = np.load(filename)
                else:
                    img = io.imread(filename)
                
                # Get labeled regions
                if img.dtype == bool:
                    labeled = measure.label(img)
                else:
                    labeled = img
                
                # Extract individual cell regions
                for region in measure.regionprops(labeled):
                    # Get bounding box
                    minr, minc, maxr, maxc = region.bbox
                    cell_mask = region.image
                    
                    # Normalize size (resize to standard size)
                    cell_mask_resized = resize(
                        cell_mask.astype(float),
                        (50, 50),
                        anti_aliasing=True
                    )
                    cell_masks.append(cell_mask_resized > 0.5)
                    
                    # Only take first cell from each image for speed
                    break
                    
            except Exception as e:
                logger.debug(f"Could not process cell from {filename}: {e}")
                continue
        
        # Compute mean shape
        if len(cell_masks) > 0:
            mean_shape = np.mean(np.array(cell_masks), axis=0)
            
            # Plot mean contour
            ax.imshow(mean_shape, cmap='gray', interpolation='bilinear')
            ax.contour(mean_shape, levels=[0.5], colors='red', linewidths=2)
            ax.set_title(f'Shape Mode {cluster_id}\n(n={len(cluster_cells)})', 
                        fontweight='bold', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No cells\nfound', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10)
            ax.set_title(f'Shape Mode {cluster_id}', fontweight='bold', fontsize=10)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved mean shape contours to {output_path}")
    plt.close()


def load_vampire_model(model_path: str):
    """
    Load a trained VAMPIRE model from pickle file.
    
    Parameters
    ----------
    model_path : str
        Path to the VAMPIRE model pickle file
        
    Returns
    -------
    model : vampire.model.Vampire
        Loaded VAMPIRE model object
        
    Examples
    --------
    >>> model = load_vampire_model('/output/fold_1/model_vampire_kfold_*.pickle')
    >>> print(f"Model has {model.n_clusters} clusters")
    """
    import pickle
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading VAMPIRE model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model: {model.n_clusters} clusters, {model.n_points} points")
    
    return model


def plot_vampire_model_contours(
    model_path: str,
    output_path: str,
    contour_scale: float = 3,
    fig_size: Tuple[int, int] = (12, 3),
    colors: Optional[List] = None,
    alpha: float = 1.0,
    lw: float = 2
) -> None:
    """
    Plot mean contours directly from a trained VAMPIRE model.
    
    This function loads a VAMPIRE model pickle file and plots the mean
    cluster contours in VAMPIRE's signature horizontal layout.
    
    Parameters
    ----------
    model_path : str
        Path to the VAMPIRE model pickle file
    output_path : str
        Path to save the plot
    contour_scale : float, optional
        Scale of the contour shapes (default: 3)
    fig_size : Tuple[int, int], optional
        Figure size (default: (12, 3))
    colors : List, optional
        Colors for each cluster. If None, uses twilight colormap
    alpha : float, optional
        Alpha transparency (default: 1.0)
    lw : float, optional
        Line width (default: 2)
        
    Examples
    --------
    >>> # Plot contours from a single fold's model
    >>> plot_vampire_model_contours(
    ...     model_path='/output/fold_1/model_vampire_kfold_*.pickle',
    ...     output_path='mean_contours.png'
    ... )
    
    Notes
    -----
    This replicates VAMPIRE's native plot_contours() function.
    The model must be already trained (fitted) to have mean_cluster_contours.
    """
    import matplotlib.pyplot as plt
    
    # Load model
    model = load_vampire_model(model_path)
    
    # Get mean cluster contours from model
    mean_cluster_contours = model.mean_cluster_contours
    num_clusters = model.n_clusters
    num_points = model.n_points
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set colors
    if colors is None:
        colors = [plt.get_cmap('twilight')(i) 
                 for i in np.linspace(0.1, 0.9, num_clusters)]
    elif isinstance(colors, str):
        colors = [colors] * num_clusters
    
    # Plot parameters
    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours
    
    for i in range(num_clusters):
        # Read contour coordinates (VAMPIRE format: [x1...xn, y1...yn])
        x = mean_cluster_contours[i, :num_points]
        y = mean_cluster_contours[i, num_points:]
        
        # Form closed shape
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        
        # Place shape in right location
        x = x * contour_scale + x_first + x_offset * i
        y = y * contour_scale
        
        # Plot shape
        ax.plot(x, y, color=colors[i], alpha=alpha, lw=lw)
        
        # Add cluster label below
        ax.text(x_first + x_offset * i, -contour_scale * 1.5, 
               f'SM {i}', ha='center', fontweight='bold', fontsize=10)
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_title('Mean Shape Contours by Cluster', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved mean contours plot to {output_path}")
    plt.close()


def plot_dendrogram_from_vampire_model(
    model_path: str,
    output_path: str,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot hierarchical clustering dendrogram from a VAMPIRE model.
    
    Parameters
    ----------
    model_path : str
        Path to the VAMPIRE model pickle file
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int], optional
        Figure size (default: (8, 6))
        
    Examples
    --------
    >>> plot_dendrogram_from_vampire_model(
    ...     '/output/fold_1/model_vampire_kfold_*.pickle',
    ...     'dendrogram.png'
    ... )
    """
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    # Load model
    model = load_vampire_model(model_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot dendrogram using model's linkage matrix
    dendrogram(
        model.linkage_matrix,
        ax=ax,
        labels=[f'SM {i}' for i in range(model.n_clusters)],
        color_threshold=0,
        above_threshold_color='black'
    )
    
    ax.set_xlabel('Shape Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering of Shape Modes', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved dendrogram to {output_path}")
    plt.close()


def plot_vampire_model_dendrogram_and_contours(
    model_path: str,
    output_path: str,
    figsize: Tuple[int, int] = (16, 6),
    contour_scale: float = 3,
    colors: Optional[List] = None
) -> None:
    """
    Combined plot showing dendrogram and mean contours from a VAMPIRE model.
    
    This creates a publication-quality figure directly from a trained VAMPIRE
    model, showing hierarchical relationships and actual mean shapes.
    
    Parameters
    ----------
    model_path : str
        Path to the VAMPIRE model pickle file
    output_path : str
        Path to save the plot
    figsize : Tuple[int, int], optional
        Figure size (default: (16, 6))
    contour_scale : float, optional
        Scale of contour shapes (default: 3)
    colors : List, optional
        Colors for each cluster. If None, uses twilight colormap
        
    Examples
    --------
    >>> plot_vampire_model_dendrogram_and_contours(
    ...     model_path='/output/fold_1/model_vampire_kfold_*.pickle',
    ...     output_path='dendrogram_and_contours.png'
    ... )
    
    Notes
    -----
    This is the recommended way to visualize VAMPIRE k-fold results, as it
    uses the actual trained model data rather than reconstructing from images.
    """
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    # Load model
    model = load_vampire_model(model_path)
    
    mean_cluster_contours = model.mean_cluster_contours
    num_clusters = model.n_clusters
    num_points = model.n_points
    
    # Create figure with two panels
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)
    
    # Left panel: Dendrogram
    ax_dendro = fig.add_subplot(gs[0, 0])
    
    dendro = dendrogram(
        model.linkage_matrix,
        ax=ax_dendro,
        labels=[f'SM {i}' for i in range(num_clusters)],
        orientation='left',
        color_threshold=0,
        above_threshold_color='black'
    )
    
    ax_dendro.set_xlabel('Distance', fontsize=11, fontweight='bold')
    ax_dendro.set_ylabel('Shape Mode', fontsize=11, fontweight='bold')
    ax_dendro.set_title('Hierarchical\nClustering', fontsize=12, fontweight='bold')
    
    # Right panel: Mean contours in dendrogram order
    ax_contours = fig.add_subplot(gs[0, 1])
    
    # Get dendrogram order
    dendro_order = dendro['leaves']
    
    # Set colors
    if colors is None:
        colors = [plt.get_cmap('twilight')(i) 
                 for i in np.linspace(0.1, 0.9, num_clusters)]
    elif isinstance(colors, str):
        colors = [colors] * num_clusters
    
    # Plot contours in dendrogram order
    x_first = 5
    x_offset = 10
    
    for plot_idx, cluster_idx in enumerate(dendro_order):
        # Read contour coordinates
        x = mean_cluster_contours[cluster_idx, :num_points]
        y = mean_cluster_contours[cluster_idx, num_points:]
        
        # Form closed shape
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        
        # Place shape in right location
        x = x * contour_scale + x_first + x_offset * plot_idx
        y = y * contour_scale
        
        # Plot shape
        ax_contours.plot(x, y, color=colors[cluster_idx], alpha=1.0, lw=2)
        
        # Add cluster label
        ax_contours.text(
            x_first + x_offset * plot_idx, 
            -contour_scale * 1.5,
            f'SM {cluster_idx}', 
            ha='center', fontweight='bold', fontsize=10
        )
    
    ax_contours.axis('equal')
    ax_contours.axis('off')
    ax_contours.set_title('Mean Shape Contours', fontsize=12, fontweight='bold')
    
    plt.suptitle('Shape Mode Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved combined plot to {output_path}")
    plt.close()


def compare_contours_across_folds(
    model_paths: List[str],
    output_path: str,
    fold_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    contour_scale: float = 3,
    colors: Optional[List] = None,
    alpha: float = 0.7,
    lw: float = 2
) -> None:
    """
    Compare mean contours across multiple folds side-by-side.
    
    This function creates a grid showing how mean shape contours compare
    across different k-fold cross-validation folds. Useful for assessing
    consistency of learned shapes.
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to VAMPIRE model pickle files, one per fold
    output_path : str
        Path to save the comparison plot
    fold_names : List[str], optional
        Names for each fold (e.g., ['Fold 1', 'Fold 2']). If None, auto-generated.
    figsize : Tuple[int, int], optional
        Figure size. If None, calculated based on number of folds
    contour_scale : float, optional
        Scale of contour shapes (default: 3)
    colors : List, optional
        Colors for each cluster. If None, uses twilight colormap
    alpha : float, optional
        Transparency for overlapping contours (default: 0.7)
    lw : float, optional
        Line width (default: 2)
        
    Returns
    -------
    None
    
    Examples
    --------
    >>> model_paths = [
    ...     '/output/fold_1/model_*.pickle',
    ...     '/output/fold_2/model_*.pickle',
    ...     '/output/fold_3/model_*.pickle',
    ... ]
    >>> compare_contours_across_folds(
    ...     model_paths,
    ...     'contours_comparison.png'
    ... )
    
    Notes
    -----
    - All models must have the same number of clusters
    - Before alignment, clusters may represent different shapes
    - After alignment (using Hungarian algorithm), same cluster IDs should
      represent similar shapes across folds
    """
    import matplotlib.pyplot as plt
    
    n_folds = len(model_paths)
    
    # Generate fold names if not provided
    if fold_names is None:
        fold_names = [f'Fold {i+1}' for i in range(n_folds)]
    
    # Load all models
    logger.info(f"Loading {n_folds} VAMPIRE models...")
    models = [load_vampire_model(path) for path in model_paths]
    
    # Validate all models have same parameters
    num_clusters = models[0].n_clusters
    num_points = models[0].n_points
    
    for i, model in enumerate(models[1:], 1):
        if model.n_clusters != num_clusters:
            raise ValueError(
                f"Model {i} has {model.n_clusters} clusters, "
                f"but model 0 has {num_clusters} clusters. "
                f"All models must have the same number of clusters."
            )
    
    # Calculate figure size
    if figsize is None:
        figsize = (4 * n_folds, 3 * num_clusters)
    
    # Create figure
    fig, axes = plt.subplots(num_clusters, n_folds, figsize=figsize)
    
    # Handle single fold or single cluster cases
    if num_clusters == 1 and n_folds == 1:
        axes = np.array([[axes]])
    elif num_clusters == 1:
        axes = axes.reshape(1, -1)
    elif n_folds == 1:
        axes = axes.reshape(-1, 1)
    
    # Set colors
    if colors is None:
        colors = [plt.get_cmap('twilight')(i) 
                 for i in np.linspace(0.1, 0.9, num_clusters)]
    elif isinstance(colors, str):
        colors = [colors] * num_clusters
    
    # Plot each cluster for each fold
    for cluster_id in range(num_clusters):
        for fold_idx, model in enumerate(models):
            ax = axes[cluster_id, fold_idx]
            
            # Get contour for this cluster
            x = model.mean_cluster_contours[cluster_id, :num_points]
            y = model.mean_cluster_contours[cluster_id, num_points:]
            
            # Form closed shape
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            # Center and scale
            x_centered = (x - x.mean()) * contour_scale
            y_centered = (y - y.mean()) * contour_scale
            
            # Plot
            ax.plot(x_centered, y_centered, 
                   color=colors[cluster_id], alpha=alpha, lw=lw)
            ax.fill(x_centered, y_centered, 
                   color=colors[cluster_id], alpha=0.2)
            
            # Set title
            if cluster_id == 0:
                ax.set_title(fold_names[fold_idx], fontsize=11, fontweight='bold')
            
            # Add cluster label on left
            if fold_idx == 0:
                ax.set_ylabel(f'Shape Mode {cluster_id}', 
                            fontsize=10, fontweight='bold')
            
            ax.axis('equal')
            ax.axis('off')
    
    plt.suptitle('Mean Shape Contours Across K-Folds', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved fold comparison to {output_path}")
    plt.close()




def extract_contours_from_images(
    predictions_df: pd.DataFrame,
    image_dir: str,
    num_points: int = 50
) -> pd.DataFrame:
    """
    Extract normalized contours from images and add to predictions DataFrame.
    
    This function loads the original images, extracts cell contours, normalizes them,
    and adds them to the predictions DataFrame for use with VAMPIRE plotting functions.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions DataFrame with 'filename' and 'cluster_id' columns
    image_dir : str
        Directory containing the labeled images
    num_points : int, optional
        Number of points for contour representation (default: 50)
        
    Returns
    -------
    pd.DataFrame
        Updated predictions DataFrame with 'normalized_contour' column added
        
    Examples
    --------
    >>> predictions_with_contours = extract_contours_from_images(
    ...     predictions_df, 
    ...     '/path/to/images',
    ...     num_points=50
    ... )
    """
    from skimage import io, measure
    from skimage.measure import find_contours
    import numpy as np
    
    image_dir = Path(image_dir)
    
    logger.info("Extracting contours from images...")
    
    contours_list = []
    valid_indices = []
    
    for idx, row in predictions_df.iterrows():
        try:
            # Get filename
            if 'filename' not in row:
                continue
            
            filename = Path(row['filename'])
            
            # Try multiple possible paths
            possible_paths = [
                filename,
                image_dir / filename.name,
                image_dir / filename,
            ]
            
            img_path = None
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            
            if img_path is None:
                continue
            
            # Load image
            if img_path.suffix == '.npy':
                img = np.load(img_path)
            else:
                img = io.imread(img_path)
            
            # Convert to labeled image if needed
            if img.dtype == bool:
                labeled = measure.label(img)
            else:
                labeled = img
            
            # Extract cell regions
            for region in measure.regionprops(labeled):
                if region.area < 10:
                    continue
                
                # Get cell mask
                cell_mask = region.image
                
                # Find contour
                contours = find_contours(cell_mask, 0.5)
                if len(contours) == 0:
                    continue
                
                # Use the longest contour
                contour = max(contours, key=len)
                
                # Normalize contour to num_points
                if len(contour) > num_points:
                    # Downsample
                    indices = np.linspace(0, len(contour)-1, num_points, dtype=int)
                    contour = contour[indices]
                elif len(contour) < num_points:
                    # Upsample by interpolation
                    old_indices = np.arange(len(contour))
                    new_indices = np.linspace(0, len(contour)-1, num_points)
                    contour_x = np.interp(new_indices, old_indices, contour[:, 0])
                    contour_y = np.interp(new_indices, old_indices, contour[:, 1])
                    contour = np.column_stack([contour_x, contour_y])
                
                # Normalize to unit circle (VAMPIRE standard)
                centroid = contour.mean(axis=0)
                contour_centered = contour - centroid
                max_dist = np.max(np.linalg.norm(contour_centered, axis=1))
                if max_dist > 0:
                    contour_normalized = contour_centered / max_dist
                else:
                    contour_normalized = contour_centered
                
                # Flatten to [x1, x2, ..., xn, y1, y2, ..., yn] format (VAMPIRE format)
                contour_flat = np.concatenate([
                    contour_normalized[:, 1],  # x coordinates
                    contour_normalized[:, 0]   # y coordinates
                ])
                
                contours_list.append(contour_flat)
                valid_indices.append(idx)
                break  # Only take first cell per image
                
        except Exception as e:
            logger.debug(f"Error extracting contour from {filename}: {e}")
            continue
    
    logger.info(f"Extracted {len(contours_list)} contours")
    
    # Add contours to DataFrame
    predictions_with_contours = predictions_df.copy()
    predictions_with_contours['normalized_contour'] = None
    predictions_with_contours['normalized_contour'] = predictions_with_contours['normalized_contour'].astype(object)
    
    for i, idx in enumerate(valid_indices):
        predictions_with_contours.at[idx, 'normalized_contour'] = contours_list[i]
    
    # Remove rows without contours
    predictions_with_contours = predictions_with_contours[
        predictions_with_contours['normalized_contour'].notna()
    ].reset_index(drop=True)
    
    logger.info(f"Predictions with contours: {len(predictions_with_contours)} rows")
    
    return predictions_with_contours


def plot_vampire_contours(
    predictions_df: pd.DataFrame,
    output_path: str,
    num_points: int = 50,
    num_clusters: int = 5,
    contour_scale: float = 3,
    fig_size: Tuple[int, int] = (12, 3),
    colors: Optional[List] = None,
    alpha: float = 1.0,
    lw: float = 2
) -> None:
    """
    Plot mean contours using VAMPIRE's plotting style.
    
    This function replicates VAMPIRE's plot_contours functionality to show
    mean shape contours for each cluster in a horizontal layout.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with 'cluster_id' and 'normalized_contour' columns
    output_path : str
        Path to save the plot
    num_points : int, optional
        Number of points in contours (default: 50)
    num_clusters : int, optional
        Number of clusters (default: 5)
    contour_scale : float, optional
        Scale of the contour shapes (default: 3)
    fig_size : Tuple[int, int], optional
        Figure size (default: (12, 3))
    colors : List, optional
        Colors for each cluster. If None, uses twilight colormap
    alpha : float, optional
        Alpha transparency (default: 1.0)
    lw : float, optional
        Line width (default: 2)
        
    Examples
    --------
    >>> # First extract contours
    >>> preds_with_contours = extract_contours_from_images(predictions_df, image_dir)
    >>> # Then plot
    >>> plot_vampire_contours(preds_with_contours, 'mean_contours.png')
    
    Notes
    -----
    Requires 'normalized_contour' column in predictions_df. Use
    extract_contours_from_images() to add this column.
    """
    import matplotlib.pyplot as plt
    
    # Check if normalized_contour column exists
    if 'normalized_contour' not in predictions_df.columns:
        raise ValueError(
            "predictions_df must have 'normalized_contour' column. "
            "Use extract_contours_from_images() first."
        )
    
    # Compute mean contours for each cluster
    mean_cluster_contours = []
    
    for cluster_id in sorted(predictions_df['cluster_id'].unique()):
        cluster_contours = predictions_df[
            predictions_df['cluster_id'] == cluster_id
        ]['normalized_contour'].tolist()
        
        # Stack all contours for this cluster
        cluster_contours_array = np.vstack(cluster_contours)
        
        # Compute mean
        mean_contour = cluster_contours_array.mean(axis=0)
        mean_cluster_contours.append(mean_contour)
    
    mean_cluster_contours = np.array(mean_cluster_contours)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set colors
    if colors is None:
        colors = [plt.get_cmap('twilight')(i) 
                 for i in np.linspace(0.1, 0.9, num_clusters)]
    elif isinstance(colors, str):
        colors = [colors] * num_clusters
    
    # Plot parameters
    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours
    
    for i in range(len(mean_cluster_contours)):
        # Read contour coordinates (VAMPIRE format: [x1...xn, y1...yn])
        x = mean_cluster_contours[i, :num_points]
        y = mean_cluster_contours[i, num_points:]
        
        # Form closed shape
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        
        # Place shape in right location
        x = x * contour_scale + x_first + x_offset * i
        y = y * contour_scale
        
        # Plot shape
        ax.plot(x, y, color=colors[i], alpha=alpha, lw=lw, label=f'SM {i}')
        
        # Add cluster label below
        ax.text(x_first + x_offset * i, -contour_scale * 1.5, 
               f'SM {i}', ha='center', fontweight='bold', fontsize=10)
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_title('Mean Shape Contours by Cluster', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved mean contours plot to {output_path}")
    plt.close()


def plot_dendrogram_and_shapes_combined(
    image_dir: str,
    predictions_df: pd.DataFrame,
    feature_columns: List[str],
    output_path: str,
    method: str = 'ward',
    figsize: Tuple[int, int] = (16, 10),
    max_cells_per_cluster: int = 30,
    shape_size: int = 50
) -> None:
    """
    Combined plot showing dendrogram and mean shape contours.
    
    Creates a publication-quality figure with hierarchical clustering
    dendrogram on the left and mean shape contours on the right.
    
    Parameters
    ----------
    image_dir : str
        Directory containing the original labeled images
    predictions_df : pd.DataFrame
        Predictions with cluster_id and feature columns
    feature_columns : List[str]
        Morphological features for clustering
    output_path : str
        Path to save the plot
    method : str, optional
        Linkage method (default: 'ward')
    figsize : Tuple[int, int], optional
        Figure size (default: (16, 10))
    max_cells_per_cluster : int, optional
        Maximum cells to average per cluster (default: 30)
    shape_size : int, optional
        Size to normalize shapes to (default: 50)
        
    Examples
    --------
    >>> plot_dendrogram_and_shapes_combined(
    ...     '/path/to/images',
    ...     predictions_df,
    ...     feature_columns,
    ...     'dendrogram_and_shapes.png'
    ... )
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    from skimage import io, measure
    from skimage.transform import resize
    
    image_dir = Path(image_dir)
    
    # Compute cluster centroids
    cluster_centroids = predictions_df.groupby('cluster_id')[feature_columns].mean()
    cluster_ids = cluster_centroids.index.tolist()
    n_clusters = len(cluster_ids)
    
    # Standardize features
    scaler = StandardScaler()
    centroids_scaled = scaler.fit_transform(cluster_centroids)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(centroids_scaled, method=method, metric='euclidean')
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)
    
    # Left panel: Dendrogram
    ax_dendro = fig.add_subplot(gs[0, 0])
    
    dendro = dendrogram(
        linkage_matrix,
        labels=[f'SM{i}' for i in cluster_ids],
        ax=ax_dendro,
        orientation='left',
        color_threshold=0,
        above_threshold_color='black'
    )
    
    ax_dendro.set_xlabel('Distance', fontsize=11, fontweight='bold')
    ax_dendro.set_ylabel('Shape Mode', fontsize=11, fontweight='bold')
    ax_dendro.set_title('Hierarchical\nClustering', fontsize=12, fontweight='bold')
    
    # Right panel: Mean shapes arranged by dendrogram order
    # Get dendrogram order
    dendro_order = dendro['leaves']
    ordered_cluster_ids = [cluster_ids[i] for i in dendro_order]
    
    # Create subplots for shapes
    n_cols = min(5, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    gs_right = gs[0, 1].subgridspec(n_rows, n_cols, wspace=0.1, hspace=0.3)
    
    logger.info("Extracting mean shapes in dendrogram order...")
    
    # Extract mean shapes for each cluster
    mean_shapes = {}
    for cluster_id in cluster_ids:
        logger.info(f"Processing cluster {cluster_id}...")
        
        # Get cells from this cluster
        cluster_cells = predictions_df[predictions_df['cluster_id'] == cluster_id]
        
        # Sample if too many cells
        if len(cluster_cells) > max_cells_per_cluster:
            cluster_cells = cluster_cells.sample(n=max_cells_per_cluster, random_state=42)
        
        # Collect cell masks
        cell_masks = []
        
        for idx, row in cluster_cells.iterrows():
            try:
                # Get filename
                if 'filename' in row:
                    filename = Path(row['filename'])
                else:
                    continue
                
                # Try multiple possible paths
                possible_paths = [
                    filename,  # Absolute path
                    image_dir / filename.name,  # Just filename in image_dir
                    image_dir / filename,  # Relative path in image_dir
                ]
                
                img_path = None
                for p in possible_paths:
                    if p.exists():
                        img_path = p
                        break
                
                if img_path is None:
                    continue
                
                # Load image
                if img_path.suffix == '.npy':
                    img = np.load(img_path)
                else:
                    img = io.imread(img_path)
                
                # Convert to labeled image if needed
                if img.dtype == bool:
                    labeled = measure.label(img)
                else:
                    labeled = img
                
                # Extract all cell regions from this image
                for region in measure.regionprops(labeled):
                    if region.area < 10:  # Skip very small regions
                        continue
                    
                    # Get the cell mask
                    cell_mask = region.image.astype(float)
                    
                    # Normalize size
                    cell_mask_resized = resize(
                        cell_mask,
                        (shape_size, shape_size),
                        anti_aliasing=True,
                        preserve_range=True
                    )
                    
                    cell_masks.append(cell_mask_resized > 0.5)
                    
            except Exception as e:
                logger.debug(f"Error processing {filename}: {e}")
                continue
        
        # Compute mean shape
        if len(cell_masks) > 0:
            mean_shape = np.mean(np.array(cell_masks), axis=0)
            mean_shapes[cluster_id] = mean_shape
            logger.info(f"  Cluster {cluster_id}: averaged {len(cell_masks)} cells")
        else:
            mean_shapes[cluster_id] = None
            logger.warning(f"  Cluster {cluster_id}: no cells found")
    
    # Plot shapes in dendrogram order
    for plot_idx, cluster_id in enumerate(ordered_cluster_ids):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = fig.add_subplot(gs_right[row, col])
        
        cluster_cells = predictions_df[predictions_df['cluster_id'] == cluster_id]
        
        if mean_shapes[cluster_id] is not None:
            mean_shape = mean_shapes[cluster_id]
            
            # Plot mean shape as grayscale
            ax.imshow(mean_shape, cmap='gray', interpolation='bilinear')
            
            # Overlay contour
            contour_levels = [0.3, 0.5, 0.7]
            colors = ['blue', 'red', 'yellow']
            for level, color in zip(contour_levels, colors):
                ax.contour(mean_shape, levels=[level], colors=color, 
                          linewidths=1.5, alpha=0.8)
            
            ax.set_title(f'SM {cluster_id}\n(n={len(cluster_cells)})', 
                        fontsize=9, fontweight='bold')
        else:
            # No cells found
            ax.text(0.5, 0.5, f'SM {cluster_id}\nNo cells', 
                   ha='center', va='center', fontsize=10,
                   transform=ax.transAxes)
            ax.set_title(f'SM {cluster_id}\n(n={len(cluster_cells)})', 
                        fontsize=9, fontweight='bold')
        
        ax.axis('off')
    
    plt.suptitle('Shape Mode Dendrogram and Mean Shapes', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved combined dendrogram and shapes to {output_path}")
    plt.close()


def create_summary_report(
    predictions_df: pd.DataFrame,
    feature_columns: List[str],
    output_dir: str
) -> None:
    """
    Create a comprehensive summary report with multiple plots.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Combined predictions from all folds
    feature_columns : List[str]
        Morphological feature columns
    output_dir : str
        Directory to save all plots
        
    Examples
    --------
    >>> create_summary_report(combined_df, feature_cols, 'summary_plots/')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating summary report...")
    
    # 1. Overall cluster distribution
    plot_cluster_distribution(
        predictions_df,
        output_dir / 'cluster_distribution_overall.png',
        title='Overall Shape Mode Distribution'
    )
    
    # 2. Cluster distribution by fold
    if 'fold' in predictions_df.columns:
        plot_cluster_distribution(
            predictions_df,
            output_dir / 'cluster_distribution_by_fold.png',
            title='Shape Mode Distribution by Fold',
            by_fold=True
        )
    
    # 3. Feature distributions
    plot_feature_distributions_by_cluster(
        predictions_df,
        feature_columns,
        output_dir / 'feature_distributions'
    )
    
    # 4. PCA visualization
    plot_pca_clusters(
        predictions_df,
        feature_columns,
        output_dir / 'pca_clusters.png'
    )
    
    logger.info(f"Summary report created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Plotting utilities for VAMPIRE k-fold cross-validation")
    print("Import this module and use the plotting functions with your data")