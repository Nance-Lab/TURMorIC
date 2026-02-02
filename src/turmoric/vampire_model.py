"""
Enhanced VAMPIRE model training with k-fold cross-validation.

This module extends the VampireModelTrainer class to support k-fold cross-validation
for robust model evaluation and full-dataset shape mode prediction.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import pandas as pd
import vampire
import logging
import random
from collections import defaultdict
from grouping_strategies import GroupingStrategy, UnderscorePositionStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VampireModelTrainer:
    """
    A class to handle VAMPIRE model training and application for brain image analysis.
    
    Supports standard train/test splitting and k-fold cross-validation for robust
    model evaluation and full-dataset shape mode predictions.
    """

    def __init__(
        self, 
        base_path: str, 
        treatments: List[str], 
        groups: List[str],
        grouping_strategy: Optional[GroupingStrategy] = None,
        split_function: Optional[Callable] = None
    ):
        """
        Initialize the VAMPIRE model trainer.

        Parameters
        ----------
        base_path : str
            Base directory path for image data
        treatments : list of str
            List of treatment conditions
        groups : list of str
            List of experimental groups
        grouping_strategy : GroupingStrategy, optional
            Strategy for grouping files to prevent data leakage.
            If None, defaults to UnderscorePositionStrategy(2)
            which extracts the 3rd underscore-separated token.
        split_function : callable, optional
            Custom function for splitting group_ids into train/test sets.
            Function signature: split_function(group_ids, k, random_seed) -> List[Tuple[List[str], List[str]]]
            Returns list of (train_groups, test_groups) tuples.
            If None, uses default k-fold split.
            
        Raises
        ------
        FileNotFoundError
            If base_path does not exist
            
        Examples
        --------
        >>> # Default: extract slice ID from position 2, k-fold split
        >>> trainer = VampireModelTrainer(base_path, treatments, groups)
        
        >>> # Custom grouping strategy
        >>> from grouping_strategies import UnderscorePositionStrategy
        >>> strategy = UnderscorePositionStrategy(1)
        >>> trainer = VampireModelTrainer(base_path, treatments, groups, strategy)
        
        >>> # Custom split function
        >>> def my_split(group_ids, k, random_seed):
        ...     # Your custom logic
        ...     return [(train_groups, test_groups)]
        >>> trainer = VampireModelTrainer(base_path, treatments, groups, 
        ...                                grouping_strategy=strategy,
        ...                                split_function=my_split)
        """
        self.base_path = Path(base_path)
        self.treatments = treatments
        self.groups = groups
        self.model_path: Optional[Path] = None
        
        # Set grouping strategy
        if grouping_strategy is None:
            logger.info("No grouping strategy specified, using default: position 2")
            self.grouping_strategy = UnderscorePositionStrategy(2)
        else:
            self.grouping_strategy = grouping_strategy
        
        logger.info(f"Grouping strategy: {self.grouping_strategy.get_description()}")
        
        # Set split function
        if split_function is None:
            logger.info("No split function specified, using default: k-fold")
            self.split_function = self._default_kfold_split
        else:
            logger.info("Using custom split function")
            self.split_function = split_function

        # Validate base path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

    def extract_features(self, image_set_path: Path) -> None:
        """
        Extract features from images using VAMPIRE.

        Parameters
        ----------
        image_set_path : Path
            Path to the image dataset
            
        Raises
        ------
        Exception
            If feature extraction fails
        """
        logger.info(f"Extracting features from: {image_set_path}")
        
        try:
            vampire.extraction.extract_properties(str(image_set_path))
            logger.info("Feature extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise
    
    def train_model(
        self, 
        image_set_path: Path, 
        model_name: str = 'li', 
        num_points: int = 50, 
        num_clusters: int = 5
    ) -> Path:
        """
        Train a VAMPIRE model on the extracted features.
        
        Parameters
        ----------
        image_set_path : Path
            Path to the training image dataset
        model_name : str, optional
            Name identifier for the model (default: 'li')
        num_points : int, optional
            Number of points for model training (default: 50)
        num_clusters : int, optional
            Number of clusters for model training (default: 5)
            
        Returns
        -------
        model_path : Path
            Path to the trained model file
            
        Raises
        ------
        FileNotFoundError
            If no model file is found after training
        Exception
            If model training fails
        """
        logger.info(f"Training VAMPIRE model: {model_name}")
        
        build_info_df = pd.DataFrame({
            'img_set_path': [str(image_set_path)],
            'output_path': [str(image_set_path)],
            'model_name': [model_name],
            'num_points': [num_points],
            'num_clusters': [num_clusters],
            'num_pc': [np.nan]
        })
        
        try:
            vampire.quickstart.fit_models(build_info_df)
            
            # Find the generated model file
            model_pattern = f"model_{model_name}_({num_points}_{num_clusters}_*)__.pickle"
            model_files = list(image_set_path.glob(model_pattern))
            
            if not model_files:
                raise FileNotFoundError(
                    f"No model file found matching pattern: {model_pattern}"
                )
            
            self.model_path = model_files[0]
            logger.info(f"Model trained successfully: {self.model_path}")
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def create_apply_dataframe(
        self, 
        test_base_path: Path, 
        model_path: Path
    ) -> pd.DataFrame:
        """
        Create a DataFrame for applying the trained model to test datasets.
        
        Parameters
        ----------
        test_base_path : Path
            Base path for test datasets
        model_path : Path
            Path to the trained model
            
        Returns
        -------
        apply_df : pd.DataFrame
            DataFrame with application configuration containing columns:
            - img_set_path
            - model_path
            - output_path
            - img_set_name
            
        Raises
        ------
        ValueError
            If no valid test datasets are found
        """
        apply_data = []
        
        for group in self.groups:
            for treatment in self.treatments:
                img_set_path = test_base_path / group / treatment
                
                # Check if path exists before adding to dataframe
                if img_set_path.exists():
                    apply_data.append({
                        'img_set_path': str(img_set_path),
                        'model_path': str(model_path),
                        'output_path': str(img_set_path),
                        'img_set_name': treatment
                    })
                    logger.debug(f"Added to apply list: {treatment}")
                else:
                    logger.warning(f"Path does not exist, skipping: {img_set_path}")
        
        if not apply_data:
            raise ValueError("No valid test datasets found")
        
        return pd.DataFrame(apply_data)
    
    def apply_model(self, test_base_path: Path) -> None:
        """
        Apply the trained model to test datasets.
        
        Parameters
        ----------
        test_base_path : Path
            Base path for test datasets
            
        Raises
        ------
        ValueError
            If model has not been trained yet
        Exception
            If model application fails
        """
        if self.model_path is None:
            raise ValueError("Model must be trained before applying")
        
        logger.info(f"Applying model to test datasets in: {test_base_path}")
        
        try:
            apply_info_df = self.create_apply_dataframe(test_base_path, self.model_path)
            logger.info(f"Applying model to {len(apply_info_df)} datasets")
            
            vampire.quickstart.transform_datasets(apply_info_df)
            logger.info("Model application completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model application: {e}")
            raise
    
    def run_full_pipeline(
        self, 
        training_subpath: str = "training/vampire_data", 
        testing_subpath: str = "testing/vampire_data"
    ) -> None:
        """
        Run the complete training and application pipeline.
        
        Parameters
        ----------
        training_subpath : str, optional
            Relative path to training data (default: "training/vampire_data")
        testing_subpath : str, optional
            Relative path to testing data (default: "testing/vampire_data")
        """
        train_path = self.base_path / training_subpath
        test_path = self.base_path / testing_subpath
        
        logger.info("Starting VAMPIRE model pipeline")
        logger.info(f"Training path: {train_path}")
        logger.info(f"Testing path: {test_path}")
        
        # Step 1: Extract features
        self.extract_features(train_path)
        
        # Step 2: Train model
        self.train_model(train_path)
        
        # Step 3: Apply model to test data
        self.apply_model(test_path)
        
        logger.info("Pipeline completed successfully")

    # ==================== K-FOLD CROSS-VALIDATION METHODS ====================
    
    def _default_kfold_split(
        self,
        group_ids: List[str],
        k: int,
        random_seed: Optional[int] = None
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Default k-fold splitting function.
        
        Parameters
        ----------
        group_ids : list of str
            List of all group identifiers
        k : int
            Number of folds
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        splits : list of tuple
            List of (train_groups, test_groups) tuples, one per fold
        """
        if len(group_ids) < k:
            logger.warning(
                f"Number of groups ({len(group_ids)}) is less than k ({k}). "
                f"Some folds will be empty."
            )
        
        # Shuffle group IDs for random fold assignment
        shuffled_groups = group_ids.copy()
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(shuffled_groups)
        
        # Split groups into k folds
        fold_size = len(shuffled_groups) // k
        fold_groups = []
        for i in range(k):
            start_idx = i * fold_size
            # Last fold gets any remainder
            end_idx = start_idx + fold_size if i < k - 1 else len(shuffled_groups)
            fold_groups.append(shuffled_groups[start_idx:end_idx])
        
        # Create train/test splits
        splits = []
        for i in range(k):
            test_groups = fold_groups[i]
            train_groups = [g for j, fold in enumerate(fold_groups) 
                           if j != i for g in fold]
            splits.append((train_groups, test_groups))
        
        return splits
    
    def get_group_ids_from_path(self, data_path: Path) -> List[str]:
        """
        Extract unique group identifiers from filenames using the grouping strategy.
        
        Uses the configured grouping_strategy to determine which files should be
        grouped together to prevent data leakage.
        
        Parameters
        ----------
        data_path : Path
            Path containing image files
            
        Returns
        -------
        group_ids : list of str
            List of unique group identifiers
            
        Notes
        -----
        The grouping strategy is set during initialization and determines how
        files are grouped (by slice, animal, subject, etc.).
        """
        group_ids = set()
        errors = []
        
        for group in self.groups:
            for treatment in self.treatments:
                treatment_path = data_path / group / treatment
                
                if not treatment_path.exists():
                    continue
                
                for file in treatment_path.iterdir():
                    if file.is_file() and not file.name.startswith('.'):
                        try:
                            group_id = self.grouping_strategy.extract_group_id(file)
                            group_ids.add(group_id)
                        except Exception as e:
                            errors.append((file.name, str(e)))
        
        if errors:
            logger.warning(f"Could not extract group IDs from {len(errors)} files")
            for filename, error in errors[:3]:  # Show first 3
                logger.warning(f"  {filename}: {error}")
            if len(errors) > 3:
                logger.warning(f"  ... and {len(errors) - 3} more")
        
        return sorted(list(group_ids))
    
    def get_files_for_groups(
        self, 
        data_path: Path, 
        group_ids: List[str]
    ) -> Dict[Tuple[str, str], List[Path]]:
        """
        Get all file paths for specified group IDs, organized by experimental group/treatment.
        
        Parameters
        ----------
        data_path : Path
            Base path containing data
        group_ids : list of str
            List of group IDs to include (e.g., slice IDs, animal IDs)
            
        Returns
        -------
        files_dict : dict
            Dictionary mapping (group, treatment) -> list of file paths
        """
        files_dict = defaultdict(list)
        
        for group in self.groups:
            for treatment in self.treatments:
                treatment_path = data_path / group / treatment
                
                if not treatment_path.exists():
                    continue
                
                for file in treatment_path.iterdir():
                    if file.is_file() and not file.name.startswith('.'):
                        try:
                            file_group_id = self.grouping_strategy.extract_group_id(file)
                            if file_group_id in group_ids:
                                files_dict[(group, treatment)].append(file)
                        except Exception as e:
                            logger.debug(f"Skipping {file.name}: {e}")
                            continue
        
        return files_dict
    
    def create_fold_directory(
        self, 
        base_dir: Path, 
        fold_name: str, 
        files_dict: Dict[Tuple[str, str], List[Path]]
    ) -> Path:
        """
        Create a temporary directory structure for a fold.
        
        Parameters
        ----------
        base_dir : Path
            Base directory for fold data
        fold_name : str
            Name of the fold (e.g., 'fold_0_train')
        files_dict : dict
            Dictionary mapping (group, treatment) -> list of file paths
            
        Returns
        -------
        fold_path : Path
            Path to created fold directory
        """
        import shutil
        
        fold_path = base_dir / fold_name
        fold_path.mkdir(parents=True, exist_ok=True)
        
        for (group, treatment), files in files_dict.items():
            target_dir = fold_path / group / treatment
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                target_file = target_dir / file.name
                shutil.copy2(file, target_file)
        
        return fold_path
    
    def k_fold_cross_validation(
        self,
        data_path: Path,
        k: int = 5,
        model_name: str = 'li',
        num_points: int = 50,
        num_clusters: int = 5,
        random_seed: int = 42,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Perform k-fold cross-validation on the dataset.
        
        This method splits data by group ID (using the configured grouping_strategy
        and split_function) to prevent data leakage, trains k models, and returns 
        predictions for the entire dataset where each sample is predicted by a model 
        that did not train on it.
        
        Parameters
        ----------
        data_path : Path
            Path to directory containing all data (organized by group/treatment)
        k : int, optional
            Number of folds (default: 5)
        model_name : str, optional
            Name identifier for models (default: 'li')
        num_points : int, optional
            Number of points for VAMPIRE training (default: 50)
        num_clusters : int, optional
            Number of shape mode clusters (default: 5)
        random_seed : int, optional
            Random seed for reproducibility (default: 42)
        output_dir : Path, optional
            Directory to save fold results. If None, uses temporary directory.
            
        Returns
        -------
        predictions_df : pd.DataFrame
            Combined predictions from all folds with columns:
            - filename: Source file path
            - shape_mode: Predicted shape mode
            - group: Experimental group
            - treatment: Treatment condition
            - fold: Fold number (0 to k-1)
            - Additional columns from VAMPIRE output
            
        Examples
        --------
        >>> # Default k-fold split
        >>> trainer = VampireModelTrainer(base_path, treatments, groups)
        >>> predictions = trainer.k_fold_cross_validation(
        ...     data_path=Path("/data/all_images"),
        ...     k=5
        ... )
        
        >>> # Custom split function
        >>> def leave_one_out(group_ids, k, random_seed):
        ...     '''Leave-one-out cross-validation'''
        ...     splits = []
        ...     for i, test_id in enumerate(group_ids):
        ...         train = [g for g in group_ids if g != test_id]
        ...         test = [test_id]
        ...         splits.append((train, test))
        ...     return splits
        >>> 
        >>> trainer = VampireModelTrainer(base_path, treatments, groups,
        ...                                split_function=leave_one_out)
        >>> predictions = trainer.k_fold_cross_validation(
        ...     data_path=Path("/data/all_images"),
        ...     k=len(group_ids)  # k is passed but may be ignored by custom function
        ... )
        
        Notes
        -----
        - Data is split at the group level (slice/animal/subject) to prevent leakage
        - Each group appears in exactly one test set across all folds
        - The grouping_strategy is set during initialization
        - The split_function can be customized for different CV strategies
        - Temporary directories are created for each fold's train/test split
        - Set random_seed for reproducible fold assignments
        """
        import tempfile
        import shutil
        
        logger.info(f"Starting cross-validation with k={k}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Grouping strategy: {self.grouping_strategy.get_description()}")
        
        # Get all unique group IDs using the grouping strategy
        all_group_ids = self.get_group_ids_from_path(data_path)
        logger.info(f"Found {len(all_group_ids)} unique groups")
        
        # Use the split function to create train/test splits
        logger.info("Applying split function...")
        splits = self.split_function(all_group_ids, k, random_seed)
        n_splits = len(splits)
        logger.info(f"Split function generated {n_splits} train/test splits")
        
        # Calculate split sizes for logging
        split_sizes = [(len(train), len(test)) for train, test in splits]
        logger.info(f"Split sizes (train, test): {split_sizes}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix='vampire_cv_'))
            logger.info(f"Using temporary directory: {output_dir}")
            cleanup_temp = True
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        
        all_predictions = []
        
        try:
            # Perform cross-validation using splits from split_function
            for fold_idx, (train_groups, test_groups) in enumerate(splits):
                logger.info(f"\n{'='*60}")
                logger.info(f"Split {fold_idx + 1}/{n_splits}")
                logger.info(f"{'='*60}")
                logger.info(f"Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
                
                # Get files for train and test sets
                train_files = self.get_files_for_groups(data_path, train_groups)
                test_files = self.get_files_for_groups(data_path, test_groups)
                
                # Create fold directories
                train_path = self.create_fold_directory(
                    output_dir, f'split_{fold_idx}_train', train_files
                )
                test_path = self.create_fold_directory(
                    output_dir, f'split_{fold_idx}_test', test_files
                )
                
                # Extract features for training data
                logger.info("Extracting features from training data...")
                self.extract_features(train_path)
                
                # Train model on this fold
                logger.info("Training model...")
                model_path = self.train_model(
                    train_path,
                    model_name=f"{model_name}_split{fold_idx}",
                    num_points=num_points,
                    num_clusters=num_clusters
                )
                
                # Apply model to test data
                logger.info("Applying model to test data...")
                apply_df = self.create_apply_dataframe(test_path, model_path)
                vampire.quickstart.transform_datasets(apply_df)
                
                # Collect predictions from test set
                fold_predictions = self.collect_fold_predictions(
                    test_path, 
                    fold_idx
                )
                
                if len(fold_predictions) > 0:
                    all_predictions.append(fold_predictions)
                    logger.info(f"Collected {len(fold_predictions)} predictions from split {fold_idx}")
                else:
                    logger.warning(f"No predictions collected from split {fold_idx}")
        
        finally:
            # Cleanup temporary directory if needed
            if cleanup_temp and output_dir.exists():
                logger.info(f"Cleaning up temporary directory: {output_dir}")
                shutil.rmtree(output_dir)
        
        # Combine all predictions
        if not all_predictions:
            logger.error("No predictions collected from any split")
            return pd.DataFrame()
        
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-validation complete")
        logger.info(f"Total predictions: {len(combined_predictions)}")
        logger.info(f"Shape mode distribution:\n{combined_predictions['shape_mode'].value_counts().sort_index()}")
        logger.info(f"{'='*60}\n")
        
        return combined_predictions
    
    def collect_fold_predictions(
        self, 
        test_path: Path, 
        fold_idx: int
    ) -> pd.DataFrame:
        """
        Collect VAMPIRE predictions from a test fold directory.
        
        Parameters
        ----------
        test_path : Path
            Path to test fold directory
        fold_idx : int
            Index of the current fold
            
        Returns
        -------
        predictions_df : pd.DataFrame
            DataFrame containing predictions with metadata
            
        Notes
        -----
        This function looks for VAMPIRE output CSV files (typically named
        *_predictions.csv or similar) in the test directory structure.
        Adjust the file pattern based on your VAMPIRE output format.
        """
        predictions_list = []
        
        # Look for VAMPIRE prediction CSV files
        prediction_files = list(test_path.rglob("*_predictions.csv"))
        
        if not prediction_files:
            # Try alternative patterns
            prediction_files = list(test_path.rglob("*predictions*.csv"))
        
        if not prediction_files:
            logger.warning(f"No prediction files found in {test_path}")
            return pd.DataFrame()
        
        for pred_file in prediction_files:
            try:
                df = pd.read_csv(pred_file)
                
                # Extract group and treatment from path
                # Assumes structure: test_path/group/treatment/file
                relative_path = pred_file.relative_to(test_path)
                path_parts = relative_path.parts
                
                if len(path_parts) >= 2:
                    group = path_parts[0]
                    treatment = path_parts[1]
                else:
                    group = "unknown"
                    treatment = "unknown"
                
                df['group'] = group
                df['treatment'] = treatment
                df['fold'] = fold_idx
                df['source_file'] = str(pred_file)
                
                predictions_list.append(df)
                
            except Exception as e:
                logger.error(f"Error reading {pred_file}: {e}")
                continue
        
        if predictions_list:
            return pd.concat(predictions_list, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def plot_kfold_results(
        self, 
        predictions_df: pd.DataFrame, 
        output_path: Optional[Path] = None
    ) -> None:
        """
        Create visualization of k-fold cross-validation results.
        
        Parameters
        ----------
        predictions_df : pd.DataFrame
            Combined predictions from k-fold CV
        output_path : Path, optional
            Path to save the plot. If None, displays plot.
            
        Notes
        -----
        Creates a multi-panel figure showing:
        1. Overall shape mode distribution
        2. Shape mode distribution by treatment
        3. Shape mode distribution by fold
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall distribution
        shape_mode_counts = predictions_df['shape_mode'].value_counts().sort_index()
        axes[0, 0].bar(shape_mode_counts.index, shape_mode_counts.values, color='steelblue')
        axes[0, 0].set_title('Overall Shape Mode Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Shape Mode')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Distribution by treatment
        treatment_dist = predictions_df.groupby(['treatment', 'shape_mode']).size().unstack(fill_value=0)
        treatment_dist.plot(kind='bar', ax=axes[0, 1], stacked=False)
        axes[0, 1].set_title('Shape Mode Distribution by Treatment', fontweight='bold')
        axes[0, 1].set_xlabel('Treatment')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Shape Mode', bbox_to_anchor=(1.05, 1))
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Distribution by fold
        fold_dist = predictions_df.groupby(['fold', 'shape_mode']).size().unstack(fill_value=0)
        fold_dist.plot(kind='bar', ax=axes[1, 0], stacked=False)
        axes[1, 0].set_title('Shape Mode Distribution by Fold', fontweight='bold')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Shape Mode')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Sample sizes by treatment and fold
        sample_sizes = predictions_df.groupby(['treatment', 'fold']).size().unstack(fill_value=0)
        sns.heatmap(sample_sizes, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Sample Sizes (Treatment × Fold)', fontweight='bold')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Treatment')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    """
    Example usage of k-fold cross-validation.
    """
    # Example configuration
    base_path = "/path/to/your/data"
    treatments = ["control", "treatment_A", "treatment_B"]
    groups = ["group_1", "group_2", "group_3"]
    
    # Initialize trainer
    trainer = VampireModelTrainer(
        base_path=base_path,
        treatments=treatments,
        groups=groups
    )
    
    # Run k-fold cross-validation
    data_path = Path(base_path) / "all_images"
    
    predictions = trainer.k_fold_cross_validation(
        data_path=data_path,
        k=5,
        num_clusters=5
    )
    
    # Save predictions
    predictions.to_csv('kfold_predictions.csv', index=False)
    
    # Plot results
    trainer.plot_kfold_results(predictions, output_path=Path('kfold_results.png'))
    
    # Analyze shape mode distribution
    print("\nShape Mode Distribution:")
    print(predictions['shape_mode'].value_counts().sort_index())
    
    print("\nBy Treatment:")
    print(predictions.groupby(['treatment', 'shape_mode']).size())