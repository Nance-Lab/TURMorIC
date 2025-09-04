#!/usr/bin/env python
# coding: utf-8

"""

Note: this script was initially written by Claude AI 
and edited by Nels Schimek

VAMPIRE Model Training and Application Script

This script trains a VAMPIRE model on brain imaging data and applies it to test datasets.
It processes multiple treatment conditions and groups for microglia morphology analysis.

Usage:
    python train_vampire_model.py [--config config.yaml]
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import vampire


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VampireModelTrainer:
    """
    A class to handle VAMPIRE model training and application for brain image analysis.
    """
    
    def __init__(self, base_path: str, treatments: List[str], groups: List[str]):
        """
        Initialize the VAMPIRE model trainer.
        
        Args:
            base_path: Base directory path for image data
            treatments: List of treatment conditions
            groups: List of experimental groups
        """
        self.base_path = Path(base_path)
        self.treatments = treatments
        self.groups = groups
        self.model_path: Optional[Path] = None
        
        # Validate base path exists
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
    
    def extract_features(self, image_set_path: Path) -> None:
        """
        Extract features from images using VAMPIRE.
        
        Args:
            image_set_path: Path to the image dataset
        """
        logger.info(f"Extracting features from: {image_set_path}")
        
        try:
            vampire.extraction.extract_properties(str(image_set_path))
            logger.info("Feature extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            raise
    
    def train_model(self, image_set_path: Path, model_name: str = 'li', 
                   num_points: int = 50, num_clusters: int = 5) -> Path:
        """
        Train a VAMPIRE model on the extracted features.
        
        Args:
            image_set_path: Path to the training image dataset
            model_name: Name identifier for the model
            num_points: Number of points for model training
            num_clusters: Number of clusters for model training
            
        Returns:
            Path to the trained model file
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
                raise FileNotFoundError(f"No model file found matching pattern: {model_pattern}")
            
            self.model_path = model_files[0]
            logger.info(f"Model trained successfully: {self.model_path}")
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def create_apply_dataframe(self, test_base_path: Path, model_path: Path) -> pd.DataFrame:
        """
        Create a DataFrame for applying the trained model to test datasets.
        
        Args:
            test_base_path: Base path for test datasets
            model_path: Path to the trained model
            
        Returns:
            DataFrame with application configuration
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
        
        Args:
            test_base_path: Base path for test datasets
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
    
    def run_full_pipeline(self, training_subpath: str = "training/vampire_data", 
                         testing_subpath: str = "testing/vampire_data") -> None:
        """
        Run the complete training and application pipeline.
        
        Args:
            training_subpath: Relative path to training data
            testing_subpath: Relative path to testing data
        """
        train_path = Path.joinpath(self.base_path, training_subpath)
        test_path = Path.joinpath(self.base_path, testing_subpath)
        
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


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    config = {
        'base_path': "/Users/nelsschimek/Documents/nancelab/Data/tommy_data/li_thresh/converted_tiffs",
        'treatments': ["Left_HC", "Right_HC"],
        'groups': ["Blast_45_angle_5", "Blast_45_angle_6", "Blast_prone_7", "Blast_prone_8", "Blast_prone_9", "Sham_3"],
        'training_subpath': "training/",
        'testing_subpath': "testing/",
        'model_params': {
            'model_name': 'li',
            'num_points': 50,
            'num_clusters': 5
        }
    }
    
    # TODO: Add YAML config file support if needed
    if config_path:
        logger.warning("Config file loading not yet implemented, using defaults")
    
    return config


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='Train and apply VAMPIRE model for brain image analysis')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize trainer
        trainer = VampireModelTrainer(
            base_path=config['base_path'],
            treatments=config['treatments'],
            groups=config['groups']
        )
        
        # Run the pipeline
        trainer.run_full_pipeline(
            training_subpath=config['training_subpath'],
            testing_subpath=config['testing_subpath']
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
