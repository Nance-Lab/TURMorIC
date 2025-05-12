import os
from turmoric.utils import organize_files_without_leakage
import click
from pathlib import Path
import json

# Define your base directory and target directories for training and testing
base_dir = "/Users/nelsschimek/Documents/nancelab/Data/caffeine"
train_dir = "/Users/nelsschimek/Documents/nancelab/Data/caffeine/training"
test_dir = "/Users/nelsschimek/Documents/nancelab/Data/caffeine/testing"

# Define a list of subfolder names or patterns to look for
treatment_conditions = ["treatment_A", "treatment_B", "treatment_C",
                        "treatment_D", "treatment_E"]
groups = ["cortex", "hippocampus", "BG", "thalamus", "ScWM", "corpus_col"]

# Create training and testing directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


@click.command()
@click.argument('metadata_json', type=click.Path(exists=True,
                                                 readable=True,
                                                 path_type=Path))
def vampire_train_test_split(metadata_json):

    """
    Function to organize files into training and testing folders
    without slice leakage
    """

    vampire_metadata = json.loads(metadata_json)
    base_dir = vampire_metadata['base_dir']
    train_dir = vampire_metadata['train_dir']
    test_dir = vampire_metadata['test_dir']
    groups = vampire_metadata['groups']
    conditions = vampire_metadata['conditions']

    organize_files_without_leakage(base_dir=base_dir,
                                   train_dir=train_dir,
                                   test_dir=test_dir,
                                   groups=groups,
                                   conditions=conditions)
    return
