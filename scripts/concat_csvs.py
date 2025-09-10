import os
import pandas as pd
from pathlib import Path
import glob

def load_and_concatenate_csvs(root_directory, output_file=None):
    """
    Recursively search for CSV files, add treatment column based on subdirectory,
    and concatenate all CSVs into a single DataFrame.
    
    Args:
        root_directory (str): Root directory to search for CSV files
        output_file (str, optional): Path to save the concatenated CSV
    
    Returns:
        pandas.DataFrame: Concatenated DataFrame with treatment column
    """
    
    # Convert to Path object for easier handling
    root_path = Path(root_directory)
    
    if not root_path.exists():
        raise ValueError(f"Directory {root_directory} does not exist")
    
    # Find all CSV files recursively
    csv_files = list(root_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files")
    
    dataframes = []
    
    for csv_file in csv_files:
        try:
            # Load the CSV
            df = pd.read_csv(csv_file)
            
            # Get the immediate parent directory name as treatment
            treatment = csv_file.parent.name
            
            # If the CSV is directly in the root directory, use root dir name
            if csv_file.parent == root_path:
                treatment = root_path.name
            
            # Add treatment column
            df['treatment'] = treatment
            
            # Add source file info (optional, can be useful for debugging)
            df['source_file'] = str(csv_file.relative_to(root_path))
            
            dataframes.append(df)
            print(f"Loaded: {csv_file.name} with treatment '{treatment}' ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if not dataframes:
        print("No CSV files could be loaded successfully")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nSuccessfully concatenated {len(dataframes)} files")
        print(f"Final dataset shape: {combined_df.shape}")
        print(f"Treatment values: {combined_df['treatment'].unique()}")
        
        # Save to file if specified
        if output_file:
            combined_df.to_csv(output_file, index=False)
            print(f"Saved concatenated data to: {output_file}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error concatenating dataframes: {e}")
        return pd.DataFrame()

def load_csvs_with_custom_treatment(root_directory, treatment_mapping=None, output_file=None):
    """
    Alternative version that allows custom treatment mapping based on directory paths.
    
    Args:
        root_directory (str): Root directory to search for CSV files
        treatment_mapping (dict, optional): Custom mapping of directory names to treatment names
        output_file (str, optional): Path to save the concatenated CSV
    
    Returns:
        pandas.DataFrame: Concatenated DataFrame with treatment column
    """
    
    root_path = Path(root_directory)
    csv_files = list(root_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {root_directory}")
        return pd.DataFrame()
    
    dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Get treatment name
            dir_name = csv_file.parent.name
            
            # Apply custom mapping if provided
            if treatment_mapping and dir_name in treatment_mapping:
                treatment = treatment_mapping[dir_name]
            else:
                treatment = dir_name
            
            df['treatment'] = treatment
            df['source_file'] = str(csv_file.relative_to(root_path))
            
            dataframes.append(df)
            print(f"Loaded: {csv_file.name} with treatment '{treatment}'")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        if output_file:
            combined_df.to_csv(output_file, index=False)
        
        return combined_df
    
    return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Basic usage
    root_dir = "/Users/nelsschimek/Documents/nancelab/Data/colin_images/li_thresh"  # Replace with your directory path
    
    # Load and concatenate all CSVs
    df = load_and_concatenate_csvs(root_dir, output_file="concatenated_data.csv")
    
    # Display basic info about the result
    if not df.empty:
        print("\nDataFrame Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nTreatment value counts:")
        print(df['treatment'].value_counts())
    
    # Example with custom treatment mapping
    # custom_mapping = {
    #     "control_group": "Control",
    #     "treatment_a": "Treatment_A",
    #     "treatment_b": "Treatment_B"
    # }
    # 
    # df_custom = load_csvs_with_custom_treatment(
    #     root_dir, 
    #     treatment_mapping=custom_mapping,
    #     output_file="custom_concatenated_data.csv"
    # )