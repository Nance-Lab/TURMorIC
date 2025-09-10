from pathlib import Path
import click
import numpy as np
import pandas as pd





@click.command()

@click.argument('output_csv', type=click.Path(exists=False, path_type=Path))
def recursively_apply_regionprops(output_csv):

    csv_list = [
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/2R_control/cd11b/li_thresh/2R_control_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/2R_OGD/cd11b/li_thresh/2R_OGD_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/24R_control/cd11b/li_thresh/24R_control_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/24R_OGD/cd11b/li_thresh/24R_OGD_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/ORST/cd11b/li_thresh/ORST_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/OGD_only/cd11b/li_thresh/OGD_only_regionprops.csv",
        "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/30mR_control/cd11b/li_thresh/30mR_control_regionprops.csv"

    ]

    regionprops_dfs = []

    for csv in csv_list:
        df = pd.read_csv(csv)
        file_name = csv.split("/")[-1]
        treatment_1 = file_name.split("_")[0]
        treatment_2 = file_name.split("_")[1]

        if treatment_1 == "ORST":
            treatment = "ORST"
        else:
            treatment = f'{treatment_1}_{treatment_2}'

        df['treatment'] = treatment
        regionprops_dfs.append(df)
    
    regionprops_df = pd.concat(regionprops_dfs)

    regionprops_df['circularity'] = 4*np.pi*regionprops_df["area"]/regionprops_df.perimeter**2
    regionprops_df['aspect_ratio'] = regionprops_df.major_axis_length/regionprops_df.minor_axis_length
    regionprops_df.to_csv(output_csv, index=False)


# Example usage
if __name__ == "__main__":
    recursively_apply_regionprops()