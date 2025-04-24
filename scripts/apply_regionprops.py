from pathlib import Path
from turmoric.cell_analysis import apply_regionprops_recursively
import click

props_list = ('area', 'bbox_area', 'centroid', 'convex_area',
              'eccentricity', 'equivalent_diameter', 'euler_number',
              'extent', 'filled_area', 'major_axis_length',
              'minor_axis_length', 'orientation', 'perimeter', 'solidity')


@click.command()
@click.argument('input_folder', type=click.Path(exists=True,
                                                readable=True, path_type=Path))
@click.argument('output_csv', type=click.Path(exists=False, path_type=Path))
def recursively_apply_regionprops(input_folder, output_csv):
    regionprops_df = apply_regionprops_recursively(input_folder, props_list)
    regionprops_df.to_csv(output_csv, index=False)


# Example usage
if __name__ == "__main__":
    recursively_apply_regionprops()
