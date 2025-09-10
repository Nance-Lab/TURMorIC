import click
from turmoric.apply_thresholds import apply_all_thresh
from pathlib import Path


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, readable=True,
                                                path_type=Path))
@click.argument('output_folder', type=click.Path(exists=False, path_type=Path))
def apply_all_thresholds(input_folder, output_folder):

    apply_all_thresh(input_folder, output_folder)


if __name__ == "__main__":
    apply_all_thresholds()
