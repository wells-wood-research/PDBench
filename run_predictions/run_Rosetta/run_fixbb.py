"Runs Rosetta predictions"

import fixbb
from pathlib import Path
import click
import os


@click.command()
@click.option(
    "--dataset",
    help="Path to .txt file with dataset list (PDB+chain, e.g., 1a2bA).",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--path_to_assemblies",
    help="Path to the directory with biological assemblies.",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--working_dir",
    help="Directory where to store results.",
    type=click.Path(),
    required=True,
)
@click.option(
    "--path_to_rosetta",
    help="Path to Rosetta executable.",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--max_processes", help="Maximum number of cores to use", type=int, default=8
)
def run_rosetta(
    dataset: str,
    working_dir: str,
    path_to_rosetta: str,
    max_processes: int,
    path_to_assemblies: str,
) -> None:
    """Runs EvoEF2 sequence predictions on a specified set.
    \f
    Parameters
    ---------
    dataset: str
        Path to .txt file with dataset list (PDB+chain, e.g., 1a2bA).
    working_dir: str
        Path to dir where to save temp files and results.
    path_to_rosetta: str
        Path to Rosetta executable.
    max_processes: int
        Maximum number of cores to use.
    path_to_assemblies: str
        Path to the directory with biological assemblies.
    nmr: bool
        If true, the code expects a PDB file with NMR states insted of biological assemblies.
    """
    with open(dataset, "r") as file:
        structures = [x.strip("\n") for x in file.readlines()]
    fixbb.multi_Rosetta(
        structures,
        max_processes=max_processes,
        working_dir=Path(working_dir),
        path_to_rosetta=Path(path_to_rosetta),
        path_to_assemblies=Path(path_to_assemblies),
    )


if __name__ == "__main__":
    run_rosetta()
