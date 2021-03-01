"Runs EvoEF2 predictions"

from benchmark import visualization
from benchmark import get_cath
from pathlib import Path
import click
import os


@click.command("run_EvoEF2")
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
    "--path_to_evoef2",
    help="Path to EvoEF2 executable.",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--max_processes", help="Maximum number of cores to use", type=int, default=8
)
def run_evoEF2(
    dataset: str,
    working_dir: str,
    path_to_evoef2: str,
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
    path_to_evoef2: str
        Path to EvoEF2 executable.
    max_processes: int
        Maximum number of cores to use.
    path_to_assemblies: str
        Path to the directory with biological assemblies.
    """

    df = get_cath.read_data("cath-domain-description-file.txt")
    filtered_df = get_cath.filter_with_user_list(df, dataset)

    get_cath.multi_Evo2EF(
        filtered_df,
        1,
        max_processes=max_processes,
        working_dir=Path(working_dir),
        path_to_evoef2=Path(path_to_evoef2),
        path_to_assemblies=Path(path_to_assemblies),
    )
