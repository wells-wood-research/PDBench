"""Runs benchmark via command line."""

import click

from benchmark.run_comparison import compare_models
from benchmark.run_evoef2 import run_evoEF2

@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(compare_models)
    cli.add_command(run_evoEF2)
    cli()
