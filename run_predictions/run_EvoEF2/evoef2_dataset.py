"""Functions for making EvoEF2 predictions."""

import ampal
import gzip
import glob
import subprocess
import multiprocessing
import os
from pathlib import Path
from benchmark import config
from sklearn.preprocessing import OneHotEncoder
import warnings
import numpy as np
import pandas as pd

def run_Evo2EF(
    pdb: str, chain: str, number_of_runs: str, working_dir: Path, path_to_evoef2: Path
) -> None:
    """Runs a shell script to predict sequence with EvoEF2

    Patameters
    ----------
    path: str
        Path to PDB biological unit.
    pdb: str
        PDB code.
    chain: str
        Chain code.
    number_of_runs: str
       Number of sequences to be generated.
    working_dir: str
      Dir where to store temporary files and results.
    path_to_EvoEF2: Path
        Location of EvoEF2 executable.
    """

    print(f"Starting {pdb}{chain}.")

    # evo.sh must be in the same directory as this file.
    p = subprocess.Popen(
        [
            os.path.dirname(os.path.realpath(__file__)) + "/evo.sh",
            pdb,
            chain,
            number_of_runs,
            working_dir,
            path_to_evoef2,
        ]
    )
    p.wait()
    print(f"{pdb}{chain} done.")


def multi_Evo2EF(
    df: pd.DataFrame,
    number_of_runs: int,
    working_dir: Path,
    path_to_assemblies: Path,
    path_to_evoef2: Path,
    max_processes: int = 8,
    nmr:bool = False,
) -> None:
    """Runs Evo2EF on all PDB chains in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with PDB and chain codes.
    number_of_runs: int
        Number of sequences to be generated for each PDB file.
    max_processes: int = 8
        Number of cores to use, default is 8.
    working_dir: Path
      Dir where to store temporary files and results.
    path_to_assemblies: Path
        Dir with biological assemblies.
    path_to_EvoEF2: Path
        Location of EvoEF2 executable.
    nmr:bool=True

    """

    inputs = []
    # remove duplicated chains
    df = df.drop_duplicates(subset=["PDB", "chain"])

    # check if working directory exists. Make one if doesn't exist.
    if not working_dir.exists():
        os.makedirs(working_dir)
    if not (working_dir / "results/").exists():
        os.makedirs(working_dir / "results/")

    print(f"{df.shape[0]} structures will be predicted.")

    for i, protein in df.iterrows():
        if not nmr:
            with gzip.open(
                path_to_assemblies / protein.PDB[1:3] / f"{protein.PDB}.pdb1.gz"
            ) as file:
                assembly = ampal.load_pdb(file.read().decode(), path=False)
            # fuse all states of the assembly into one state to avoid EvoEF2 errors.
            empty_polymer = ampal.Assembly()
            chain_id = []
            for polymer in assembly:
                for chain in polymer:
                    empty_polymer.append(chain)
                    chain_id.append(chain.id)
            # relabel chains to avoid repetition, remove ligands.
            str_list = string.ascii_uppercase.replace(protein.chain, "")
            index = chain_id.index(protein.chain)
            chain_id = list(str_list[: len(chain_id)])
            chain_id[index] = protein.chain
            empty_polymer.relabel_polymers(chain_id)
            pdb_text = empty_polymer.make_pdb(alt_states=False, ligands=False)
            # writing new pdb with AMPAL fixes most of the errors with EvoEF2.
            with open((working_dir / protein.PDB).with_suffix(".pdb1"), "w") as pdb_file:
                pdb_file.write(pdb_text)
                
        #pick first nmr structure        
        else:
            with gzip.open(
                path_to_assemblies / protein.PDB[1:3] / f"pdb{protein.PDB}.ent.gz"
            ) as file:
                assembly = ampal.load_pdb(file.read().decode(), path=False)
            pdb_text = assembly[0].make_pdb(alt_states=False)
            # writing new pdb with AMPAL fixes most of the errors with EvoEF2.
            with open((working_dir / protein.PDB).with_suffix(".pdb1"), "w") as pdb_file:
                pdb_file.write(pdb_text)
                
        inputs.append(
                (
                    protein.PDB,
                    protein.chain,
                    str(number_of_runs),
                    working_dir,
                    path_to_evoef2,
                )
            )

    with multiprocessing.Pool(max_processes) as P:
        P.starmap(run_Evo2EF, inputs)
        
def seq_to_arr(working_dir: Path, user_list: Path, ignore_uncommon:bool=True):
    """Produces prediction format compatible with the benchmarking tool.
      working_dir: Path
          Dir where EvoEF2 results are stored.
      user_list: Path
          Path to .txt file with protein chains to include in the benchmark"""
          
    with open(Path(user_list)) as file:
        chains=[x.strip('\n') for x in file.readlines()]
    predicted_sequences = []
    path = Path(working_dir)
    enc=OneHotEncoder(categories=[config.acids],sparse=False)
    with open(path/'datasetmap.txt','w') as file:
        file.write(f"ignore_uncommon {ignore_uncommon}\ninclude_pdbs\n##########\n")
        for protein in chains:
            prediction_path = path / "results"/f"{protein}.txt"
            # check for empty and missing files
            if prediction_path.exists() and os.path.getsize(prediction_path) > 0:
                with open(prediction_path) as prediction:
                    seq = prediction.readline().split()[0]
                    if seq != "0":
                        predicted_sequences+=list(seq)
                       
                        file.write(f"{protein} {len(seq)}\n")
                    else:
                        warnings.warn(
                            f"EvoEF2: {protein} prediction does not exits, EvoEF2 returned 0."
                        )
            else:
                warnings.warn(
                    f"EvoEF2: {protein} prediction does not exits."
                )
    arr=enc.fit_transform(np.array(predicted_sequences).reshape(-1, 1))
    pd.DataFrame(arr).to_csv(path/"evoEF2.csv", header=None, index=None)

