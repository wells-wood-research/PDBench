import ampal
import gzip
import glob
import subprocess
import multiprocessing
import os
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import warnings
import numpy as np
import pandas as pd
import string
import urllib
from sklearn import metrics
import numpy as np

acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

standard_residues = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def atom_to_hetatm(pdb: Path):
    """Rosetta labels non-standard acids as ATOM instead of HETATM. This crashes AMPAL."""
    with open(pdb, "r") as file:
        text = file.readlines()
    for i, line in enumerate(text):
        if line[0:6].strip() == "ATOM" and line[17:20].strip() not in standard_residues:
            text[i] = "HETATM" + text[i][6:]
    with open(pdb, "w") as file:
        file.writelines(text)


def run_Rosetta(
    pdb: str,
    chain: str,
    working_dir: Path,
    path_to_Rosetta: Path,
    path_to_assemblies: Path,
) -> None:
    """Runs Rosetta design with fixed backbone
    Patameters
    ----------
    pdb: str
        PDB code.
    chain: str
        Chain code.
    working_dir: str
      Dir where to store temporary files and results.
    path_to_Rosetta: Path
        Location of Rosetta executable.
    path_to_assemblies:Path
        Location of input PDB structures.
    """

    print(f"Starting {pdb}{chain}.")
    # make resfile to predict only the specified chain, skip non-canonical residues
    assembly = ampal.load_pdb(Path(path_to_assemblies / pdb).with_suffix(".pdb"))
    with open(working_dir / ("resfile_" + pdb), "w") as file:
        file.write("NATRO\nstart\n")
        for i, x in enumerate(assembly[chain]):
            file.write(f"{x.id} {chain} ALLAA\n")
    p = subprocess.run(
        f'{path_to_Rosetta} -s {Path(path_to_assemblies/pdb).with_suffix(".pdb")} -linmem_ig 10 -ignore_unrecognized_res -overwrite -resfile {working_dir/("resfile_"+pdb)} -out:path:all {working_dir/"results"}',
        shell=True,
    )
    print(f"{pdb}{chain} done.")


def seq_to_arr(working_dir: Path, user_list: Path, ignore_uncommon: bool = False):
    """Produces prediction format compatible with the benchmarking tool.
    working_dir: Path
        Dir where Rosetta results are stored.
    user_list: Path
        Path to .txt file with protein chains to include in the benchmark"""

    with open(Path(user_list)) as file:
        chains = [x.strip("\n") for x in file.readlines()]
    predicted_sequences = []
    path = working_dir / "results"
    enc = OneHotEncoder(categories=[acids], sparse=False)
    with open(path / "datasetmap.txt", "w") as file:
        file.write(f"ignore_uncommon {ignore_uncommon}\ninclude_pdbs\n##########\n")
        for protein in chains:
            prediction_path = path / f"{protein[:4]}_0001.pdb"
            # check for empty and missing files
            if prediction_path.exists():
                try:
                    assembly = ampal.load_pdb(prediction_path)
                # fix malformed files
                except ValueError:
                    atom_to_hetatm(prediction_path)
                    assembly = ampal.load_pdb(prediction_path)
                # exclude positions with non-cannonical amino acids
                if ignore_uncommon == True:
                    # path to pdb has changed, change it manualy if you decide to use this option.
                    temp_assembly = ampal.load_pdb(working_dir / f"{protein[:4]}.pdb")
                    true_seq = temp_assembly[protein[-1]].sequence
                    print(metrics.accuracy_score(list(seq), list(true_seq)))
                    assert len(seq) == len(
                        true_seq
                    ), f"{protein} sequence lengths don't match"
                    seq = "".join(
                        [
                            pred_ch
                            for pred_ch, true_ch in zip(list(seq), list(true_seq))
                            if true_ch != "X"
                        ]
                    )
                    if seq.find("X") != -1:
                        warnings.warn(
                            f"Rosetta: {protein} has remaining non-canonical acids."
                        )
                seq = assembly[protein[-1]].sequence
                predicted_sequences += list(seq)
                file.write(f"{protein} {len(seq)}\n")
            else:
                warnings.warn(f"Rosetta: {protein} prediction does not exits.")
    arr = enc.fit_transform(np.array(predicted_sequences).reshape(-1, 1))
    pd.DataFrame(arr).to_csv(path / "rosetta.csv", header=None, index=None)


def multi_Rosetta(
    structures: list,
    working_dir: Path,
    path_to_assemblies: Path,
    path_to_rosetta: Path,
    max_processes: int = 8,
) -> None:
    """Runs Rosetta on all PDB chains in the DataFrame.
    Parameters
    ----------
    structures:List
        List with PDB and chain codes.
    number_of_runs: int
        Number of sequences to be generated for each PDB file.
    max_processes: int = 8
        Number of cores to use, default is 8.
    working_dir: Path
      Dir where to store temporary files and results.
    path_to_assemblies: Path
        Dir with biological assemblies.
    path_to_rosetta: Path
        Location of rosetta executable.
    """

    inputs = []

    # check if working directory exists. Make one if doesn't exist.
    if not working_dir.exists():
        os.makedirs(working_dir)
    if not (working_dir / "results").exists():
        os.makedirs(working_dir / "results")
    print(f"{len(structures)} structures will be predicted.")

    for protein in structures:
        inputs.append(
            (
                protein[:4],
                protein[4],
                working_dir,
                path_to_rosetta,
                path_to_assemblies,
            )
        )
    with multiprocessing.Pool(max_processes) as P:
        P.starmap(run_Rosetta, inputs)


if __name__ == "__main__":
    # seq_to_arr(Path('/home/s1706179/Rosetta/data_polyglycine/'),Path('/home/s1706179/Rosetta/data/set.txt'),False)
    seq_to_arr(
        Path("/home/s1706179/Rosetta/data_nmr_polyglycine/"),
        Path("/home/s1706179/Rosetta/data/nmr_set.txt"),
        False,
    )
