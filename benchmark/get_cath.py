"""Functions for creating CATH datasets"""

import numpy as np
import pandas as pd
import ampal
import gzip
import glob
import subprocess
import multiprocessing

classes = {
    "1": "Mainly Alpha",
    "2": "Mainly Beta",
    "3": "Alpha Beta",
    "4": "Few Secondary Structures",
    "6": "Special",
}

architectures = {
    "1.10": "Orthogonal Bundle",
    "1.20": "Up-down Bundle",
    "1.25": "Alpha Horseshoe",
    "1.40": "Alpha solenoid",
    "1.50": "Alpha/alpha barrel",
    "2.10": "Ribbon",
    "2.20": "Single Sheet",
    "2.30": "Roll",
    "2.40": "Beta Barrel",
    "2.50": "Clam",
    "2.60": "Sandwich",
    "2.70": "Distorted Sandwich",
    "2.80": "Trefoil",
    "2.90": "Orthogonal Prism",
    "2.100": "Aligned Prism",
    "2.102": "3-layer Sandwich",
    "2.105": "3 Propeller",
    "2.110": "4 Propeller",
    "2.115": "5 Propeller",
    "2.120": "6 Propeller",
    "2.130": "7 Propeller",
    "2.140": "8 Propeller",
    "2.150": "2 Solenoid",
    "2.160": "3 Solenoid",
    "2.170": "Beta Complex",
    "2.180": "Shell",
    "3.10": "Roll",
    "3.15": "Super Roll",
    "3.20": "Alpha-Beta Barrel",
    "3.30": "2-Layer Sandwich",
    "3.40": "3-Layer(aba) Sandwich",
    "3.50": "3-Layer(bba) Sandwich",
    "3.55": "3-Layer(bab) Sandwich",
    "3.60": "4-Layer Sandwich",
    "3.65": "Alpha-beta prism",
    "3.70": "Box",
    "3.75": "5-stranded Propeller",
    "3.80": "Alpha-Beta Horseshoe",
    "3.90": "Alpha-Beta Complex",
    "3.100": "Ribosomal Protein L15; Chain: K; domain 2",
    "4.10": "Irregular",
    "6.10": "Helix non-globular",
    "6.20": "Other non-globular",
}


def read_data(CATH_file: str, working_dir: str) -> pd.DataFrame:
    """If CATH .csv exists, loads the DataFrame. If CATH .txt exists, makes DataFrame and saves it.

    Parameters
    ----------
    CATH_file: str
        Name of CATH file.
    working_dir: str
        Path to CATH file.

    Returns
    -------
    DataFrame containing CATH and PDB codes."""

    try:
        df = pd.read_csv(working_dir + CATH_file + ".csv", index_col=0)
        # start, stop needs to be str
        df["start"] = df["start"].apply(str)
        df["stop"] = df["stop"].apply(str)
        return df

    except IOError:
        cath_info = []
        temp = []
        start_stop = []
        # ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/
        with open(working_dir + CATH_file + ".txt") as file:
            for line in file:
                if line[:6] == "DOMAIN":
                    # PDB
                    temp.append(line[10:14])
                    # chain
                    temp.append(line[14])
                if line[:6] == "CATHCO":
                    # class, architecture, topology, homologous superfamily
                    cath = [int(i) for i in line[10:].strip("\n").split(".")]
                    temp = temp + cath
                if line[:6] == "SRANGE":
                    j = line.split()
                    # start and stop resi, can be multiple for the same chain
                    # must be str to deal with insertions (1A,1B) later.
                    start_stop.append([str(j[1][6:]), str(j[2][5:])])
                if line[:2] == "//":
                    # keep fragments from the same chain as separate entries
                    for fragment in start_stop:
                        cath_info.append(temp + fragment)
                    start_stop = []
                    temp = []
        df = pd.DataFrame(
            cath_info,
            columns=[
                "PDB",
                "chain",
                "class",
                "architecture",
                "topology",
                "hsf",
                "start",
                "stop",
            ],
        )
        df.to_csv(working_dir + CATH_file + ".csv")
        return df


def get_sequence(series: pd.Series) -> str:
    """Gets a sequence of CATH structure segment from PDB file.

    Parameters
    ----------
    series: pd.Series
        Series containing one CATH instance.

    Returns
    -------
    If PDB exists, returns sequence, start index and stop index If not, returns np.NaN

    Notes
    -----
    Unnatural amino acids are labelled x"""

    try:
        with gzip.open(
            "/home/shared/datasets/pdb/"
            + series.PDB[1:3]
            + "/pdb"
            + series.PDB
            + ".ent.gz",
            "rb",
        ) as protein:
            assembly = ampal.load_pdb(protein.read().decode(), path=False)
            # convert pdb res id into sequence index,
            # some files have discontinuous residue ids so ampal.get_slice_from_res_id() does not work
            start = 0
            stop = 0
            # if nmr structure, get 1st model
            if isinstance(assembly, ampal.AmpalContainer):
                chain = assembly[0][series.chain]
            else:
                chain = assembly[series.chain]
            for i, residue in enumerate(chain):
                # deal with insertions
                if series.start[-1].isalpha():
                    if (residue.id + residue.insertion_code) == series.start:
                        start = i
                else:
                    if residue.id == series.start:
                        start = i
                if series.stop[-1].isalpha():
                    if (residue.id + residue.insertion_code) == series.stop:
                        stop = i
                else:
                    if residue.id == series.stop:
                        stop = i
        # remove 'X' and convert start/stop residues to list indexes
        # Evo2EF skipps unnatural amino acids but AMPAL puts 'X'. string length and start/stop index must be checked
        filtered_sequence = "".join([x for x in chain.sequence if x != "X"])
        filtered_fragment = "".join(
            [x for x in chain[start : (stop + 1)].sequence if x != "X"]
        )
        new_start = filtered_sequence.find(filtered_fragment)
        new_stop = new_start + len(filtered_fragment) - 1
        return filtered_fragment, new_start, new_stop
    # some pdbs are obsolete, return NaN
    except:
        return np.NaN, np.NaN, np.NaN


def get_pdbs(
    df: pd.DataFrame, cls: int, arch: int = 0, topo: int = 0, homologous_sf: int = 0
) -> pd.DataFrame:
    """Gets PDBs based on CATH code, at least class has to be specified.

    Parameters
    ----------
        df: pd.DataFrame
            DataFrame containing CATH dataset.
        cls: int
            CATH class
        arch: int = 0
            CATH architecture
        topo: int = 0
            CATH topology
        homologous_sf: int = 0
            CATH homologous superfamily

    Returns
    -------
    DataFrame containing PDBs with specified CATH code."""

    if homologous_sf != 0:
        return df.loc[
            (df["class"] == cls)
            & (df["topology"] == topo)
            & (df["architecture"] == arch)
            & (df["hsf"] == homologous_sf)
        ].copy()
    elif topo != 0:
        return df.loc[
            (df["class"] == cls)
            & (df["topology"] == topo)
            & (df["architecture"] == arch)
        ].copy()
    elif arch != 0:
        return df.loc[(df["class"] == cls) & (df["architecture"] == arch)].copy()
    else:
        return df.loc[(df["class"] == cls)].copy()


def append_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Get sequences for all entries in the dataframe, changes start and stop from PDB resid to index number.

    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe

    Returns
    -------
    DataFrame with existing sequences"""
    working_copy = df.copy()

    (
        working_copy.loc[:, "sequence"],
        working_copy.loc[:, "start"],
        working_copy.loc[:, "stop"],
    ) = zip(*[get_sequence(x) for i, x in df.iterrows()])
    # remove missing entries
    working_copy.dropna(inplace=True)
    # change index from float to int
    working_copy.loc[:, "start"] = working_copy["start"].apply(int)
    working_copy.loc[:, "stop"] = working_copy["stop"].apply(int)
    return working_copy


def filter_with_pisces(df: pd.DataFrame, seq_id: int, res: float) -> pd.DataFrame:
    """Takes CATH datarame and makes it non-redundant based on PISCES dataset

    Parameters
    ----------
    df: pd.DataFrame
        CATH DataFrame
    seq_id: int
        Sequence identity cutoff
    res: float
        Resolution cutoff

    Returns
    -------
    A non-redundant DataFrame

    Raises
    ------
    ValueError if seq id or resolution is incorrect"""

    # check for wrong inputs
    allowed_seq_id = [20, 25, 30, 40, 50, 60, 70, 80, 90]
    allowed_res = [1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
    if seq_id not in allowed_seq_id:
        print("Check sequence id")
        raise ValueError
    elif res not in allowed_res:
        print("Check resolution")
        raise ValueError
    # make copy to prevent changes in original df
    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    path_to_pisces = [
        x
        for x in glob.glob(
            "/home/shared/datasets/pisces/cullpdb_pc%i_res%.1f_*" % (seq_id, res)
        )
        if x[-5:] != "fasta"
    ][0]
    with open(path_to_pisces) as file:
        pisces = [x.split()[0] for x in file.readlines()[1:]]
    return df.loc[frame_copy["PDB+chain"].isin(pisces)]


def filter_with_TS50(df: pd.DataFrame) -> pd.DataFrame:
    """Takes CATH datarame and returns PDB chains from TS50 dataset

    Parameters
    ----------
    df: pd.DataFrame
        CATH DataFrame

    Returns
    -------
    TS50 DataFrame

    Reference
    ----------
     https://doi.org/10.1002/prot.25868 (ProDCoNN)"""
    ts50 = [
        "1AHSA",
        "1BVYF",
        "1PDOA",
        "2VA0A",
        "3IEYB",
        "2XR6A",
        "3II2A",
        "1OR4A",
        "2QDLA",
        "3NZMA",
        "3VJZA",
        "1ETEA",
        "2A2LA",
        "2FVVA",
        "3L4RA",
        "1LPBA",
        "3NNGA",
        "2CVIA",
        "3GKNA",
        "2J49A",
        "3FHKA",
        "3PIVA",
        "3LQCA",
        "3GFSA",
        "3E8MA",
        "1DX5I",
        "3NY7A",
        "3K7PA",
        "2CAYA",
        "1I8NA",
        "1V7MV",
        "1H4AX",
        "3T5GB",
        "3Q4OA",
        "3A4RA",
        "2I39A",
        "3AQGA",
        "3EJFA",
        "3NBKA",
        "4GCNA",
        "2XDGA",
        "3GWIA",
        "3HKLA",
        "3SO6A",
        "3ON9A",
        "4DKCA",
        "2GU3A",
        "2XCJA",
        "1Y1LA",
        "1MR1C",
    ]
    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(ts50)]


def most_likely_sequence(sequence: list) -> str:
    acids = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "W",
        18: "Y",
        19: "V",
    }
    seq = np.array(sequence)
    probability_matrix = []
    for x in range(20):
        # rows represent amino acids, columns represent sequence.
        probability_matrix.append(
            [position.count(acids[x]) / len(position) for position in zip(*seq)]
        )
    probability_matrix = np.array(probability_matrix)
    most_likely_sequence = [acids[x] for x in np.argmax(probability_matrix, axis=0)]
    return "".join(most_likely_sequence)


def lookup_blosum62(a: str, b: str) -> int:
    """Returns score from the matrix.

    Parameters
    ----------
    a: str
        First residue code.
    b: str
        Second residue code.

    Returns
    --------
    Score"""

    blosum62 = {
        ("W", "F"): 1,
        ("L", "R"): -2,
        ("S", "P"): -1,
        ("V", "T"): 0,
        ("Q", "Q"): 5,
        ("N", "A"): -2,
        ("Z", "Y"): -2,
        ("W", "R"): -3,
        ("Q", "A"): -1,
        ("S", "D"): 0,
        ("H", "H"): 8,
        ("S", "H"): -1,
        ("H", "D"): -1,
        ("L", "N"): -3,
        ("W", "A"): -3,
        ("Y", "M"): -1,
        ("G", "R"): -2,
        ("Y", "I"): -1,
        ("Y", "E"): -2,
        ("B", "Y"): -3,
        ("Y", "A"): -2,
        ("V", "D"): -3,
        ("B", "S"): 0,
        ("Y", "Y"): 7,
        ("G", "N"): 0,
        ("E", "C"): -4,
        ("Y", "Q"): -1,
        ("Z", "Z"): 4,
        ("V", "A"): 0,
        ("C", "C"): 9,
        ("M", "R"): -1,
        ("V", "E"): -2,
        ("T", "N"): 0,
        ("P", "P"): 7,
        ("V", "I"): 3,
        ("V", "S"): -2,
        ("Z", "P"): -1,
        ("V", "M"): 1,
        ("T", "F"): -2,
        ("V", "Q"): -2,
        ("K", "K"): 5,
        ("P", "D"): -1,
        ("I", "H"): -3,
        ("I", "D"): -3,
        ("T", "R"): -1,
        ("P", "L"): -3,
        ("K", "G"): -2,
        ("M", "N"): -2,
        ("P", "H"): -2,
        ("F", "Q"): -3,
        ("Z", "G"): -2,
        ("X", "L"): -1,
        ("T", "M"): -1,
        ("Z", "C"): -3,
        ("X", "H"): -1,
        ("D", "R"): -2,
        ("B", "W"): -4,
        ("X", "D"): -1,
        ("Z", "K"): 1,
        ("F", "A"): -2,
        ("Z", "W"): -3,
        ("F", "E"): -3,
        ("D", "N"): 1,
        ("B", "K"): 0,
        ("X", "X"): -1,
        ("F", "I"): 0,
        ("B", "G"): -1,
        ("X", "T"): 0,
        ("F", "M"): 0,
        ("B", "C"): -3,
        ("Z", "I"): -3,
        ("Z", "V"): -2,
        ("S", "S"): 4,
        ("L", "Q"): -2,
        ("W", "E"): -3,
        ("Q", "R"): 1,
        ("N", "N"): 6,
        ("W", "M"): -1,
        ("Q", "C"): -3,
        ("W", "I"): -3,
        ("S", "C"): -1,
        ("L", "A"): -1,
        ("S", "G"): 0,
        ("L", "E"): -3,
        ("W", "Q"): -2,
        ("H", "G"): -2,
        ("S", "K"): 0,
        ("Q", "N"): 0,
        ("N", "R"): 0,
        ("H", "C"): -3,
        ("Y", "N"): -2,
        ("G", "Q"): -2,
        ("Y", "F"): 3,
        ("C", "A"): 0,
        ("V", "L"): 1,
        ("G", "E"): -2,
        ("G", "A"): 0,
        ("K", "R"): 2,
        ("E", "D"): 2,
        ("Y", "R"): -2,
        ("M", "Q"): 0,
        ("T", "I"): -1,
        ("C", "D"): -3,
        ("V", "F"): -1,
        ("T", "A"): 0,
        ("T", "P"): -1,
        ("B", "P"): -2,
        ("T", "E"): -1,
        ("V", "N"): -3,
        ("P", "G"): -2,
        ("M", "A"): -1,
        ("K", "H"): -1,
        ("V", "R"): -3,
        ("P", "C"): -3,
        ("M", "E"): -2,
        ("K", "L"): -2,
        ("V", "V"): 4,
        ("M", "I"): 1,
        ("T", "Q"): -1,
        ("I", "G"): -4,
        ("P", "K"): -1,
        ("M", "M"): 5,
        ("K", "D"): -1,
        ("I", "C"): -1,
        ("Z", "D"): 1,
        ("F", "R"): -3,
        ("X", "K"): -1,
        ("Q", "D"): 0,
        ("X", "G"): -1,
        ("Z", "L"): -3,
        ("X", "C"): -2,
        ("Z", "H"): 0,
        ("B", "L"): -4,
        ("B", "H"): 0,
        ("F", "F"): 6,
        ("X", "W"): -2,
        ("B", "D"): 4,
        ("D", "A"): -2,
        ("S", "L"): -2,
        ("X", "S"): 0,
        ("F", "N"): -3,
        ("S", "R"): -1,
        ("W", "D"): -4,
        ("V", "Y"): -1,
        ("W", "L"): -2,
        ("H", "R"): 0,
        ("W", "H"): -2,
        ("H", "N"): 1,
        ("W", "T"): -2,
        ("T", "T"): 5,
        ("S", "F"): -2,
        ("W", "P"): -4,
        ("L", "D"): -4,
        ("B", "I"): -3,
        ("L", "H"): -3,
        ("S", "N"): 1,
        ("B", "T"): -1,
        ("L", "L"): 4,
        ("Y", "K"): -2,
        ("E", "Q"): 2,
        ("Y", "G"): -3,
        ("Z", "S"): 0,
        ("Y", "C"): -2,
        ("G", "D"): -1,
        ("B", "V"): -3,
        ("E", "A"): -1,
        ("Y", "W"): 2,
        ("E", "E"): 5,
        ("Y", "S"): -2,
        ("C", "N"): -3,
        ("V", "C"): -1,
        ("T", "H"): -2,
        ("P", "R"): -2,
        ("V", "G"): -3,
        ("T", "L"): -1,
        ("V", "K"): -2,
        ("K", "Q"): 1,
        ("R", "A"): -1,
        ("I", "R"): -3,
        ("T", "D"): -1,
        ("P", "F"): -4,
        ("I", "N"): -3,
        ("K", "I"): -3,
        ("M", "D"): -3,
        ("V", "W"): -3,
        ("W", "W"): 11,
        ("M", "H"): -2,
        ("P", "N"): -2,
        ("K", "A"): -1,
        ("M", "L"): 2,
        ("K", "E"): 1,
        ("Z", "E"): 4,
        ("X", "N"): -1,
        ("Z", "A"): -1,
        ("Z", "M"): -1,
        ("X", "F"): -1,
        ("K", "C"): -3,
        ("B", "Q"): 0,
        ("X", "B"): -1,
        ("B", "M"): -3,
        ("F", "C"): -2,
        ("Z", "Q"): 3,
        ("X", "Z"): -1,
        ("F", "G"): -3,
        ("B", "E"): 1,
        ("X", "V"): -1,
        ("F", "K"): -3,
        ("B", "A"): -2,
        ("X", "R"): -1,
        ("D", "D"): 6,
        ("W", "G"): -2,
        ("Z", "F"): -3,
        ("S", "Q"): 0,
        ("W", "C"): -2,
        ("W", "K"): -3,
        ("H", "Q"): 0,
        ("L", "C"): -1,
        ("W", "N"): -4,
        ("S", "A"): 1,
        ("L", "G"): -4,
        ("W", "S"): -3,
        ("S", "E"): 0,
        ("H", "E"): 0,
        ("S", "I"): -2,
        ("H", "A"): -2,
        ("S", "M"): -1,
        ("Y", "L"): -1,
        ("Y", "H"): 2,
        ("Y", "D"): -3,
        ("E", "R"): 0,
        ("X", "P"): -2,
        ("G", "G"): 6,
        ("G", "C"): -3,
        ("E", "N"): 0,
        ("Y", "T"): -2,
        ("Y", "P"): -3,
        ("T", "K"): -1,
        ("A", "A"): 4,
        ("P", "Q"): -1,
        ("T", "C"): -1,
        ("V", "H"): -3,
        ("T", "G"): -2,
        ("I", "Q"): -3,
        ("Z", "T"): -1,
        ("C", "R"): -3,
        ("V", "P"): -2,
        ("P", "E"): -1,
        ("M", "C"): -1,
        ("K", "N"): 0,
        ("I", "I"): 4,
        ("P", "A"): -1,
        ("M", "G"): -3,
        ("T", "S"): 1,
        ("I", "E"): -3,
        ("P", "M"): -2,
        ("M", "K"): -1,
        ("I", "A"): -1,
        ("P", "I"): -3,
        ("R", "R"): 5,
        ("X", "M"): -1,
        ("L", "I"): 2,
        ("X", "I"): -1,
        ("Z", "B"): 1,
        ("X", "E"): -1,
        ("Z", "N"): 0,
        ("X", "A"): 0,
        ("B", "R"): -1,
        ("B", "N"): 3,
        ("F", "D"): -3,
        ("X", "Y"): -1,
        ("Z", "R"): 0,
        ("F", "H"): -1,
        ("B", "F"): -3,
        ("F", "L"): 0,
        ("X", "Q"): -1,
        ("B", "B"): 4,
    }
    if (a, b) in blosum62.keys():
        return blosum62[a, b]
    else:
        return blosum62[b, a]


def sequence_recovery(true_seq: str, predicted_seq: str) -> float:
    """Calculates sequence recovery.

    Parameters
    ----------
    true_seq: str,
        True sequence.
    predicted_seq: str
        Predicted sequence.

    Returns
    --------
    Sequence recovery rate"""

    correct = 0
    for i, acid in enumerate(true_seq):
        if acid == predicted_seq[i]:
            correct = correct + 1
    return correct / len(predicted_seq)


def fuzzy_score(true_seq: str, predicted_seq: str) -> float:

    """Calculates fuzzy sequence recovery, amino acids of the same type (positive blosum62 score) are considered as correct predictions.

    Parameters
    ----------
    true_seq: str,
        True sequence.
    predicted_seq: str
        Predicted sequence.

    Returns
    --------
    Fuzzy sequence recovery rate

    Reference
    ---------
    https://doi.org/10.1002/prot.25868 (ProDCoNN)"""

    correct = 0
    for i, acid in enumerate(true_seq):
        if lookup_blosum62(acid, predicted_seq[i]) > 0:
            correct = correct + 1
    return correct / len(true_seq)


def run_Evo2EF(path: str, pdb: str, chain: str, number_of_runs: str):
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

    Returns
    -------
    Nothing."""

    p = subprocess.Popen(
        [
            "/home/s1706179/project/sequence-recovery-benchmark/evo.sh",
            path,
            pdb,
            chain,
            number_of_runs,
        ]
    )
    p.wait()
    print("%s%s done" % (pdb, chain))


def multi_Evo2EF(df: pd.DataFrame, number_of_runs: int, max_processes: int = 8):
    """Runs Evo2EF on all PDB chains in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with PDB and chain codes.
    number_of_runs: int
        Number of sequences to be generated for each PDB file.
    max_processes: int = 8
        Number of cores to use, default is 8.
    Returns
    --------
    Nothing."""

    inputs = []
    # remove duplicated chains
    df = df.drop_duplicates(subset=["PDB", "chain"])
    for i, protein in df.iterrows():
        path = (
            "/home/shared/datasets/biounit/"
            + protein.PDB[1:3]
            + "/"
            + protein.PDB
            + ".pdb1.gz"
        )
        inputs.append((path, protein.PDB, protein.chain, str(number_of_runs)))
    with multiprocessing.Pool(max_processes) as P:
        P.starmap(run_Evo2EF, inputs)


def load_predictions(df: pd.DataFrame) -> pd.DataFrame:
    predicted_sequences = []
    for i, protein in df.iterrows():
        try:
            with open(
                "/home/s1706179/project/sequence-recovery-benchmark/evo_dataset/%s%s.txt"
                % (protein.PDB, protein.chain)
            ) as prediction:
                predicted_sequences.append(
                    [
                        y.split()[0]
                        for y in prediction.readlines()
                        if y.split()[0] != "0"
                    ]
                )
        except FileNotFoundError:
            print("%s%s prediction does not exits." % (protein.PDB, protein.chain))
            predicted_sequences.append(np.NaN)
    df["predicted_sequences"] = predicted_sequences
    return df


def score(df: pd.DataFrame, score_type: str = "sequence_recovery") -> list:
    """Scores all predicted sequences in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    score_type: str ='sequence_recovery'
        Can choose between 'sequence_recovery' and 'fuzzy score'.

    Returns
    --------
    A list with scores."""

    scores = []
    for i, protein in df.iterrows():
        # supports multiple predictions
        start = protein.start
        stop = protein.stop
        # check if sequence exists
        if protein.predicted_sequences == []:
            print("Check %s %s, something went wrong" % (protein.PDB, protein.chain))
            scores.append(np.NaN)
        # check if length matches
        elif len(protein.predicted_sequences[0][start : stop + 1]) == len(
            protein.sequence
        ):
            if score_type == "sequence_recovery":
                scores.append(
                    sum(
                        [
                            sequence_recovery(
                                protein.sequence, prediction[start : stop + 1]
                            )
                            for prediction in protein.predicted_sequences
                        ]
                    )
                    / len(protein.predicted_sequences)
                )
            elif score_type == "fuzzy_score":
                scores.append(
                    sum(
                        [
                            fuzzy_score(protein.sequence, prediction[start : stop + 1])
                            for prediction in protein.predicted_sequences
                        ]
                    )
                    / len(protein.predicted_sequences)
                )
        else:
            print("Check %s %s, something went wrong" % (protein.PDB, protein.chain))
            scores.append(np.NaN)
    return scores
