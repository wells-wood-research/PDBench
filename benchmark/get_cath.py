"""Functions for creating and scoring CATH datasets"""

import numpy as np
import pandas as pd
import ampal
import gzip
import glob
import subprocess
import multiprocessing
import os
from pathlib import Path
from sklearn import metrics
from benchmark import config
import string
from subprocess import CalledProcessError
import re
from scipy.stats import entropy
from benchmark import visualization
from typing import Tuple, List, Iterable
import warnings
from sklearn.preprocessing import LabelBinarizer
import wget
import click

def download_data(out_dir: Path) -> None:
    """Download CATH file.
    
    Parameters
    ----------
    out_dir: Path:
        Directory where to store the file."""
    if click.confirm(
            f"CATH file does not exist. It will be downloaded to {out_dir.resolve()}. Continue? "
        ):
         wget.download('ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-description-file.txt', out=str(out_dir))
    else:
        exit()
   
def read_data(CATH_file: str) -> pd.DataFrame:
    """If CATH .csv exists, loads the DataFrame. If CATH .txt exists, makes DataFrame and saves it. If file doesn't exist, downloads it.

    Parameters
    ----------
    CATH_file: str
        CATH .txt file name.

    Returns
    -------
    df:pd.DataFrame
        DataFrame containing CATH and PDB codes."""
    path = Path(CATH_file)
    #download if doesn't exist.
    if not path.exists():
        download_data(path.parent)
    # load .csv if exists, faster than reading .txt
    if path.with_suffix(".csv").exists():
        df = pd.read_csv(path.with_suffix(".csv"), index_col=0)
        # start, stop needs to be str
        df["start"] = df["start"].apply(str)
        df["stop"] = df["stop"].apply(str)
        return df

    else:
        cath_info = []
        temp = []
        start_stop = []
        with open(path) as file:
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
        df.to_csv(path.with_suffix(".csv"))
        return df


def tag_dssp_data(assembly: ampal.Assembly) -> None:
    """Same as ampal.dssp.tag_dssp_data(), but fixed a bug with insertions. Tags each residue in ampal.Assembly with secondary structure. Works in place.

    Parameters
    ----------
    assembly: ampal.Assembly
        Protein assembly."""

    dssp_out = ampal.dssp.run_dssp(assembly.pdb, path=False)
    dssp_data = ampal.dssp.extract_all_ss_dssp(dssp_out, path=False)
    for i, record in enumerate(dssp_data):
        rnum, sstype, chid, _, phi, psi, sacc = record
        # deal with insertions
        if len(chid) > 1:
            for i, res in enumerate(assembly[chid[1]]):
                if res.insertion_code == chid[0] and assembly[chid[1]][i].tags == {}:
                    assembly[chid[1]][i].tags["dssp_data"] = {
                        "ss_definition": sstype,
                        "solvent_accessibility": sacc,
                        "phi": phi,
                        "psi": psi,
                    }
                    break

        else:
            assembly[chid][str(rnum)].tags["dssp_data"] = {
                "ss_definition": sstype,
                "solvent_accessibility": sacc,
                "phi": phi,
                "psi": psi,
            }


def get_sequence(
    series: pd.Series, path_to_pdb: Path
) -> Tuple[str, str, int, int, List[int]]:
    """Gets a sequence of from PDB file, CATH fragment indexes and secondary structure labels.

    Parameters
    ----------
    series: pd.Series
        Series containing one CATH instance.
    path_to_assemblies:Path
        Path to directory with biologcial assemblies.

    Returns
    -------
    sequence: str
        True sequence.
    dssp: str
        dssp codes.
    start: int
        CATH fragment start residue number, same as in PDB. NOT EQUAL TO SEQUENCE INDEX.
    stop:int
        CATH fragment stop residue number, same as in PDB. NOT EQUAL TO SEQUENCE INDEX.
    uncommon_index:list
        List with residue number of uncommon amino acids.
    """

    path = path_to_pdb / series.PDB[1:3] / f"pdb{series.PDB}.ent.gz"

    if path.exists():
        with gzip.open(path, "rb") as protein:
            assembly = ampal.load_pdb(protein.read().decode(), path=False)
            # convert pdb res id into sequence index,
            # some files have discontinuous residue ids so ampal.get_slice_from_res_id() does not work
            start = 0
            stop = 0
            # if nmr structure, get 1st model
            if isinstance(assembly, ampal.AmpalContainer):
                assembly = assembly[0]
            # run dssp
            try:
                tag_dssp_data(assembly)
            except CalledProcessError:
                raise CalledProcessError(f"dssp failed on {series.PDB}.pdb.")
            # some biological assemblies are broken
            try:
                chain = assembly[series.chain]
            except KeyError:
                raise KeyError(f"{series.PDB}.pdb is missing chain {series.chain}.")

            # compatibility with evoef and leo's model, store non-canonical residue index in a separate column and include regular amino acid in the sequence
            sequence = ""
            uncommon_index = []
            dssp = ""
            for i, residue in enumerate(chain):
                # add dssp data, assume random structure if dssp did not return anything for this residue
                try:
                    dssp += residue.tags["dssp_data"]["ss_definition"]
                except KeyError:
                    dssp += " "
                # deal with uncommon residues
                one_letter_code = ampal.amino_acids.get_aa_letter(residue.mol_code)
                if one_letter_code == "X":
                    try:
                        uncommon_index.append(i)
                        sequence += ampal.amino_acids.get_aa_letter(
                            config.UNCOMMON_RESIDUE_DICT[residue.mol_code]
                        )
                    except KeyError:
                        raise ValueError(
                            f"{series.PDB}.pdb has unrecognized amino acid {residue.mol_code}."
                        )
                else:
                    sequence += one_letter_code

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
        if uncommon_index==[]:
            uncommon_index=np.NaN
        return sequence, dssp, start, stop, uncommon_index
    else:
        raise FileNotFoundError(
            f"{series.PDB}.pdb is missing, download it or remove it from your dataset."
        )


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
    df:pd.DataFrame
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


def get_resolution(df: pd.DataFrame, path_to_pdb: Path) -> List[float]:
    """Gets resolution of each structure in DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info.
    path_to_pdb: Path
        Path to the directory with PDB files.

    Returns
    -------
    res: list
        List with resolutions."""

    res = []
    for i, protein in df.iterrows():
        path = path_to_pdb / protein.PDB[1:3] / f"pdb{protein.PDB}.ent.gz"

        if path.exists():
            with gzip.open(path, "rb") as pdb:
                pdb_text = pdb.read().decode()
            item = re.findall("REMARK   2 RESOLUTION.*$", pdb_text, re.MULTILINE)
            
            if item[0].split()[3]!='NOT':
                res.append(float(item[0].split()[3]))
            #nmr structures have no resolution
            else:
                res.append(np.NaN)
        else:
            res.append(np.NaN)
    return res


def append_sequence(
    df: pd.DataFrame, path_to_pdb: Path
) -> pd.DataFrame:
    """Get sequences for all entries in the dataframe, changes start and stop from PDB resid to index number,adds resolution of each chain.

    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe.
    path_to_pdb: Path
        Path to the directory with PDB files.


    Returns
    -------
    working_copy:pd.DataFrame
        DataFrame with appended sequences,dssp data, start/stop numbers, uncommon index list and resolution data."""

    # make copy to avoid changing original df.
    working_copy = df.copy()
    sequence, dssp, start, stop, uncommon_index = zip(
        *[get_sequence(x, path_to_pdb) for i, x in df.iterrows()]
    )
    working_copy.loc[:, "sequence"] = sequence
    working_copy.loc[:, "dssp"] = dssp
    working_copy.loc[:, "start"] = start
    working_copy.loc[:, "stop"] = stop
    working_copy.loc[:, "uncommon_index"]=uncommon_index
    working_copy.loc[:, "resolution"] = get_resolution(working_copy, path_to_pdb)

    return working_copy


def filter_with_user_list(
    df: pd.DataFrame, path: Path, ispisces: bool = False
) -> pd.DataFrame:
    """Selects PDB chains specified in .txt file. Multiple CATH entries for the same protein are removed to leave only one example.
    Parameters
    ----------
    df: pd.DataFrame
        CATH info containing dataframe
    path: Path
        Path to dataset .txt file
    ispisces:bool = False
        Reads pisces formating if True, otherwise pdb+chain, e.g., 1a2bA\n.

    Returns
    -------
    DataFrame with selected chains."""

    path = Path(path)
    with open(path) as file:
        if ispisces:
            filtr = [x.split()[0] for x in file.readlines()[1:]]
        else:
            filtr = [x.upper().strip("\n") for x in file.readlines()]
    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(filtr)].drop_duplicates(
        subset=["PDB", "chain"]
    )


def filter_with_resolution(
    df: pd.DataFrame, minimum: float, maximum: float
) -> pd.DataFrame:
    """Gets DataFrame slice with chain resolution between min and max.

    Parameters:
    -----------
    df: pd.DataFrame
        CATH DataFrame.
    minimum:float
    maximum:float

    Returns
    -------
    DataFrame with chains."""

    return df[(df["resolution"] >= minimum) & (df["resolution"] < maximum)]


def lookup_blosum62(res_true: str, res_prediction: str) -> int:
    """Returns score from the matrix.

    Parameters
    ----------
    res_true: str
        First residue code.
    res_prediction: str
        Second residue code.

    Returns
    --------
    Score from the matrix."""

    if (res_true, res_prediction) in config.blosum62.keys():
        return config.blosum62[res_true, res_prediction]
    else:
        return config.blosum62[res_prediction, res_true]

def load_prediction_matrix(
    df: pd.DataFrame, path_to_dataset: Path, path_to_probabilities: Path
) -> dict:
    """Loads predicted probabilities from .csv file to dictionary, drops entries for which sequence prediction fails.
    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe.
    path_to_dataset: Path
        Path to prediction dataset labels.
    path_to_probabilities:Path
        Path to .csv file with probabilities.

    Returns
    -------
    empty_dict:dict
        Dictionary with predicted sequences, key is PDB+chain."""

    path_to_dataset = Path(path_to_dataset)
    path_to_probabilities = Path(path_to_probabilities)
    counter=0
    with open(path_to_dataset) as file:
        labels = [x.strip('\n').split() for x in file.readlines()[3:]]
    predictions = pd.read_csv(path_to_probabilities, header=None).values
    empty_dict = {k: [] for k in df.PDB.values + df.chain.values}
    for chain in labels:
        if chain[0] in empty_dict:
            empty_dict[chain[0]]=predictions[counter:counter+int(chain[1])]
        counter+=int(chain[1])
    # drop keys with missing values
    filtered_empty_dict = {
         k: v for k, v in empty_dict.items() if len(v) != 0
    }
    # warn about missing predictions
    missing_structures = [x for x in empty_dict if x not in filtered_empty_dict]
    if len(missing_structures) > 0:
        warnings.warn(f"{path_to_probabilities.name}: {*missing_structures,} predictions are missing.")
    return filtered_empty_dict


def most_likely_sequence(probability_matrix: np.array) -> str:
    """Makes protein sequence from probability matrix.

    Parameters
    ----------
    probability_matrix: np.array
        Array in shape n,20 with probabilities for each amino acid.

    Returns
    -------
    String with the sequence"""

    if len(probability_matrix) > 0:
        most_likely_seq = [
            config.acids[x] for x in np.argmax(probability_matrix, axis=1)
        ]
        return "".join(most_likely_seq)
    else:
        return ""


def format_sequence(
    df: pd.DataFrame,
    predictions: dict,
    by_fragment: bool = True,
    ignore_uncommon:bool=False,
) -> Tuple[np.array, np.array, np.array, List[List], List[List]]:
    """
    Concatenates and formats all sequences in the DataFrame for metrics calculations.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20).

    Returns
    -------
    sequece:np.array
        Array with protein sequence.
    prediction:np.array
        Array of predicted protein residues or probability matrix, shape n or n,20.
    dssp: np.array
        Array with dssp data.
    true_secondary:List[List[Union(chr,np.array)]]
        List with true sequences split by secondary structure type. Entries can be character lists or np.arrays with probability matrices. Format:[helices,sheets,loops,random].
    predicted_secondary:List[List[Union[chr,np.array]]
        List with predicted sequences split by secondary structure type. Entries can be character lists or np.arrays with probability matrices. Format:[helices,sheets,loops,random].
    """
    sequence = ""
    dssp = ""
    # Store failed structures
    failed = []
    prediction = np.empty([0, 20])
    for i, protein in df.iterrows():
        if protein.PDB + protein.chain in predictions:
            start = protein.start
            stop = protein.stop
            predicted_sequence = predictions[protein.PDB + protein.chain]
            # remove uncommon acids
            if ignore_uncommon and type(protein.uncommon_index)==list:
                protein_sequence = "".join(
                    [
                        x
                        for i, x in enumerate(protein.sequence)
                        if i not in protein.uncommon_index
                    ]
                )
                protein_dssp = "".join(
                    [
                        x
                        for i, x in enumerate(protein.dssp)
                        if i not in protein.uncommon_index
                    ]
                )
                # update start and stop indexes
                start = start - (np.array(protein.uncommon_index) <= start).sum()
                stop = stop - (np.array(protein.uncommon_index) <= stop).sum()
            else:
                protein_sequence = protein.sequence
                protein_dssp = protein.dssp

            # check length
            if len(protein_sequence) != len(predicted_sequence):
                # prediction is multimer-this is for compatibility with older EvoEF2 runs. Fixed now.
                if len(predicted_sequence) % len(protein_sequence) == 0:
                    predicted_sequence = predicted_sequence[0 : len(protein_sequence)]
                else:
                    failed.append(protein.PDB + protein.chain)
                    continue

            if by_fragment:
                protein_sequence = protein_sequence[start : stop + 1]
                protein_dssp = protein_dssp[start : stop + 1]
                predicted_sequence = predicted_sequence[start : stop + 1]

            if len(protein_sequence) == len(predicted_sequence) and len(
                protein_sequence
            ) == len(protein_dssp):
                sequence += protein_sequence
                dssp += protein_dssp
                prediction = np.concatenate(
                    [prediction, predicted_sequence], axis=0
                )
            else:
                failed.append(protein.PDB + protein.chain)
    # Get all failed structures.
    if len(failed) > 0:
        raise ValueError(
            f"Sequence, predicted sequence and dssp length do not match for these structures: {*failed,}"
        )

    sequence = np.array(list(sequence))
    dssp = np.array(list(dssp))
    # format secondary structures
    true_secondary = [[], [], [], []]
    prediction_secondary = [[], [], [], []]
    # combine secondary structures for simplicity.
    assert len(dssp)==len(sequence) and len(dssp)==len(prediction), 'format_sequence failed; dssp, sequence and prediction have different lengths.'
    for structure, truth, pred in zip(dssp, sequence, prediction):
        if structure == "H" or structure == "I" or structure == "G":
            true_secondary[0].append(truth)
            prediction_secondary[0].append(pred)
        elif structure == "E":
            true_secondary[1].append(truth)
            prediction_secondary[1].append(pred)
        elif structure == "B" or structure == "T" or structure == "S":
            true_secondary[2].append(truth)
            prediction_secondary[2].append(pred)
        else:
            true_secondary[3].append(truth)
            prediction_secondary[3].append(pred)
    return sequence, prediction, dssp, true_secondary, prediction_secondary


def score(
    df: pd.DataFrame,
    predictions: dict,
    by_fragment: bool = True,
    ignore_uncommon=False,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Concatenates and scores all predicted sequences in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20).

    Returns
    --------
    accuracy: List[float]
        List with accuracy. Format: [overal,helices,sheets,loops,random].
    top_three: List[float]
        List with top_three accuracy. Same format.
    similarity: List[float]
        List with similarity scores.
    recall: List[float]
        List with macro average recall.
    precision: List[float]
        List with macro average precision."""
    sequence, prediction, dssp, true_secondary, predicted_secondary = format_sequence(
        df, predictions, by_fragment, ignore_uncommon,
    )
    accuracy = []
    recall = []
    similarity = []
    top_three = []
    precision = []

    most_likely_seq = list(most_likely_sequence(prediction))
    accuracy.append(metrics.accuracy_score(sequence, most_likely_seq))
    recall.append(
        metrics.recall_score(
            sequence, most_likely_seq, average="macro", zero_division=0
        )
    )
    precision.append(
        metrics.precision_score(
            sequence, most_likely_seq, average="macro", zero_division=0
        )
    )
    assert len(sequence)==len(most_likely_seq)
    similarity_score = [
        1 if lookup_blosum62(a, b) > 0 else 0
        for a, b in zip(sequence, most_likely_seq)
    ]
    if len(similarity_score)>0:
        similarity.append(sum(similarity_score) / len(similarity_score))
    else:
        similarity.append(np.NaN)
    #check if probabilities or encoded sequences, encoded sequence has 0 entropy.
    is_prob=sum(entropy(prediction, base=2, axis=1))
    if is_prob:
        top_three.append(
            metrics.top_k_accuracy_score(sequence, prediction, k=3, labels=config.acids)
        )
    else:
         top_three.append(np.NaN)
    for seq_type in range(len(true_secondary)):
        # not all architectures have examples of all secondary structure types.
        if len(true_secondary[seq_type]) > 0:
            secondary_sequence = list(
                most_likely_sequence(predicted_secondary[seq_type])
            )
            accuracy.append(
                metrics.accuracy_score(true_secondary[seq_type], secondary_sequence)
            )
            recall.append(
                metrics.recall_score(
                    true_secondary[seq_type],
                    secondary_sequence,
                    average="macro",
                    zero_division=0,
                )
            )
            precision.append(
                metrics.precision_score(
                    true_secondary[seq_type],
                    secondary_sequence,
                    average="macro",
                    zero_division=0,
                )
            )
            assert len(true_secondary[seq_type])==len(secondary_sequence)
            similarity_score = [
                1 if lookup_blosum62(a, b) > 0 else 0
                for a, b in zip(true_secondary[seq_type], secondary_sequence)
            ]
            if is_prob:
                top_three.append(
                    metrics.top_k_accuracy_score(
                        true_secondary[seq_type],
                        predicted_secondary[seq_type],
                        k=3,
                        labels=config.acids,
                    )
                )
            else:
                top_three.append(np.NaN)
            similarity.append(sum(similarity_score) / len(similarity_score))
        else:
            accuracy.append(np.NaN)
            top_three.append(np.NaN)
            similarity.append(np.NaN)
            recall.append(np.NaN)
            precision.append(np.NaN)
    return accuracy, top_three, similarity, recall, precision


def score_by_architecture(
    df: pd.DataFrame,
    predictions: dict,
    by_fragment: bool = True,
    ignore_uncommon: bool = False,
) -> pd.DataFrame:
    """Groups predictions by architecture and scores each separately.

    Parameters
    ----------
    df:pd.DataFrame
        DataFrame containing predictions, cath codes and true sequences.
    predictions: dict,
        Dictionary with predictions, key is PDB+chain.
    by_fragment: bool =True
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon:bool=False
        If true, skips uncommon amino acids when formating true sequence.
    score_sequence:bool =False
        Set to True if scoring a sequence, False if scoring a probability array.

    Returns
    -------
    DataFrame with accuracy, similarity, recall and precision for each architecture type."""

    architectures = df.drop_duplicates(subset=["class", "architecture"])[
        "architecture"
    ].values
    classes = df.drop_duplicates(subset=["class", "architecture"])["class"].values
    scores = []
    names = []
    assert len(classes)==len(architectures)
    for cls, arch in zip(classes, architectures):
        accuracy, top_three, similarity, recall, precision = score(
            get_pdbs(df, cls, arch),
            predictions,
            by_fragment,
            ignore_uncommon,
        )
        scores.append(
            [accuracy[0], top_three[0], similarity[0], recall[0], precision[0]]
        )
        # lookup normal names
        names.append(config.architectures[f"{cls}.{arch}"])
    score_frame = pd.DataFrame(
        scores,
        columns=["accuracy", "top3_accuracy", "similarity", "recall", "precision"],
        index=[classes, architectures],
    )
    score_frame["name"] = names
    return score_frame


def score_each(
    df: pd.DataFrame,
    predictions: dict,
    by_fragment: bool = True,
    ignore_uncommon=False,
) -> Tuple[List[float], List[float]]:
    """Calculates accuracy and recall for each protein in DataFrame separately.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20).

    Returns
    --------
    accuracy: List[float]
        List with accuracy for each protein in DataFrame
    recall: List[float]
        List with macro average recall for each protein in Dataframe."""

    accuracy = []
    recall = []
    for i, protein in df.iterrows():
        if protein.PDB + protein.chain in predictions:
            start = protein.start
            stop = protein.stop
            predicted_sequence = predictions[protein.PDB + protein.chain]

            # remove uncommon acids
            if ignore_uncommon and type(protein.uncommon_index)==list:
                protein_sequence = "".join(
                    [
                        x
                        for i, x in enumerate(protein.sequence)
                        if i not in protein.uncommon_index
                    ]
                )
                start = start - (np.array(protein.uncommon_index) <= start).sum()
                stop = stop - (np.array(protein.uncommon_index) <= stop).sum()
            else:
                protein_sequence = protein.sequence

            # check length
            if len(protein_sequence) != len(predicted_sequence):
                # prediction is multimer
                if len(predicted_sequence) % len(protein_sequence) == 0:
                    predicted_sequence = predicted_sequence[0 : len(protein_sequence)]
                else:
                    print(
                        f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match."
                    )
                    accuracy.append(np.NaN)
                    recall.append(np.NaN)
                    continue
            if by_fragment:
                protein_sequence = protein_sequence[start : stop + 1]
                predicted_sequence = predicted_sequence[start : stop + 1]
            
            accuracy.append(
                metrics.accuracy_score(
                    list(protein_sequence),
                    list(most_likely_sequence(predicted_sequence)),
                )
            )
            recall.append(
                metrics.recall_score(
                    list(protein_sequence),
                    list(most_likely_sequence(predicted_sequence)),
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            accuracy.append(np.NaN)
            recall.append(np.NaN)

    return accuracy, recall


def get_by_residue_metrics(
    sequence: np.array, prediction: np.array,
) -> pd.DataFrame:
    """Calculates recall,precision and f1 for each amino acid.
    Parameters
    ----------
    sequence:np.array
        True sequence array with characters.
    prediction:np.array
        Predicted sequence, array with characters or probability matrix.

    Returns
    -------
    entropy_frame:pd.DataFrame
        DataFrame with recall, precision, f1 score, entropy and AUC for each amino acids.
    """

    entropy_arr = entropy(prediction, base=2, axis=1)
    # calculate auc values
    labels = LabelBinarizer().fit(config.acids).transform(sequence)
    roc_auc = []
    for i in range(len(config.acids)):
        fpr, tpr, _ = metrics.roc_curve(labels[:, i], prediction[:, i])
        roc_auc.append(metrics.auc(fpr, tpr))
    prediction = list(most_likely_sequence(prediction))

    # prevents crashing when not all amino acids are predicted
    entropy_frame = pd.DataFrame(index=config.acids)
    entropy_frame = entropy_frame.join(
        pd.DataFrame({"sequence": prediction, "entropy": entropy_arr})
        .groupby(by="sequence")
        .mean()
    )
    prec, rec, f1, sup = metrics.precision_recall_fscore_support(sequence, prediction)

    entropy_frame.loc[:, "recall"] = rec
    entropy_frame.loc[:, "precision"] = prec
    entropy_frame.loc[:, "f1"] = f1
    entropy_frame.loc[:, "auc"] = roc_auc
    return entropy_frame


def get_angles(protein: pd.Series, path_to_assemblies: Path) -> np.array:
    """Gets backbone torsion angles for protein.

    Parameters
    ----------
        protein: pd.Series
            Series containing protein info.
        path_to_assemblies: Path
            Path to the directory with biological assemblies.
    Returns
    -------
    torsion_angles: np.array
        Array with torsion angles."""

    path = path_to_assemblies / protein.PDB[1:3] / f"pdb{protein.PDB}.ent.gz"
    if path.exists():
        with gzip.open(path, "rb") as file:
            assembly = ampal.load_pdb(file.read().decode(), path=False)
            # check is assembly has multiple states, pick the first
            if isinstance(assembly, ampal.AmpalContainer):
                assembly = assembly[0]
            chain = assembly[protein.chain]
            torsion_angles = ampal.analyse_protein.measure_torsion_angles(chain)
    return torsion_angles


def format_angle_sequence(
    df: pd.DataFrame,
    predictions: dict,
    path_to_assemblies: Path,
    by_fragment: bool = False,
    ignore_uncommon=False,
) -> Tuple[str, Iterable, str, List[List[float]]]:
    """Gets Psi and Phi angles for all residues in predictions, can skip uncommon acids.


    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    path_to_assemblies: Path
        Path to the directory with biological assemblies.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
        
    Returns
    -------
    sequece:str
        Protein sequence.
    prediction: str or np.array
        Predicted protein sequence or probability matrix.
    dssp: str
        String with dssp data
    torsion:List[List[float]]
        List with torsion angles. Format:[[omega,phi,psi]].
    """

    sequence = ""
    dssp = ""
    torsion = []
    prediction = np.empty([0, 20])
    for i, protein in df.iterrows():
        if protein.PDB + protein.chain in predictions:
            start = protein.start
            stop = protein.stop
            predicted_sequence = predictions[protein.PDB + protein.chain]
            protein_angle = get_angles(protein, path_to_assemblies)

            # remove uncommon acids
            if ignore_uncommon and type(protein.uncommon_index)==list:
                protein_sequence = "".join(
                    [
                        x
                        for i, x in enumerate(protein.sequence)
                        if i not in protein.uncommon_index
                    ]
                )
                protein_dssp = "".join(
                    [
                        x
                        for i, x in enumerate(protein.dssp)
                        if i not in protein.uncommon_index
                    ]
                )
                protein_angle = [
                    x
                    for i, x in enumerate(protein_angle)
                    if i not in protein.uncommon_index
                ]
                # update start and stop indexes
                start = start - (np.array(protein.uncommon_index) <= start).sum()
                stop = stop - (np.array(protein.uncommon_index) <= stop).sum()
            else:
                protein_sequence = protein.sequence
                protein_dssp = protein.dssp

            # check length
            if len(protein_sequence) != len(predicted_sequence):
                # prediction is multimer
                if len(predicted_sequence) % len(protein_sequence) == 0:
                    predicted_sequence = predicted_sequence[0 : len(protein_sequence)]
                else:
                    print(
                        f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match."
                    )
                    continue

            if by_fragment:
                protein_sequence = protein_sequence[start : stop + 1]
                protein_dssp = protein_dssp[start : stop + 1]
                predicted_sequence = predicted_sequence[start : stop + 1]
                protein_angle = protein_angle[start : stop + 1]

            if (
                len(protein_sequence) == len(predicted_sequence)
                and len(protein_sequence) == len(protein_dssp)
                and len(protein_angle) == len(predicted_sequence)
            ):
                sequence += protein_sequence
                dssp += protein_dssp
                torsion += protein_angle
                prediction = np.concatenate(
                    [prediction, predicted_sequence], axis=0
                    )
            else:
                print(
                    f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match."
                )

    return sequence, prediction, dssp, torsion
