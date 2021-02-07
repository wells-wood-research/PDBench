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

def read_data(CATH_file: str) -> pd.DataFrame:
    """If CATH .csv exists, loads the DataFrame. If CATH .txt exists, makes DataFrame and saves it.

    Parameters
    ----------
    CATH_file: str
        PATH to CATH .txt file.

    Returns
    -------
    DataFrame containing CATH and PDB codes."""
    path=Path(CATH_file)
    #load .csv if exists, faster than reading .txt
    if path.with_suffix('.csv').exists():
        df = pd.read_csv(path.with_suffix('.csv'), index_col=0)
        # start, stop needs to be str
        df["start"] = df["start"].apply(str)
        df["stop"] = df["stop"].apply(str)
        return df

    else:
        cath_info = []
        temp = []
        start_stop = []
        # ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/
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
        df.to_csv(path.with_suffix('.csv'))
        return df

def tag_dssp_data(assembly: ampal.Assembly):
    """Same as ampal.dssp.tag_dssp_data(), but fixed a bug with insertions. Tags each residue in ampal.Assembly with secondary structure.

    Parameters
    ----------
    assembly: ampal.Assembly
        Protein assembly."""

    dssp_out = ampal.dssp.run_dssp(assembly.pdb, path=False)
    dssp_data = ampal.dssp.extract_all_ss_dssp(dssp_out, path=False)
    for i,record in enumerate(dssp_data):
        rnum, sstype, chid, _, phi, psi, sacc = record
        #deal with insertions
        if len(chid)>1:
            for i,res in enumerate(assembly[chid[1]]):
                if res.insertion_code==chid[0] and assembly[chid[1]][i].tags=={}:
                    assembly[chid[1]][i].tags['dssp_data'] = {
                    'ss_definition': sstype,
                    'solvent_accessibility': sacc,
                    'phi': phi,
                    'psi': psi
                    }
                    break
                
        else:
            assembly[chid][str(rnum)].tags['dssp_data'] = {
            'ss_definition': sstype,
            'solvent_accessibility': sacc,
            'phi': phi,
            'psi': psi
            }

def get_sequence(series: pd.Series) -> str:
    """Gets a sequence of from PDB file, CATH fragment indexes and secondary structure labels.

    Parameters
    ----------
    series: pd.Series
        Series containing one CATH instance.
    path:str
        Path to PDB dataset directory.

    Returns
    -------
    If PDB exists, returns sequence, dssp sequence, and start and stop index for CATH fragment. If not, returns np.NaN

    Notes
    -----
    Unnatural amino acids are removed"""
    
    #path=config.PATH_TO_PDB/series.PDB[1:3]/f"pdb{series.PDB}.ent.gz"
    path=config.PATH_TO_ASSEMBLIES/series.PDB[1:3]/f"{series.PDB}.pdb1.gz"
            
    if path.exists():
        with gzip.open(path,"rb") as protein:
            assembly = ampal.load_pdb(protein.read().decode(), path=False)
            #check is assembly has multiple states, pick the first
            if isinstance(assembly,ampal.assembly.AmpalContainer):
                if assembly[0].id.count('_state_')>0:
                    assembly=assembly[0]
            # convert pdb res id into sequence index,
            # some files have discontinuous residue ids so ampal.get_slice_from_res_id() does not work
            start = 0
            stop = 0
            # if nmr structure, get 1st model
            if isinstance(assembly, ampal.AmpalContainer):
                assembly = assembly[0]
            try:
                chain = assembly[series.chain]
            except KeyError:
                return np.NaN, np.NaN, np.NaN, np.NaN
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
        # Evo2EF skips unnatural amino acids but AMPAL puts 'X'. string length and start/stop index must be checked
        filtered_sequence = "".join([x for x in chain.sequence if x != "X"])
        filtered_fragment = "".join(
            [x for x in chain[start : (stop + 1)].sequence if x != "X"]
        )
        new_start = filtered_sequence.find(filtered_fragment)
        new_stop = new_start + len(filtered_fragment) - 1
        try:
            tag_dssp_data(assembly)
            dssp = "".join(
            [x.tags['dssp_data']['ss_definition'] for x in chain if x.id != "X"]
            )
        #dssp can fail on some broken residues(e.g., missing side chain)     
        except KeyError:
            dssp=np.NaN
        return filtered_sequence, dssp, new_start, new_stop
    # some pdbs are obsolete or broken, return np.NaN
    else:
        print(f"{series.PDB}.pdb1 is missing.")
        return np.NaN, np.NaN, np.NaN, np.NaN

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
        working_copy.loc[:, "dssp"],
        working_copy.loc[:, "start"],
        working_copy.loc[:, "stop"],
    ) = zip(*[get_sequence(x) for i, x in df.iterrows()])
    # remove missing entries
    working_copy.dropna(inplace=True)
    # change index from float to int
    working_copy.loc[:, "start"] = working_copy["start"].apply(int)
    working_copy.loc[:, "stop"] = working_copy["stop"].apply(int)
    return working_copy

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

    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(config.ts50)]
    
def filter_with_user_list(df: pd.DataFrame, path: str, ispisces:bool = False)->pd.DataFrame:
    """Selects PDB chains specified in .txt file.
    Parameters
    ----------
    df: pd.DataFrame
        CATH info containing dataframe
    path: str
        Path to .txt file
    ispisces:bool = False
        Reads pisces formating if True, otherwise pdb+chain, e.g., 1a2bA\n.
    
    Returns
    -------
    DataFrame with selected chains,duplicates are removed."""
    path=Path(path)
    with open(path) as file:
        if ispisces:
            filtr = [x.split()[0] for x in file.readlines()[1:]]
        else:
            filtr=[x.upper().strip('\n') for x in file.readlines()]
    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(filtr)].drop_duplicates(subset=['PDB','chain'])

def lookup_blosum62(res_true: str, res_prediction: str) -> int:
    """Returns score from the matrix.

    Parameters
    ----------
    a: str
        First residue code.
    b: str
        Second residue code.

    Returns
    --------
    Score from the matrix."""

    if (res_true, res_prediction) in config.blosum62.keys():
      return config.blosum62[res_true, res_prediction] 
    else:
      return config.blosum62[res_prediction, res_true]

def secondary_score(true_seq: np.array, predicted_seq: np.array, dssp: str) -> list:
    """Calculates sequence recovery rate for each secondary structure type.

    Parameters
    ----------
    true_seq: str,
        True sequence.
    predicted_seq: str
        Predicted sequence.
    dssp: str
        string with dssp resutls

    Returns
    -------
    List with sequence recovery for helices, beta sheets, random coils and structured loops"""

    alpha=0
    alpha_counter=0
    beta=0
    beta_counter=0
    random=0
    random_counter=0
    loop=0
    loop_counter=0

    for structure,result in zip(dssp,true_seq==predicted_seq):
        if structure=="H" or structure=="I" or structure=="G":
            alpha_counter+=1
            if result:
                alpha+=1
        if structure=='E':
            beta_counter+=1
            if result:
                beta+=1
        if structure=="B" or structure=="T" or structure=="S":
            loop_counter+=1
            if result:
                loop+=1
        if structure==' ':
            random_counter+=1
            if result:
                random+=1
            
    if alpha_counter!=0:
        alpha=alpha/alpha_counter
    else:
        alpha=np.NaN
    
    if beta_counter!=0:
        beta=beta/beta_counter
    else:
        beta=np.NaN
    
    if loop_counter!=0:
        loop=loop/loop_counter
    else:
        loop=np.NaN
    
    if random_counter!=0:
        random=random/random_counter
    else:
        random=np.NaN
    return alpha,beta,loop,random
    


def run_Evo2EF(pdb: str, chain: str, number_of_runs: str, working_dir: Path):
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
      Dir where to store temporary files and results

    Returns
    -------
    Nothing."""

    #evo.sh must be in the same directory as this file.
    p = subprocess.Popen(
        [
            os.path.dirname(os.path.realpath(__file__))+"/evo.sh",
            pdb,
            chain,
            number_of_runs,
            working_dir,
        ]
    )
    p.wait()
    print(f"{pdb}{chain} done.")


def multi_Evo2EF(df: pd.DataFrame, number_of_runs: int, working_dir: str, max_processes: int = 8):
    """Runs Evo2EF on all PDB chains in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with PDB and chain codes.
    number_of_runs: int
        Number of sequences to be generated for each PDB file.
    max_processes: int = 8
        Number of cores to use, default is 8.
    working_dir: str
      Dir where to store temporary files and results.
      
    Returns
    --------
    Nothing."""

    inputs = []
    # remove duplicated chains
    df = df.drop_duplicates(subset=["PDB", "chain"])

    #check if working directory exists. Make one if doesn't exist.
    working_dir=Path(working_dir)
    if not working_dir.exists():
      os.makedirs(working_dir)
    if not (working_dir/'results').exists():
      os.makedirs(working_dir/'/results')

    for i, protein in df.iterrows():
        with gzip.open(config.PATH_TO_ASSEMBLIES/protein.PDB[1:3]/f"{protein.PDB}.pdb1.gz") as file:
            assembly = ampal.load_pdb(file.read().decode(), path=False)
        #fuse all states of the assembly into one state to avoid EvoEF2 errors.
        empty_polymer=ampal.Assembly()
        chain_id=[]
        for polymer in assembly:
            for chain in polymer:
                empty_polymer.append(chain)
                chain_id.append(chain.id)
        #relabel chains to avoid repetition
        str_list=string.ascii_uppercase.replace(protein.chain, "")
        index=chain_id.index(protein.chain)
        chain_id=list(str_list[:len(chain_id)])
        chain_id[index]=protein.chain
        empty_polymer.relabel_polymers(chain_id)
        pdb_text=empty_polymer.make_pdb(alt_states=False,ligands=False)
        #writing new pdb with AMPAL fixes most of the errors with EvoEF2.
        with open((working_dir/protein.PDB).with_suffix(".pdb1"),'w') as pdb_file:
            pdb_file.write(pdb_text)
        inputs.append((protein.PDB, protein.chain, str(number_of_runs),working_dir))

    with multiprocessing.Pool(max_processes) as P:
        P.starmap(run_Evo2EF, inputs)


def load_predictions(df: pd.DataFrame,path:str) -> pd.DataFrame:
    """Loads predicted sequences from .txt to CATH DataFrame, drops entries for which sequence prediction fails.
        Parameters
        ----------
        df: pd.DataFrame
            CATH dataframe
        path:str
            Path to prediction directory.
        
        Returns
        -------
        DataFrame with appended prediction."""

    predicted_sequences = []
    path=Path(path)
    for i, protein in df.iterrows():
        prediction_path = path/f"{protein.PDB}{protein.chain}.txt"
        # check for empty and missing files
        if prediction_path.exists() and os.path.getsize(prediction_path)>0:
            with open(prediction_path) as prediction:
                seq = prediction.readline().split()[0]
                if seq == '0':
                    print(
                        f"{protein.PDB}{protein.chain} prediction does not exits, EvoEF2 returned 0."
                    )
                    seq=np.NaN
                elif len(seq) != len(protein.sequence):
                    # assume that biological assembly is a multimer
                    if len(seq) % len(protein.sequence) == 0:
                        seq = seq[0 : len(protein.sequence)]    
                    else:
                        print(
                            f"{protein.PDB}{protein.chain} prediction and true sequence have different length."
                        )
                        seq=np.NaN
        else:
            print(f"{protein.PDB}{protein.chain} prediction does not exits.")
            seq=np.NaN
        predicted_sequences.append(seq)
    #avoid changing the original dataframe
    df=df.copy()
    df["predicted_sequences"] = predicted_sequences
    df.dropna(inplace=True)
    # drop empty lists
    return df


def score(
    df: pd.DataFrame, by_fragment: bool=True
) -> list:
    """Concatenates and scores all predicted sequences in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.

    Returns
    --------
    A list with sequence recovery, similarity, f1 and secondary structure scores"""

    sequence=''
    prediction=''
    dssp=''
    for i,x in df.iterrows():
        if by_fragment:
            sequence+=x.sequence[x.start:x.stop+1]
            prediction+=x.predicted_sequences[x.start:x.stop+1]
            dssp+=x.dssp[x.start:x.stop+1]
        else:
            sequence+=x.sequence
            prediction+=x.predicted_sequences
            dssp+=x.dssp
    sequence=np.array(list(sequence))
    prediction=np.array(list(prediction))
    dssp=np.array(list(dssp))
    #check if length match
    assert len(sequence)==len(prediction), 'Sequence and predicted sequence lengths are not equal.'
    sequence_recovery=metrics.accuracy_score(sequence,prediction)
    
    f1=metrics.f1_score(sequence,prediction,average='macro')
    
    similarity_score=[1 if lookup_blosum62(a,b)>0 else 0 for a,b in zip(sequence,prediction)]
    similarity_score=sum(similarity_score)/len(similarity_score)
    
    alpha,beta,loop,random=secondary_score(prediction,sequence,dssp)
    
    return sequence_recovery,similarity_score,f1,alpha,beta,loop,random

def score_by_architecture(df:pd.DataFrame,by_fragment: bool=True)->pd.DataFrame:
    """Groups the predictions by architecture and scores each separately.

        Parameters
        ----------
        df:pd.DataFrame
            DataFrame containing predictions, cath codes and true sequences.
        by_fragment: bool =True
            If true scores only CATH fragments, if False, scores entire chain.
        
        Returns
        -------
        DataFrame with accuracy, similarity, f1, and secondary structure accuracy."""

    architectures=df.drop_duplicates(subset=['class','architecture'])['architecture'].values
    classes=df.drop_duplicates(subset=['class','architecture'])['class'].values
    scores=[]
    names=[]
    for cls,arch in zip(classes,architectures):
        scores.append(score(get_pdbs(df,cls,arch),by_fragment))
        #lookup normal names
        names.append(config.architectures[f"{cls}.{arch}"])
    score_frame=pd.DataFrame(scores,columns=['accuracy','similarity','f1','alpha','beta','struct_loops','random'],index=[classes,architectures]) 
    #get meaningful names
    score_frame['name']=names
    return score_frame
