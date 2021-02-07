import pandas as pd
from benchmark import config
import ampal
from benchmark import get_cath
import gzip
from pathlib import Path
import numpy as np

def _annotate_ampalobj_with_data_tag(
    ampal_structure,
    data_to_annotate,
    tags,
) -> ampal.assembly:
    """
    Assigns a data point to each residue equivalent to the prediction the
    tag value. The original value of the tag will be reset to the minimum value
    to allow for a more realistic color comparison.
    Parameters
    ----------
    ampal_structure : ampal.Assembly or ampal.AmpalContainer
        Ampal structure to be modified. If an ampal.AmpalContainer is passed,
        this will take the first Assembly in the ampal.AmpalContainer `ampal_structure[0]`.
    data_to_annotate : numpy.ndarray of numpy.ndarray of floats
        Numpy array with data points to annotate (x, n) where x is the
        numer of arrays with data points (eg, [ entropy, accuracy ] ,
        x = 2n) and n is the number of residues in the structure.
    tags : t.List[str]
        List of string tags of the pdb object (eg. "b-factor")
    Returns
    -------
    ampal_structure : Assembly
        Ampal structure with modified B-factor values.

    Notes
    -----
    Same as _annotate_ampalobj_with_data_tag from TIMED but can deal with missing unnatural amino acids for compatibility with EvoEF2."""

    assert len(tags) == len(
        data_to_annotate
    ), "The number of tags to annotate and the type of data to annotate have different lengths."
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(ampal_structure, ampal.AmpalContainer):
        warnings.warn(
            f"Selecting the first state from the NMR structure {ampal_structure.id}"
        )
        ampal_structure = ampal_structure[0]

    if len(data_to_annotate) > 1:
        assert len(data_to_annotate[0]) == len(data_to_annotate[1]), (
            f"Data to annotatate has shape {len(data_to_annotate[0])} and "
            f"{len(data_to_annotate[1])}. They should be the same."
        )

    for i, tag in enumerate(tags):
        # Reset existing values:
        for atom in ampal_structure.get_atoms(ligands=True, inc_alt_states=True):
            atom.tags[tag] = np.min(data_to_annotate[i])

    # Apply data as tag:
    for chain in ampal_structure:
        for i, tag in enumerate(tags):

            # Check if chain is Polypeptide (it might be DNA for example...)
            if isinstance(chain, ampal.Polypeptide):
                if len(chain) != len(data_to_annotate[i]):
                    #EvoEF2 predictions drop uncommon amino acids
                    if len(chain)-chain.sequence.count('X')==len(data_to_annotate[i]):
                        for residue in chain:
                            counter=0
                            if ampal.amino_acids.get_aa_letter(residue)=='X':
                                continue
                            else:
                                for atom in residue:
                                    atom.tags[tag] =  data_to_annotate[i][counter]
                                    counter+=1
                    else:
                        print('Length is not equal')
                        return  
                for residue, data_val in zip(chain, data_to_annotate[i]):
                    for atom in residue:
                        atom.tags[tag] = data_val

    return ampal_structure

def show_accuracy(df:pd.DataFrame, pdb:str, output:str):
    accuracy=[]
    sequence=df[df.PDB==pdb].sequence.values[0]
    predictions=df[df.PDB==pdb].predicted_sequences.values[0]
    for resa,resb in zip(sequence,predictions):
        #correct predictions are given constant score so they stand out in the figure.
        #e.g., spectrum b, blue_white_red, maximum=6,minimum=-6 gives nice plots. Bright red shows correct predictions
        #Red shades indicate substitutions with positive score, white=0, blue shades show substiutions with negative score.
        if resa==resb:
            accuracy.append(6)
        #incorrect predictions will be coloured by blossum62 score.
        else:
            accuracy.append(get_cath.lookup_blosum62(resa,resb))
    path_to_protein=config.PATH_TO_ASSEMBLIES/pdb[1:3]/f"pdb{pdb}.ent.gz"
    with gzip.open(path_to_protein, "rb") as protein:
        assembly = ampal.load_pdb(protein.read().decode(), path=False)
    ##add entropy in the future
    curr_annotated_structure = _annotate_ampalobj_with_data_tag(assembly,[accuracy],tags=["bfactor"])
    with open(output, "w") as f:
        f.write(curr_annotated_structure.pdb)