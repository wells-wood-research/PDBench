from benchmark import get_cath
import numpy as np
from pathlib import Path
import os
import sys


location=Path(__file__).parent.resolve()
PATH_TO_PDB=Path(sys.argv[1])
assert (PATH_TO_PDB.exists()), 'PDB directory is missing!'

def test_load_CATH():
    """Tests basic benchmark functions - loading data, calculating metrics, ect."""
    
    cath_location = location.parents[0]/"cath-domain-description-file.txt"
    cath_df = get_cath.read_data(cath_location)
    new_df=get_cath.filter_with_user_list(cath_df,location/'test_set.txt')
    # check shape
    assert new_df.shape == (10, 8), "DataFrame shape is incorrect"
    pdbs = get_cath.get_pdbs(new_df,1,20)
    assert pdbs.shape == (1, 8), "Filtered shape is incorrect"

    # check sequence, 1a41A02 fragment.
    new_df = get_cath.append_sequence(new_df,PATH_TO_PDB)
    fragment_sequence=new_df[new_df.PDB == "1a41"]
    sequence=fragment_sequence.sequence.values[0]
    start=fragment_sequence.start.values[0]
    stop=fragment_sequence.stop.values[0]
    assert (sequence[start:stop+1] == "IRIKDLRTYGVNYTFLYNFWTNVKSISPLPSPKKLIALTIKQTAEVVGHTPSISKRAYMATTILEMVKDKNFLDVVSKTTFDEFLSIVVDHVKS"
    ), "Sequence assigned incorrectly"

    #check sequence, 1cruA00 fragment
    fragment_sequence=new_df[new_df.PDB == "1cru"]
    sequence=fragment_sequence.sequence.values[0]
    start=fragment_sequence.start.values[0]
    stop=fragment_sequence.stop.values[0]
    assert (sequence[start:stop+1] == "DVPLTPSQFAKAKSENFDKKVILSNLNKPHALLWGPDNQIWLTERATGKILRVNPESGSVKTVFQVPEIVNDADGQNGLLGFAFHPDFKNNPYIYISGTFKNPKSKELPNQTIIRRYTYNKSTDTLEKPVDLLAGLPSSKDHQSGRLVIGPDQKIYYTIGDQGRNQLAYLFLPNQAQHTPTQQELNGKDYHTYMGKVLRLNLDGSIPKDNPSFNGVVSHIYTLGHRNPQGLAFTPNGKLLQSEQGPNSDDEINLIVKGGNYGWPNVAGYKDDSGYAYANYSAAANKSIKDLAQNGVKVAAGVPVTKESEWTGKNFVPPLKTLYTVQDTYNYNDPTCGEMTYICWPTVAPSSAYVYKGGKKAITGWENTLLVPSLKRGVIFRIKLDPTYSTTYDDAVPMFKSNNRYRDVIASPDGNVLYVLTDTAGNVQKDDGSVTNTLENPGSLIKFT"
    ), "Sequence assigned incorrectly"
    
    #load predictions
    path_to_file=Path(location/'test_data.csv')
    with open(path_to_file.with_suffix('.txt')) as datasetmap:
      predictions = get_cath.load_prediction_matrix(new_df, path_to_file.with_suffix('.txt'), path_to_file)
                
    # check accuracy and recall
    accuracy,recall=get_cath.score_each(new_df,predictions,by_fragment=True)
    assert (
        abs(accuracy[0] - 0.298) <= 0.001
    ), "Sequence recovery calculated incorrectly"
    
    accuracy,recall=get_cath.score_each(new_df,predictions,by_fragment=True)
    assert (
        abs(recall[3] - 0.384) <= 0.001
    ), "Macro-recall calculated incorrectly"
    
def test_command_line():
    """Tests command line interface"""
    os.system(f'python {location.parents[0]/"run_benchmark.py"} --dataset {location/"test_set.txt"} --path_to_pdb {PATH_TO_PDB} --path_to_models {location} --training_set {location/"trainingset.txt"}')
    assert (Path(location/'test_data.csv.pdf').exists()), 'Failed to produce plots!'
    assert (Path(location/'test_data_1a41.pdb').exists()), 'Failed to produce PDB with accuracy and entropy!'
if __name__=='__main__':
  test_load_CATH()
  test_command_line()
