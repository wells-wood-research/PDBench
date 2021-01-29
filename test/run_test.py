from benchmark import get_cath
import numpy as np


def test_load_CATH():
    working_dir = "/home/s1706179/project/sequence-recovery-benchmark/test/"
    cath_df = get_cath.read_data("test", working_dir)

    # check shape
    assert cath_df.shape == (11, 8), "DataFrame shape is incorrect"
    pdbs = get_cath.get_pdbs(cath_df, 2, 40, 10, 10)
    assert pdbs.shape == (7, 8), "PDB shape is incorrect"

    # check pdb codes
    assert np.all(
        pdbs.PDB.value_counts().values == [4, 2, 1]
    ), "PDBs returned incorrectly"

    # check sequence
    pdbs=get_cath.append_sequence(pdbs)
    assert pdbs.shape == (7, 10), "Sequence appended incorrectly"

    example_with_insertion = get_cath.get_pdbs(cath_df,2,40,20,10)
    example_with_insertion=get_cath.append_sequence(example_with_insertion)
    return example_with_insertion
    assert (
        example_with_insertion[example_with_insertion.PDB == "1cea"].sequence.values
        == "ECKTGNGKNYRGTMSKTKNGITCQKWSSTSPHRPRFSPATHPSEGLEENYCRNPDNDPQGPWCYTTDPEKRYDYCDILEC"
    ), "Sequence assigned incorrectly"
    

    # check pisces
    new_df = get_cath.filter_with_pisces(cath_df, 20, 1.6)
    index = new_df.PDB.values
    assert index[0] == "3su6" and index[1] == "1oai", "Pisces returned incorect PDBs"

    # check sequence recovery rate
    example_with_sequence = get_cath.get_pdbs(new_df, 1, 10, 8, 10)
    example_with_sequence=get_cath.append_sequence(example_with_sequence)
    df_with_predictions = get_cath.load_predictions(example_with_sequence)
    get_cath.score(df_with_predictions)
    assert (
        abs(df_with_predictions['sequence_recovery'].iloc[0] - 0.266949) <= 0.001
    ), "Sequence recovery calculated incorrectly"

    assert (
        abs(df_with_predictions['similarity_score'].iloc[0] - 0.508475) <= 0.001
    ), "Sequence similarity calculated incorrectly"
    
    assert (
        abs(df_with_predictions['random_coil'].iloc[0] - 0.479167) <= 0.001
    ), "Sequence similarity for random coils calculated incorrectly"