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
    get_cath.append_sequence(pdbs)
    assert pdbs.shape == (7, 9), "Sequence appenden incorrectly"

    example_with_insertion = get_cath.get_pdbs(cath_df, 2, 40, 20, 10)
    get_cath.append_sequence(example_with_insertion)
    assert (
        example_with_insertion[example_with_insertion.PDB == "1cea"].sequence.values[0]
        == "ECKTGNGKNYRGTMSKTKNGITCQKWSSTSPHRPRFSPATHPSEGLEENYCRNPDNDPQGPWCYTTDPEKRYDYCDILEC"
    ), "Sequence assigned incorrectly"
