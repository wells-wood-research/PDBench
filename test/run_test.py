from benchmark import get_cath
import numpy as np


def test_load_CATH():
    cath_location = "/home/s1706179/project/sequence-recovery-benchmark/test/test.txt"
    cath_df = get_cath.read_data(cath_location)

    # check shape
    assert cath_df.shape == (12, 8), "DataFrame shape is incorrect"
    pdbs = get_cath.get_pdbs(cath_df, 2, 40, 10, 10)
    assert pdbs.shape == (7, 8), "PDB shape is incorrect"

    # check pdb codes
    assert np.all(
        pdbs.PDB.value_counts().values == [4, 2, 1]
    ), "PDBs returned incorrectly"

    # check sequence
    pdbs = get_cath.append_sequence(pdbs)
    assert pdbs.shape == (7, 10), "Sequence appended incorrectly"

    example_with_insertion = get_cath.get_pdbs(cath_df, 2, 40, 20, 10)
    example_with_insertion = get_cath.append_sequence(example_with_insertion)
    assert (
        example_with_insertion[example_with_insertion.PDB == "1cea"].sequence.values
        == "ECKTGNGKNYRGTMSKTKNGITCQKWSSTSPHRPRFSPATHPSEGLEENYCRNPDNDPQGPWCYTTDPEKRYDYCDILEC"
    ), "Sequence assigned incorrectly"

    # check pisces
    new_df = get_cath.filter_with_user_list(
        cath_df,
        "/home/shared/datasets/pisces/cullpdb_pc20_res1.6_R0.25_d200702_chains3689",
        ispisces=True,
    )
    index = new_df.PDB.values
    assert index[0] == "3su6" and index[1] == "1a62", "Pisces returned incorect PDBs"

    # check sequence recovery rates
    new_df = get_cath.append_sequence(new_df)
    df_with_predictions = get_cath.load_predictions(
        new_df.iloc[1:2], "/home/s1706179/project/sequence-recovery-benchmark/test/"
    )
    list_with_scores = get_cath.score(df_with_predictions, by_fragment=False)
    assert (
        abs(list_with_scores[0] - 0.344262) <= 0.001
    ), "Sequence recovery calculated incorrectly"

    assert (
        abs(list_with_scores[1] - 0.483607) <= 0.001
    ), "Sequence similarity calculated incorrectly"

    assert (
        abs(list_with_scores[6] - 0.28000) <= 0.001
    ), "Sequence recovery for random coils calculated incorrectly"
