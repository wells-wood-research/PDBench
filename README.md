# PDBench: Evaluating Computational Methods for Protein Sequence Design
PDBench is a dataset and software package for evaluating fixed-backbone sequence design algorithms. The structures included in PDBench have been chosen to account for the diversity and quality of observed protein structures, giving a more holistic view of performance.

## Installation
If you don't have Cython, please run ```pip install Cython``` first.
```sh
git clone https://github.com/wells-wood-research/sequence-recovery-benchmark.git
cd sequence-recovery-benchmark/
pip install .
```

### Note
You can download PDB database from https://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/.

PDBench uses DSSP to predict secondary structure type; DSSP can be installed from https://github.com/PDB-REDO/dssp.

### Testing
Run ```python test/run_test.py /path/to/pdb/```.

## Dataset
Our benchmark set contains 595 protein structures spanning 40 protein architectures that are clustered into 4 fold classes, as presented in the [CATH database](https://www.cathdb.info/). The `special` category contains proteins that do not have regular secondary structure.
Crystal structures with <3Å resolution and up to 90% sequence identity were carefully chosen to cover the structural diversity present in the PDB. This ensures that the performance is evaluated on high- and low-quality inputs and the results are not biased towards the most common protein architectures. Structure list can be found in **/dataset_visualization/crystal_structure_benchmark.txt**
![benchmark](https://user-images.githubusercontent.com/77202997/138559253-590a6536-064f-4e72-b2ca-ffa852cc4fd9.png)

## Usage
To evaluate models in my_models directory using PDB structures listed in benchmark_set.txt:

1. Generate predictions and format them as a **N x 20**  matrix where **N** is the number of amino acids in the polypeptide chain. Each row in the matrix encodes prediction probabilities across the 20 canonical amino-acid classes. 
2. Concatenate all predictions and save them as CSV file.
3. For each model, generate a dataset map with the list of PDB codes and sequence lengths of structures in CSV file.
4. Run:
```sh
python run_benchmark.py --dataset benchmark_set.txt --path_to_pdb /pdb/ --path_to_models /my_models/
```
Please see the **/examples/** for input and output files.

## Evaluation metrics
We calculate four groups of metrics: 

**1)** recall, precision, AUC, F1 score, Shannon's entropy, confusion matrix and prediction bias **for each amino acid class**; 

**2)** accuracy, macro-precision, macro-recall, similarity and top-3 accuracy **for each protein chain**; 

**3)** accuracy, macro-precision, macro-recall, similarity and top-3 accuracy **for each secondary structure type**;  

**4)** accuracy, macro-precision, macro-recall, similarity and top-3 accuracy **for each protein architecture**. 

All metrics except similarity and prediction bias are calculated with SciKit-Learn (see [here](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) for more info about the metrics). Prediction bias is a metric measuring the discrepancy between the occurrence of a residue and the number of times it is predicted.

We use macro-averaged metrics to cap the performance per amino acid thus avoiding models predicting the most common amino acids (eg. Leucine) to obtain high performance through over-prediction.

Accuracy-based metrics are useful, but there is functional redundancy between amino acids, as many side chains have similar chemistry. The similarity of amino acids can be determined by the relative frequency of substitution of one amino acid for another observed in natural structures, using substitution matrices such as BLOSUM62, which we combine into a similarity score for the sequence.

### Examples

**1.** We tested two state-of-the-art physics-based models: [Rosetta](https://www.nature.com/articles/s41592-020-0848-2) and [EvoEF2](https://pubmed.ncbi.nlm.nih.gov/31588495/), and three DNN models: [ProDCoNN](https://onlinelibrary.wiley.com/doi/full/10.1002/prot.25868), [DenseCPD](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00043) and [TIMED](https://github.com/wells-wood-research/timed-design).

![readme](https://user-images.githubusercontent.com/17524568/162149140-e29614e5-6575-4118-9025-44eb31172f14.png)

**2.** The composition of the amino acids in proteins is not uniformly distributed, and can vary significantly between different protein folds. Therefore, we investigated the effect of balancing amino acid distributions in the dataset prior to training.  When TIMED and ProDCoNN were trained without balancing, the prediction bias increased for the most common amino acids, while macro-recall decreased. This was extremely evident in α-helices.

<img src="https://user-images.githubusercontent.com/77202997/138561002-0ab8be86-3265-447a-88ce-bfada2f6f66d.png">

**3.** F1 score for amino acids shows that hydrophobic amino acids such as leucine or glycine are predicted very accurately while hydrophilic amino acids are predicted less accurately. This trend holds for physics-based and DNN models.
![F1_score](https://user-images.githubusercontent.com/77202997/138563139-61bcfc78-f720-4bb2-8b6d-1d6d16e69ef0.png)



For more in-depth analysis, see our [paper](https://arxiv.org/abs/2109.07925). To see all the other metrics produced for different models, for all secondary structure types and all 40 protein architectures, go to **/examples**.

## Optional features
```sh
--training_set TEXT    Path to TXT file with the training set, training structures will be excluded from evaluation.
 
 --torsions            Produces predicted, true and difference Ramachandran plots for each model.
 ```
<img src="https://user-images.githubusercontent.com/77202997/138558548-0ce1a7ef-fd51-473d-a811-4cebe8080c02.png" width="350" align="left" />Torsion angle frequency difference plots between true and predicted amino acids are useful for detecting unreasonably placed amino acids. For example, we found that although glycine destabilizes α-helices, one of our models had significantly increased Glycine frequency in α-helical well (-60°, -60°).

 
If your model skips non-canonical amino acids (e.g. EvoEF2), you can set ignore_uncommon True in the dataset map. This will remove non-canonical acids from calculations.


<img src="https://user-images.githubusercontent.com/77202997/138557135-7a1441a8-f72d-45c6-9e57-503f23e30ca3.png" width="300" align="right" />If you want to visualize prediction accuracy and entropy on a particular protein chains, you can list them in the dataset map, e.g. ```include_pdbs 1a41A 1l2sB```. Open created PDB files with PyMol and show accuracy:```spectrum q, blue_white_red, maximum=6,minimum=-6```, entropy: ```cartoon putty```.

## Cite This Work

```
@article{castorina_2023_pdbench,
    author = {Castorina, Leonardo V and Petrenas, Rokas and Subr, Kartic and Wood, Christopher W},
    title = "{PDBench: Evaluating Computational Methods for Protein-Sequence Design}",
    journal = {Bioinformatics},
    year = {2023},
    month = {01},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btad027},
    url = {https://doi.org/10.1093/bioinformatics/btad027},
    note = {btad027},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btad027/48691588/btad027.pdf},
}
```
