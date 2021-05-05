#! /bin/sh
#inputs:
#$1 pdb name
#$2 chain
#$3 number of sequences to predict
#$5 path to working dir
PATH_TO_EVOEF2=/home/s1706179/project/EvoEF2/EvoEF2
cd $4
#get a specified number of sequences and print results to a .txt file
for i in $(seq $3); do
    $PATH_TO_EVOEF2 --command=ProteinDesign --ppint --design_chains=$2 --pdb=$1.pdb1 > /dev/null
    cat $4/$1_bestseq.txt > $4/results/$1$2.txt
    #remove working files to save space
    rm $1*
done