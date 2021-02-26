#! /bin/sh
#inputs:
#$1 pdb name
#$2 chain
#$3 number of sequences to predict
#$4 path to working dir
#$5 Path to EvoEF2

cd $4
#get a specified number of sequences and print results to a .txt file
for i in $(seq $3); do
    $5 --command=ProteinDesign --ppint --design_chains=$2 --pdb=$1.pdb1 > /dev/null
    cat $4/$1_bestseq.txt > $4/results/$1$2.txt
    #remove working files to save space
    rm $1*
done