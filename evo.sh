#! /bin/sh
#unzip into working dir, keep original file.
cd /home/s1706179/project/sequence-recovery-benchmark/temp 
gzip -dkc < $1 > /home/s1706179/project/sequence-recovery-benchmark/temp/$2.pdb1
#get a specified number of sequences and print results into terminal
for i in $(seq $4); do
    /home/s1706179/project/EvoEF2/EvoEF2 --command=ProteinDesign --ppint --design_chains=$3 --pdb=/home/s1706179/project/sequence-recovery-benchmark/temp/$2.pdb1 > /dev/null
    cat /home/s1706179/project/sequence-recovery-benchmark/temp/$2_bestseq.txt >> /home/s1706179/project/sequence-recovery-benchmark/evo_dataset/$2$3.txt
    rm -f $2*
done