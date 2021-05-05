from benchmark import get_cath
from benchmark import config
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

def format_secondary(df):
    secondary=[[],[],[],[]]
    for i,chain in df.iterrows():
        for structure, residue in zip(list(chain.dssp), list(chain.sequence)):
            if structure == "H" or structure == "I" or structure == "G":
                secondary[0].append(residue)
            elif structure == "E":
                secondary[1].append(residue)
            elif structure == "B" or structure == "T" or structure == "S":
                secondary[2].append(residue)
            else:
                secondary[3].append(residue)
    return secondary


def describe_set(dataset,path_to_pdb):
    plt.ioff()
    df = get_cath.read_data("cath-domain-description-file.txt")
    filtered_df = get_cath.filter_with_user_list(df, Path(dataset))
    df_with_sequence = get_cath.append_sequence(filtered_df, Path(path_to_pdb))
    #testing set has multiple chains from the same PDB, need to drop repeats for resolution plots.
    resolution=df_with_sequence.drop_duplicates(subset=['PDB']).resolution.values
    
    fig,ax=plt.subplots(2,5,figsize=(25,10))
    hist=np.histogram(resolution,bins=6,range=(0,3))
    counts=hist[0]/len(resolution)
    
    ax[0][0].bar(range(len(counts)),counts)
    ax[0][0].set_xlabel(r'Resolution, $\AA$')
    ax[0][0].set_ylabel('Fraction of structures')
    ax[0][0].set_xticks([0,1,2,3,4,5])
    ax[0][0].set_xticklabels(['[0, 0.5)','[0.5, 1)','[1, 1.5)','[1.5, 2)','[2, 2.5)','[2.5, 3]'])
    
    colors = sns.color_palette()
    arch=filtered_df.drop_duplicates(subset=['class'])['class'].values
    grouped=filtered_df.groupby(by=['class','architecture']).count()
    
    previous_position=0
    gs = ax[0, 0].get_gridspec()
    for a in ax[0, 1:]:
        a.remove()
    ax_big = fig.add_subplot(gs[0, 1:])
    for x in arch:
        if x==1 or x==2 or x==3 or x==4:
            architectures=grouped.loc[x]
            ax_big.bar(range(previous_position,previous_position+architectures.shape[0]),architectures.PDB.values/filtered_df.shape[0],color=colors[x],label=config.classes[x])
            previous_position+=architectures.shape[0]
        #combine 4 and 6 for siplicity
        if x==6:
            architectures=grouped.loc[6]
            ax_big.bar(range(previous_position,previous_position+architectures.shape[0]),architectures.PDB.values/filtered_df.shape[0],color=colors[4])
        
    
    #get names
    cls_arch=[f"{x[0]}.{x[1]}" for x in grouped.index]
    names=[config.architectures[label] for label in cls_arch]
    
    ax_big.set_xticks(range(grouped.shape[0]))
    ax_big.set_xticklabels(names, rotation=90, fontdict={"horizontalalignment": "center", "size": 12})
    ax_big.set_ylabel('Fraction of structures')
    ax_big.set_title('CATH architectures')
    #make space for legend
    ax_big.set_xlim(-0.8,grouped.shape[0]+4)
    ax_big.legend()
    
    #get secondary structures, filtering with training set will get multiple CATH entries for the same chain.
    secondary=format_secondary(df_with_sequence)
    #plot residue distribution
    ss_types=["Helices", "Sheets", "Structured loops", "Random"]
    for x in range(len(secondary)):
        ax[1][x].bar(config.acids,np.unique(secondary[x],return_counts=True)[1]/len(secondary[x]))
        ax[1][x].set_ylabel('Fraction of structures')
        ax[1][x].set_xlabel('Amino acids')
        ax[1][x].set_title(ss_types[x])
    #flatten the list
    all_structures=[x for y in secondary for x in y]
    ax[1][4].bar(config.acids,np.unique(all_structures,return_counts=True)[1]/len(all_structures))
    ax[1][4].set_ylabel('Fraction of structures')
    ax[1][4].set_xlabel('Amino acids')
    ax[1][4].set_title('All structures')
    plt.tight_layout()
    plt.savefig(dataset+'.pdf')
    
describe_set("/home/s1706179/project/sequence-recovery-benchmark/nmr_benchmark.txt","/home/shared/datasets/pdb/")