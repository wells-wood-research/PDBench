import numpy as np
import pandas as pd
import ampal 
import gzip

classes={'1'   :'Mainly Alpha',
'2'    :'Mainly Beta',
'3'    :'Alpha Beta',
'4'    :'Few Secondary Structures',
'6'    :'Special'}

architectures={'1.10':'Orthogonal Bundle',
'1.20':'Up-down Bundle',
'1.25':'Alpha Horseshoe',
'1.40':'Alpha solenoid',
'1.50':'Alpha/alpha barrel',
'2.10':'Ribbon',
'2.20':'Single Sheet',
'2.30':'Roll',
'2.40':'Beta Barrel',
'2.50':'Clam',
'2.60':'Sandwich',
'2.70':'Distorted Sandwich',
'2.80':'Trefoil',
'2.90':'Orthogonal Prism',
'2.100':'Aligned Prism',
'2.102':'3-layer Sandwich',
'2.105':'3 Propeller',
'2.110':'4 Propeller',
'2.115':'5 Propeller',
'2.120':'6 Propeller',
'2.130':'7 Propeller',
'2.140':'8 Propeller',
'2.150':'2 Solenoid',
'2.160':'3 Solenoid',
'2.170':'Beta Complex',
'2.180':'Shell',
'3.10':'Roll',
'3.15':'Super Roll',
'3.20':'Alpha-Beta Barrel',
'3.30':'2-Layer Sandwich',
'3.40':'3-Layer(aba) Sandwich',
'3.50':'3-Layer(bba) Sandwich',
'3.55':'3-Layer(bab) Sandwich',
'3.60':'4-Layer Sandwich',
'3.65':'Alpha-beta prism',
'3.70':'Box',
'3.75':'5-stranded Propeller',
'3.80':'Alpha-Beta Horseshoe',
'3.90':'Alpha-Beta Complex',
'3.100':'Ribosomal Protein L15; Chain: K; domain 2',
'4.10':'Irregular',
'6.10':'Helix non-globular',
'6.20':'Other non-globular'}

def read_data(CATH_file: str, working_dir: str) -> pd.DataFrame:
    try:
        df=pd.read_csv(working_dir+CATH_file+'.csv', index_col=0)
        #start stop needs to be str
        df['start']=df['start'].apply(str)
        df['stop']=df['stop'].apply(str)
        return df
        
    except IOError:
        cath_info=[]
        temp=[]
        start_stop=[]
        for line in open(working_dir+CATH_file+'.txt'):
            if line[:6]=='DOMAIN': 
                #PDB
                temp.append(line[10:14])
                #chain
                temp.append(line[14])
            if line[:6]=='CATHCO': 
                 #class, architecture, topology, homologous superfamily
                    cath=[int(i) for i in line[10:].strip('\n').split('.')]
                    temp=temp+cath
            if line[:6]=='SRANGE':
                j=line.split()
                #start and stop resi, can be multiple for the same chain
                #!!!!! 
                # dealing with insertions, e.g. 1A, 1B, 1C. The letter is removed for compatibility with ampal
                #!!!
                start_stop.append([str(j[1][6:]),str(j[2][5:])])
            if line[:2]=='//':
                #keep fragments from the same chain as separate entries
                for fragment in start_stop:
                    cath_info.append(temp+fragment)
                start_stop=[]
                temp=[]
        df=pd.DataFrame(cath_info,columns=['PDB','chain','class','architecture','topology','hsf','start', 'stop'])
        df.to_csv(working_dir+CATH_file+'.csv')
        return df

#gets fold sequence directly from PDB
#sequences in original CATH txt file are missing uncommon residues, e.g. phosphoserine.
def get_sequence(series: pd.Series) -> str:
    try:
        with gzip.open('/home/shared/datasets/pdb/'+series.PDB[1:3]+'/pdb'+series.PDB+'.ent.gz','rb') as protein:
                assembly=ampal.load_pdb(protein.read().decode(), path=False)
                #convert pdb res id into sequence index, 
                #some files have discontinuous res ids so ampal.get_slice_from_res_id() does not work
                start=0
                stop=0
                #if nmr structure, get 1st model-fix this in the future
                if isinstance(assembly,ampal.AmpalContainer):
                    chain=assembly[0][series.chain]
                else:
                    chain=assembly[series.chain]
                for i,residue in enumerate(chain):
                    #deal with insertions
                    if series.start[-1].isalpha():
                        if (residue.id+residue.insertion_code)==series.start:
                            start=i      
                    else: 
                        if residue.id==series.start:
                            start=i   
                    if series.stop[-1].isalpha():
                        if (residue.id+residue.insertion_code)==series.stop:
                            stop=i          
                    else:
                        if residue.id==series.stop:
                            stop=i            
        return chain[start:(stop+1)].sequence
    #some pdbs are obsolete, return NaN
    except FileNotFoundError:
        return np.NaN

#return pdbs based on class, architecture, topology, homologous superfamily, at least class has to be specified
def get_pdbs(df: pd.DataFrame, c: int, a: int=0 , t: int=0, hsf: int=0) -> pd.DataFrame:
    if hsf!=0:
        return df.loc[(df['class']==c) & (df['topology']==t) & (df['architecture']==a) & (df['hsf']==hsf)].copy()
    elif t!=0:
        return df.loc[(df['class']==c) & (df['topology']==t) & (df['architecture']==a)].copy()
    elif a!=0:
        return df.loc[(df['class']==c) & (df['architecture']==a)].copy()
    else:
        return df.loc[(df['class']==c)].copy()
    
#get sequences for all entries in the dataframe, appends DataFrame inplace.
def append_sequence(df):
    df.loc[:,'sequence']=[get_sequence(x) for i,x in df.iterrows()]
    return df.sequence.isna().sum()
