import ampal
import gzip
from pathlib import Path
import string
import urllib

def gly_resid(pdb: Path, chain:chr):
    """Rewrite PDB,change all amino acids to Glycine.
      Parameters
      ----------
      pdb: Path
          Location of pdb file
      chain: chr
          Chain identifier, only this chain will be changed to polyG."""
          
    with open(pdb,'r') as file:
        text=file.readlines()
    for i,line in enumerate(text):
        if line[21]==chain:
            text[i]='ATOM  '+text[i][6:17]+'GLY'+text[i][20:]
    with open(pdb,'w') as file:
        file.writelines(text)

def fetch_pdb(
    pdb_code: str,
    output_folder:Path,
    pdb_request_url: str = "https://files.rcsb.org/download/" ,
    is_pdb:bool=False,
) -> None:
    """
    Downloads a specific pdb file into a specific folder.
    Parameters
    ----------
    pdb_code : str
        Code of the PDB file to be downloaded.
    output_folder : Path
        Output path to save the PDB file.
    pdb_request_url : str
        Base URL to download the PDB files.
    is_pdb:bool=False
        If True, get .pdb, else get biological assembly.
    """
    if is_pdb:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb.gz"
    else:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb1.gz"
    print(f'{pdb_code_with_extension} is missing and will be downloaded!')
    urllib.request.urlretrieve(pdb_request_url + pdb_code_with_extension,filename=output_folder / pdb_code_with_extension)  
      
def polyglycine(dataset:Path,path_to_assemblies:Path,working_dir:Path,is_pdb:bool=False): 
    """Converts protein chains into polyglycine chains.
    Parameters
    -----------
    dataset:Path
        Path to the dataset list containing PDB+chain info (e.g. 1a2bA)
    path_to_assemblies:Path
        Path to the directory with protein structure files; missing files will be downloaded automatically.
    working_dir:Path
        Path to the directory where polyglycine structures will be saved.
    is_pdb:bool
        If True, expects and downloads PDBs. If False, expects/downloads biological assembly."""
                      
    with open(dataset,'r') as file:
        structures = [x.strip("\n") for x in file.readlines()]
    if is_pdb:
        suffix='.pdb.gz'
    else:
        suffix='.pdb1.gz'
    for protein in structures:
        if not Path(path_to_assemblies / (protein[:4]+suffix)).exists():
            fetch_pdb(protein,path_to_assemblies,is_pdb=is_pdb)
        
        with gzip.open(path_to_assemblies / (protein[:4]+suffix)) as file:
            assembly = ampal.load_pdb(file.read().decode(), path=False)
            protein_chain=protein[-1]
            if not is_pdb:
                flag=0
                # fuse all states of the assembly into one state.
                empty_polymer = ampal.Assembly()
                chain_id = []
                for polymer in assembly:
                    for chain in polymer:
                        #remove side chains from the chain of interest
                        #some assemblies have multiple chains with the same id, use flag to remove side chains only from the first one.
                        if chain.id==protein_chain and flag==0:
                            empty_polymer.append(chain.backbone)
                            flag=1
                        else:
                            empty_polymer.append(chain)
                        chain_id.append(chain.id)
                # relabel chains to avoid repetition, remove ligands.
            
                str_list = string.ascii_uppercase.replace(protein_chain, "")
                #assemblies such as viral capsids are longer than the alphabet
                if len(empty_polymer)>=len(str_list):
                     str_list=str_list*10
                index = chain_id.index(protein_chain)
                chain_id = list(str_list[: len(chain_id)])
                chain_id[index] = protein_chain
                empty_polymer.relabel_polymers(chain_id)
                
            else:
                empty_polymer = ampal.Assembly()
                #pick first state of NMR
                if isinstance(assembly, ampal.assembly.AmpalContainer):
                    assembly=assembly[0]
                for chain in assembly:
                    if chain.id==protein_chain:
                        empty_polymer.append(chain.backbone)
                    else:
                        empty_polymer.append(chain)
            # writing new pdb with AMPAL fixes most of the errors with EvoEF2 and Rosetta.
            pdb_text = empty_polymer.make_pdb(alt_states=False, ligands=False)
            with open((working_dir / protein[:4]).with_suffix(".pdb"), "w") as pdb_file:
                pdb_file.write(pdb_text)
            #change res ids to GLY for the backbone-only chain
            gly_resid((working_dir / protein[:4]).with_suffix(".pdb"),protein_chain)
            
                
if __name__=='__main__':
    #biological assemblies of crystal structures
    polyglycine(Path("/home/s1706179/Rosetta/data/set.txt"), Path("/home/s1706179/Rosetta/assemblies/"),Path("/home/s1706179/Rosetta/empty_backbones/"),False)
    #first state of NMR structures
    #polyglycine(Path("/home/s1706179/Rosetta/data/nmr_set.txt"), Path("/home/s1706179/Rosetta/nmr_structures/"),Path("/home/s1706179/Rosetta/empty_nmr_backbones/"),True)