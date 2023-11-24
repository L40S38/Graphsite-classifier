import os, sys
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

import argparse
import glob

# to work with some non orthodox pdbs
import warnings
warnings.filterwarnings('ignore')

class RemoveHetatm(Select):
    #Trueとなったものだけ描画される
    def __init__(self,residue_id_list):
        super(RemoveHetatm).__init__()
        self.residue_id_list = residue_id_list

    def accept_residue(self, residue):
        het,resseq,icode = residue.get_id()

        if resseq in self.residue_id_list:
            return True
        else:
            return False
        
class OnlyHetatm(Select):
    #Trueとなったものだけ描画される
    def __init__(self,residue_id):
        super(OnlyHetatm).__init__()
        self.residue_id = residue_id

    def accept_residue(self, residue):
        het,resseq,icode = residue.get_id()

        if resseq == self.residue_id:
            return True
        else:
            return False

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()

def renumber(pdbf):
    io = PDBIO()
    parser = PDBParser()
    my_pdb_structure = parser.get_structure(pdbf.replace(".pdb", ""), pdbf)
    residue_N = 1
    non_hetatm_residue = set()
    for chain in my_pdb_structure.get_chains():
        for residue in chain.get_residues():
            hetatm = True
            for atom in residue.get_atoms():
                if atom.get_name()=="CA":
                    hetatm = False
                    break
            if hetatm:
                _, resseq, _ = residue.get_id()
                res_name = residue.get_resname()
                io.set_structure(my_pdb_structure)
                io.save(pdbf.replace(".pdb",f"_{res_name}_{resseq}.pdb"), select=OnlyHetatm(resseq))
            else:
                residue.id = (residue.id[0], residue_N, " ")
                print(residue.get_resname())
                non_hetatm_residue.add(residue_N)
                residue_N += 1
    io.set_structure(my_pdb_structure)
    io.save(pdbf,  preserve_atom_numbering=True, select=RemoveHetatm(non_hetatm_residue))
    return

if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    list_of_pdbs = [l for l in glob.glob(data_dir+"/PDB_CHAINS/**/*.pdb")]
    for pdbf in list_of_pdbs:
        try:
            renumber(pdbf)
        except Exception as e:
            # Remove unchanged PDBs
            print(f"{e},remove pdb {pdbf}")
            if os.path.exists(pdbf):
                os.remove(pdbf)
            pass
