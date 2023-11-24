import sys
from Bio import SeqIO
import os, sys
import glob
import subprocess
import argparse
import rdkit
from rdkit import Chem
import yaml

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    list_of_chain_dir = [l for l in glob.glob(data_dir+"/PDB_CHAINS/*")]
    prot_to_mol = {}
    for dir in list_of_chain_dir:
        pdbfs = [f for f in glob.glob(dir+"/*.pdb") if len(f.split("/")[-1].replace(".pdb",""))>6] # remove protein file
        for pdbf in pdbfs:
            try:
                mol = Chem.MolFromPDBFile(pdbf)
                smiles = Chem.MolToSmiles(mol)
            except:
                continue
            mol = Chem.MolFromSmiles(smiles)
            num_atom_count = len(mol.GetAtoms())
            if num_atom_count<2:
                continue
            print(pdbf,smiles)
            prot_to_mol[pdbf.split("/")[-1][:6]] = smiles
            break

    list_of_pocket_dir = [l for l in glob.glob(data_dir+"/POCKETS_TRANSFORMED_MOL2/*")]
    pocket_to_mol = {}
    for dir in list_of_pocket_dir:
        pocket_to_mol[dir.split("/")[-1]] = prot_to_mol[dir.split("/")[-1][:6]]
        print(dir.split("/")[-1],prot_to_mol[dir.split("/")[-1][:6]])
    with open(data_dir+"/pocket-smiles.yaml","w") as f:
        yaml.dump(pocket_to_mol,f)