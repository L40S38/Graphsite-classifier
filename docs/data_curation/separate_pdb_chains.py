import os, sys
import pandas as pd
from Bio.PDB import *
from Bio.PDB import PDBParser, PDBIO
import subprocess
import argparse
import glob

#execution with os.hogehoge translated to subprocess.run

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()

def read_pdb(data_dir,pdbf):
    print(data_dir,pdbf)
    io = PDBIO()
    pdb = PDBParser().get_structure(pdbf.replace(".pdb", ""), data_dir+"/"+pdbf)
    for chain in pdb.get_chains():
        chain = chain.get_id()
        # Need to have pdbtools installed in the system to run pdb_selchain 
        os.makedirs(data_dir+"/PDB_CHAINS/"+pdbf.replace(".pdb", "_"+chain)+"/",exist_ok=True)
        subprocess.run("source ~/anaconda3/etc/profile.d/conda.sh; conda activate data_curation; pdb_selchain -"+chain+" "+data_dir+"/"+pdbf+" > "+data_dir+"/PDB_CHAINS/"+pdbf.replace(".pdb", "_"+chain)+"/"+pdbf.replace(".pdb", "_"+chain+".pdb"),shell=True,executable='/bin/bash',timeout=10)
    return



if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    list_of_pdbs = [l for l in glob.glob(data_dir+"/*.pdb")]
    print(list_of_pdbs)
    #os.system("mkdir PDB_CHAINS")
    os.makedirs(data_dir+"/PDB_CHAINS",exist_ok=True)
    for pdbf in list_of_pdbs:
        pdbf = pdbf.split("/")[-1]
        read_pdb(data_dir,pdbf)
