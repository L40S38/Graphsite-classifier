import sys
from Bio import SeqIO
import os, sys
import glob
import subprocess
import argparse

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()

def pdb2seq(data_dir,pdbf):
    PDBFile = pdbf
    with open(PDBFile, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            header = '>' + pdbf.split("/")[-1].replace(".pdb", "")
            seq = record.seq
            with open(pdbf.replace(".pdb", ".fa"), "w") as f:
                f.write(header+"\n"+str(seq))
    return


if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    list_of_pdb_with_cavity_center = [l for l in glob.glob(data_dir+"/PDB_CHAINS/**/*.pdb")]
    #os.system("mkdir PROFILES")
    os.makedirs(data_dir+"/PROFILES",exist_ok=True)
    for pdbf in list_of_pdb_with_cavity_center:
        pdb2seq(data_dir,pdbf)
        subprocess.run("sed -i 's/X//g' "+pdbf.replace(".pdb", ".fa"),shell=True,executable='/bin/bash')
