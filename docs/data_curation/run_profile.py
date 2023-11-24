#for file in *.fa; do sh SCRATCH-1D_1.3/pkg/PROFILpro_1.2/bin/generate_profiles.sh $file $file.profile num_rounds num_threads; done

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

def run_profile(data_dir,pdbf):
    FASTAFile = pdbf
    subprocess.run("~/SCRATCH-1D_1.3/pkg/PROFILpro_1.2/bin/generate_profiles.sh "+FASTAFile+" "+FASTAFile.replace(".fa",".profile")+" 4 4",shell=True,executable='/bin/sh')
    return


if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    fasta_files = [l for l in glob.glob(data_dir+"/PDB_CHAINS/**/*.fa")]
    #os.system("mkdir PROFILES")
    #os.makedirs(data_dir+"/PROFILES",exist_ok=True)
    for fasta in fasta_files:
        run_profile(data_dir,fasta)