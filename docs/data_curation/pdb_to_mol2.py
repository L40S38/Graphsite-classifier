from Bio.PDB import *
from numpy import array
from numpy import linalg as LA
import os, sys
import os, sys
from Bio import PDB
import numpy as np
import math
from math import acos, sin, cos
import subprocess
from biopandas.pdb import PandasPdb
import pandas as pd
import argparse
import glob
import re

#ls = [l for l in os.listdir() if "_atm.pdb" in l]

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()


def transform(data_dir,l):
    num = int(re.findall("pocket(.*)_atm_rot.pdb",l.split("/")[-1])[0])
    os.makedirs(data_dir+"/POCKETS_TRANSFORMED_MOL2/"+l.split("/")[-2]+f"{num:02d}"+"/",exist_ok=True)
    subp = subprocess.Popen("obabel -ipdb "+l+" -omol2 -gen3d -O"+data_dir+"/POCKETS_TRANSFORMED_MOL2/"+l.split("/")[-2]+f"{num:02d}"+"/"+l.split("/")[-2]+f"{num:02d}.mol2",shell=True,executable='/bin/bash')
    subp.wait()
    return


if __name__ == "__main__":
	#ls = [l for l in os.listdir("POCKETS_UNTRANSFORMED") if ".pdb" in l]
    args = get_args()
    data_dir = args.data_dir
    ls = [l for l in glob.glob(data_dir+"/POCKETS_TRANSFORMED/**/*.pdb")]
    os.makedirs(data_dir+"/POCKETS_TRANSFORMED_MOL2",exist_ok=True)
    for l in ls:
        transform(data_dir,l)