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

#ls = [l for l in os.listdir() if "_atm.pdb" in l]

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()


def transform(data_dir,dir):
    os.makedirs(data_dir+"/POCKETS_TRANSFORMED/"+dir,exist_ok=True)
    pocket_pdbs = glob.glob(data_dir+"/POCKETS_UNTRANSFORMED/"+dir+"/"+dir+"_out/pockets/*.pdb")
    for l in pocket_pdbs:
        ppdb = PandasPdb()
        ppdb.read_pdb(l)
        # This is my covariance matrix obtained from 3 x N points
        cov_mat = np.cov(np.array(ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']].T))
        eigen_values, eigen_vectors = LA.eig(cov_mat)
        # matmul(R, R.T) = I => that eigen_vectors is rotation matrix		
        # get the box coordinates to get the center of the pocket
        box = subprocess.check_output("obabel -ipdb "+l+" -obox | grep 'CENTER' | tr -s ' '", shell=True)
        box = ''.join(map(chr,box)).rstrip().split(" ")[5:]
        box = pd.DataFrame([float(x) for x in box])
        box = box.T
        box.columns = ['x_coord', 'y_coord', 'z_coord']
        df = pd.DataFrame(index=range(ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']].shape[0]),columns=range(ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']].shape[1]))
        df.columns = ['x_coord', 'y_coord', 'z_coord']
        df['x_coord'] = df['x_coord'].fillna(box.x_coord.values[0])
        df['y_coord'] = df['y_coord'].fillna(box.y_coord.values[0])
        df['z_coord'] = df['z_coord'].fillna(box.z_coord.values[0])
        ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']] = ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']].subtract(df)
        ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']] = np.matmul(ppdb.df["ATOM"][['x_coord', 'y_coord', 'z_coord']].to_numpy(), eigen_vectors)
        ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
        ppdb.to_pdb(path=data_dir+"/POCKETS_TRANSFORMED/"+dir+"/"+l.split("/")[-1].replace(".pdb","_rot.pdb"),records=None, gz=False, append_newline=True)
    return


if __name__ == "__main__":
	#ls = [l for l in os.listdir("POCKETS_UNTRANSFORMED") if ".pdb" in l]
    args = get_args()
    data_dir = args.data_dir
    os.makedirs(data_dir+"/POCKETS_TRANSFORMED/",exist_ok=True)
    dirs = [dir for dir in glob.glob(data_dir+"/POCKETS_UNTRANSFORMED/*")]
    for dir in dirs:
        dir = dir.split("/")[-1]
        transform(data_dir,dir)
