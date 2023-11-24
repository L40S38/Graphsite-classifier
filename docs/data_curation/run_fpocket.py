from Bio.PDB import *
from Bio.PDB.PDBIO import Select
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
import subprocess
import shutil
#ls = [l for l in os.listdir() if "_atm.pdb" in l]

def get_args():
    parser = argparse.ArgumentParser("python")

    parser.add_argument("-data_dir",
                        required=False,
                        default='/home/shiozawa.l/add_val_data',
                        help="the path where .pdb files are saved at (data_dir/*.pdb)")

    return parser.parse_args()

class PocketSelect(Select):
    #Trueとなったものだけ描画される
    def __init__(self,residue_id_list):
        super(PocketSelect).__init__()
        self.residue_id_list = residue_id_list

    def accept_residue(self, residue):
        het,resseq,icode = residue.get_id()

        if resseq in self.residue_id_list:
            return True
        else:
            return False

def fpocket(data_dir,dir,pdbf):
    print(dir,pdbf)
    if os.path.exists(dir.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED")):
        shutil.rmtree(dir.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED"))
    os.makedirs(dir.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED"),exist_ok=True)
    subp = subprocess.Popen("cp "+pdbf+" "+dir.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED"),shell=True,executable='/bin/bash')
    subp.wait()
    subp = subprocess.run("fpocket -f "+pdbf.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED"),shell=True,executable='/bin/bash')
    protein = PDBParser().get_structure(pdbf.replace(".pdb",""),pdbf)
    for pocket_pdb in glob.glob(pdbf.replace("PDB_CHAINS","POCKETS_UNTRANSFORMED").replace(".pdb","_out/pockets/*.pdb")):
        pocket = PDBParser().get_structure(pocket_pdb.replace(".pdb",""),pocket_pdb)
        residue_ids = {}
        for residue in pocket.get_residues():
            het, resseq, icode = residue.get_id()
            residue_ids[resseq] = residue.get_resname()
        io = PDBIO()
        io.set_structure(protein)
        io.save(pocket_pdb,select=PocketSelect(residue_ids.keys()))
        

    return


if __name__ == "__main__":
	#ls = [l for l in os.listdir("POCKETS_UNTRANSFORMED") if ".pdb" in l]
    args = get_args()
    data_dir = args.data_dir
    os.makedirs(data_dir+"/POCKETS_UNTRANSFORMED/",exist_ok=True)
    dir = [dir for dir in glob.glob(data_dir+"/PDB_CHAINS/*")]
    for dir in dir:
        pdbfs = glob.glob(dir+"/*.pdb")
        for pdbf in pdbfs:
            if len(pdbf.split("/")[-1].replace(".pdb",""))==6: #format such as 0aa0_A.pdb
                fpocket(data_dir,dir,pdbf)
