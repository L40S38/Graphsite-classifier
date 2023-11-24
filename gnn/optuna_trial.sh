#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
export CUDA_VISIBLE_DEVICE=4,5,6,7
mpirun -n 4 python optuna_train_classifier.py
