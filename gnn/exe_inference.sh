#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pocket2drug
python inference.py -unseen_data_dir ~/osf_data/unseen_data/ -unseen_data_classes ../data/clusters.yaml -trained_model ../trained_models/trained_classifier_model_0.pt -config ./train_classifier.yaml
