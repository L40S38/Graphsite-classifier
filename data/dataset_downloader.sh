#!/bin/bash

wget https://osf.io/m9yg6/download -O ~/osf_data/clusters.yaml
wget https://osf.io/jcrua/download -O ~/osf_data/unseen_data.tar.gz
tar xvzf ~/osf_data/unseen_data.tar.gz -C ~/osf_data
wget https://osf.io/94ywb/download -O ~/osf_data/negative-data.tar.gz
tar xvzf ~/osf_data/negative-data.tar.gz -C ~/osf_data
wget https://osf.io/qebrd/download -O ~/osf_data/negative_data_output_probs.yaml
wget https://osf.io/fcyxt/download -O ~/osf_data/negative_pocket_list.txt
