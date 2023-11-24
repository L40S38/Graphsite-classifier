# Copyright: Wentao Shi, 2020
import argparse
import yaml
import random
import os
import torch
import torch.nn as nn
from dataloader import read_cluster_file_from_yaml
from dataloader import divide_clusters_train_test
from dataloader import pocket_loader_gen
from dataloader import merge_clusters
from model import GraphsiteClassifier, FocalLoss
from torch_geometric.utils import degree
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import sklearn.metrics as metrics
import json
import copy
import matplotlib
import matplotlib.pyplot as plt

# ignore Future Warning
import warnings
warnings.simplefilter('ignore', FutureWarning)


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-run',
                        required=False,
                        default=0,
                        type=int,
                        help='which experiment.')

    parser.add_argument('-seed',
                        required=False,
                        default=23,
                        type=int,
                        help='random seed for splitting dataset.')
    
    parser.add_argument('-config',
                        required=False,
                        default='./train_classifier.yaml',
                        help='config file such as train_classifier.yaml')
    
    parser.add_argument('-gpu',
                        required=False,
                        default=0,
                        help='which gpu to use')

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    seed = int(args.seed)
    random.seed(seed)
    print('seed: ', seed)
    run = int(args.run)
    config_file = args.config

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cluster_file_dir = config['cluster_file_dir']
    pocket_dir = config['pocket_dir']
    pop_dir = config['pop_dir']
    trained_model_dir = config['trained_model_dir'] + \
        'trained_classifier_model_{}.pt'.format(run)
    loss_dir = config['loss_dir'] + \
        'train_classifier_results_{}.json'.format(run)
    confusion_matrix_dir = config['confusion_matrix_dir']
    print('save trained model at: ', trained_model_dir)
    print('save loss at: ', loss_dir)

    merge_info = config['merge_info']
    features_to_use = config['features_to_use']
    num_features = len(features_to_use)
    print('how to merge clusters: ', merge_info)
    print('features to use: ', features_to_use)

    num_epoch = config['num_epoch']
    print('number of epochs: ', num_epoch)

    batch_size = config['batch_size']
    print('batch size: ', batch_size)

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']

    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)

    # divide the clusters into train, validation and test
    train_clusters, test_clusters = divide_clusters_train_test(clusters)

    train_dict = {}
    for i,cluster in enumerate(train_clusters):
        train_dict[i] = cluster

    test_dict = {}
    for i,cluster in enumerate(test_clusters):
        test_dict[i] = cluster

    os.makedirs(cluster_file_dir.replace("clusters.yaml","folds/"),exist_ok=True)

    with open(cluster_file_dir.replace("clusters.yaml",f"folds/train_fold_{args.seed}.yaml"),"w") as f:
        yaml.dump(train_dict,f)

    with open(cluster_file_dir.replace("clusters.yaml",f"folds/test_fold_{args.seed}.yaml"),"w") as f:
        yaml.dump(test_dict,f)