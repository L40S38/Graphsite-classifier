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


def train():
    """
    Train the model for 1 epoch, then return the averaged loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    """
    model.train()

    # increasing gamma of FocalLoss
    if which_loss == 'Focal' and focal_gamma_ascent == True:
        if epoch in focal_gamma_ascent_epoch:
            global gamma
            gamma += 1
            print('epoch {}, gamma increased to {}.'.format(epoch, gamma))
            loss_function.set_gamma(gamma)

    loss_total = 0

    # all the predictions for the epoch
    epoch_pred = []

    # all the labels for the epoch
    epoch_label = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(output, data.y)
        loss.backward()
        loss_total += loss.item() * data.num_graphs
        optimizer.step()
        pred = output.max(dim=1)[1]

        # convert prediction and label to list
        # used to compute evaluation metrics
        pred_cpu = list(pred.cpu().detach().numpy())
        label = list(data.y.cpu().detach().numpy())

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    # accuracy of entire epoch
    train_acc = metrics.accuracy_score(
        epoch_label, epoch_pred)

    # averaged training loss
    train_loss = loss_total / train_size

    return train_loss, train_acc


def test():
    """
    Returns loss and accuracy on validation set.
    Global vars: val_loader, val_size, device, model
    """
    model.eval()

    loss_total = 0

    # all the predictions for the epoch
    epoch_pred = []

    # all the labels for the epoch
    epoch_label = []

    for data in test_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(output, data.y)
        loss_total += loss.item() * data.num_graphs
        pred = output.max(dim=1)[1]

        # used to compute evaluation metrics
        pred_cpu = list(pred.cpu().detach().numpy())

        # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy())

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    # accuracy of entire epoch
    test_acc = metrics.accuracy_score(
        epoch_label, epoch_pred)

    # averaged training loss
    test_loss = loss_total / test_size

    return test_loss, test_acc


def compute_class_weights(clusters):
    """
    Compute the weights of each class/cluster 
    according to number of data.   

    Arguments:
    clusters - list of lists of pockets.
    """
    cluster_lengths = [len(x) for x in clusters]
    cluster_weights = np.array([1/x for x in cluster_lengths])
    # normalize the weights with mean
    cluster_weights = cluster_weights/np.mean(cluster_weights)

    return cluster_weights


def gen_classification_report(dataloader):
    """
    Generate a detailed classification report.
    """
    model.eval()

    # all the predictions for the epoch
    epoch_pred = []

    # all the labels for the epoch
    epoch_label = []

    for data in dataloader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = output.max(dim=1)[1]

        # used to compute evaluation metrics
        pred_cpu = list(pred.cpu().detach().numpy())

        # used to compute evaluation metrics
        label = list(data.y.cpu().detach().numpy())

        epoch_pred.extend(pred_cpu)
        epoch_label.extend(label)

    report = metrics.classification_report(epoch_label, epoch_pred, digits=4)
    confusion_mat = metrics.confusion_matrix(
        y_true=epoch_label, y_pred=epoch_pred, normalize='true')
    return report, confusion_mat


def plot_cm(cm, figure_path):
    """
    Plot the input confusion matrix. 
    """
    import seaborn as sns
    #sns.set_theme()
    font = {'size': 8}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    #colors= "coolwarm"
    #colors = "summer"
    #colors = "viridis"
    #colors = "inferno"
    #colors = "magma"
    #colors = "cividis"
    #cmap = "YlGnBu"
    cmap = sns.light_palette("green", as_cmap=True)
    ax.set_title('Normalized confusion matrix')
    ax = sns.heatmap(cm, 
                     annot=True, fmt='.2',
                     #linewidths=.5, 
                     cmap=cmap)
    ax.set(xlabel='Predicted label', ylabel='True label')
    plt.savefig(figure_path, bbox_inches='tight')

def divide_clusters_train_test_from_yaml(clusters,data_dir,val_fold=0):
    # test folds
    # load the validation dict
    with open(data_dir + 'pockets_fold{}.yaml'.format(val_fold), 'r') as f:
        val_dict = yaml.full_load(f)

    # put the data in train folds together
    train_clusters = [list(filter(lambda x: x not in val_dict.keys(), cluster)) for cluster in clusters]
    test_clusters = [list(filter(lambda x: x in val_dict.keys(), cluster)) for cluster in clusters]
    print(f"num each in train_clusters:{[len(cluster) for cluster in train_clusters]}")
    print(f"num each in test_clusters:{[len(cluster) for cluster in test_clusters]}")

    return train_clusters, test_clusters

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

    # detect cpu or gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available()
                          else 'cpu')
    print('device: ', device)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)

    # divide the clusters into train, validation and test
    #train_clusters, test_clusters = divide_clusters_train_test(clusters)
    train_clusters, test_clusters = divide_clusters_train_test_from_yaml(clusters, data_dir='../../Pocket2Drug/data/folds/')
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in test set: ', num_test_pockets)
    print('first 5 pockets in train set of cluster 0 before merging (to verify reproducibility):')
    print(train_clusters[0][0:5])
    print('first 5 pockets in test set of cluster 0 before merging (to verify reproducibility):')
    print(test_clusters[0][0:5])

    train_loader, train_size, train_set = pocket_loader_gen(
        pocket_dir=pocket_dir,
        pop_dir=pop_dir,
        clusters=train_clusters,
        features_to_use=features_to_use,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader, test_size, _ = pocket_loader_gen(
        pocket_dir=pocket_dir,
        pop_dir=pop_dir,
        clusters=test_clusters,
        features_to_use=features_to_use,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    which_model = config['which_model']
    model_size = config['model_size']
    num_layers = config['num_layers']
    which_loss = config['which_loss']
    assert which_model in ['jk', 'residual', 'normal', 'pna', 'jknwm', 'jkgin']
    assert which_loss in ['CrossEntropy', 'Focal']

    # the degrees for the PNA model
    if which_model == 'pna':
        deg = torch.zeros(31, dtype=torch.long)
        for data in train_set:
            # print(cnt)
            d = degree(
                data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
    else:
        deg = None

    # the channel number for nmm model
    num_channels = config['num_channels']

    model = GraphsiteClassifier(
        num_classes=num_classes, #len(clusters)
        num_features=num_features, #11
        dim=model_size, #128
        train_eps=True, 
        num_edge_attr=1,
        which_model=which_model, #jknwm
        num_layers=num_layers, #6
        num_channels=num_channels, #3
        deg=deg
    ).to(device)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay, 
        amsgrad=False
    )
    print('optimizer:')
    print(optimizer)

    # decay learning rate when validation accuracy stops increasing.
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5,
        patience=10, 
        cooldown=40,
        min_lr=0.0001, 
        verbose=True
    )
    print('learning rate scheduler: ')
    print(scheduler)

    # compute class weights as a tensor of size num_classes
    use_class_weights = config['use_class_weights']
    if use_class_weights == True:
        class_weights = compute_class_weights(train_clusters)
        class_weights = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights = torch.ones(num_classes).to(device)

    if which_loss == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif which_loss == 'Focal':
        gamma = config['initial_focal_gamma']
        print('initial gamma of FocalLoss: ', gamma)
        loss_function = FocalLoss(
            gamma=gamma, alpha=class_weights, reduction='mean')
        focal_gamma_ascent = config['focal_gamma_ascent']
        if focal_gamma_ascent == True:
            focal_gamma_ascent_epoch = config['focal_gamma_ascent_epoch']
            print('increase gamma of FocalLoss at epochs: ',
                  focal_gamma_ascent_epoch)
    print('loss function:')
    print(loss_function)

    best_test_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    print('begin training...')
    for epoch in range(1, 1 + num_epoch):
        train_loss, train_acc = train()
        test_loss, test_acc = test()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print('epoch: {}, train loss: {}, acc: {}; test loss: {}, acc: {}'.format(
            epoch, train_loss, train_acc, test_loss, test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), trained_model_dir)

        scheduler.step(test_acc)

    print('best test acc {} at epoch {}.'.format(
        best_test_acc, best_test_epoch))

    # save the history of loss and accuracy
    results = {'train_losses': train_losses, 'train_accs': train_accs,
               'val_losses': test_losses, 'val_accs': test_accs}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

    print('****************************************************************')
    # load the best model to generate a detailed classification report
    model.load_state_dict(best_model)
    train_report, train_confusion_mat = gen_classification_report(train_loader)
    test_report, test_confusion_mat = gen_classification_report(test_loader)

    font = {'size': 8}
    matplotlib.rc('font', **font)

    print('train report:')
    print(train_report)
    #print('train confusion matrix:')
    #print(train_confusion_mat)

    confusion_matrix_path = confusion_matrix_dir + \
        'confusion_matrix_{}_train.png'.format(run)
    plot_cm(train_confusion_mat, confusion_matrix_path)
    print('---------------------------------------')

    print('test report: ')
    print(test_report)
    #print('test confusion matrix:')
    #print(test_confusion_mat)

    confusion_matrix_path = confusion_matrix_dir + \
        'confusion_matrix_{}_test.png'.format(run)
    plot_cm(test_confusion_mat, confusion_matrix_path)
    print('---------------------------------------')

    print('program finished.')