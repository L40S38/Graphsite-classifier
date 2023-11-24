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

# hyper parameter tuning
import optuna
from optuna.trial import TrialState
from joblib import parallel_backend

# multi gpu processing
from multiprocessing import Manager

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

    return parser.parse_args()


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


def gen_classification_report(model,device,dataloader):
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

GPU_ID=4
def main():
    gpu_id = GPU_ID
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
        'trained_classifier_model_{}.pt'.format('gpu'+str(gpu_id))
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

    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    # detect cpu or gpu
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available()
                          else 'cpu')
    print('device: ', device)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)

    # divide the clusters into train, validation and test
    train_clusters, test_clusters = divide_clusters_train_test(clusters)
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

    def objective(trial: optuna.trial.Trial):
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
            num_classes=num_classes, 
            num_features=num_features,
            dim=model_size, 
            train_eps=True, 
            num_edge_attr=1,
            which_model=which_model, 
            num_layers=num_layers,
            num_channels=num_channels, 
            deg=deg
        ).to(device)
        print('model architecture:')
        print(model)

        # compute class weights as a tensor of size num_classes
        use_class_weights = config['use_class_weights']
        if use_class_weights == True:
            class_weights = compute_class_weights(train_clusters)
            class_weights = torch.FloatTensor(class_weights).to(device)
        else:
            class_weights = torch.ones(num_classes).to(device)
        gamma = -1
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
        #learning_rate = config['learning_rate']
        learning_rate = trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
        #weight_decay = config['weight_decay']
        weight_decay = trial.suggest_float("weight_decay",1e-8,1e-2,log=True)
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
            #cooldown=40,
            cooldown=trial.suggest_int("cooldown", 5, 100),
            #min_lr=0.0001, 
            min_lr=trial.suggest_float("min_lr", 1e-5, 1.0, log=True),
            verbose=True
        )
        print('learning rate scheduler: ')
        print(scheduler)

        best_test_acc = 0
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        print('begin training...')

        def train(gamma):

            model.train()

            # increasing gamma of FocalLoss
            if which_loss == 'Focal' and focal_gamma_ascent == True:
                if epoch in focal_gamma_ascent_epoch:
                    #global gamma
                    gamma += 1
                    print('epoch {}, gamma increased to {}.'.format(epoch, gamma))
                    loss_function.set_gamma(gamma)
            print(gamma)
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

            return train_loss, train_acc, gamma


        def test():
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
        
        for epoch in range(1, 1 + num_epoch):
            train_loss, train_acc, gamma = train(gamma)
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
            trial.report(test_acc,epoch) #最大化したい値をreportする。第2引数はepoch, stepなど

        #print('best test acc {} at epoch {}.'.format(
        #    best_test_acc, best_test_epoch))

        # save the history of loss and accuracy
        #results = {'train_losses': train_losses, 'train_accs': train_accs,
        #       'val_losses': test_losses, 'val_accs': test_accs}
        #with open(loss_dir, 'w') as fp:
        #    json.dump(results, fp)

        print('trial finished.')

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="maximize",load_if_exists=True, pruner=pruner,storage="sqlite:///example.db") #ある値を最大化したい
    study.optimize(objective, n_trials=100, timeout=24*3600, gc_after_trial=True) #objectiveでtrialの設定、100回のトライアル、1日過ぎたらタイムアウト

    # trialの取得
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED]) 
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    study.trial_dataframe().to_csv(loss_dir+"optuna_trial_result.csv",encoding="utf-8")

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":

    processes = []
    for rank in range(4):
        p = torch.multiprocessing.Process(target=main)
        GPU_ID = rank
        p.start()
        processes.append(p)
    for p in processes:
        p.join()