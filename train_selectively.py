"""
Select hard pairs online during training.
"""
import argparse
import os
import random
import yaml
import torch
from dataloader import read_cluster_file_from_yaml, select_classes, divide_clusters, pocket_loader_gen, cluster_by_chem_react
from dataloader import merge_clusters
from model import SelectiveSiameseNet, SelectiveContrastiveLoss
import json


def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-cluster_file_dir',
                        default='../data/clusters_after_remove_files_with_no_popsa.yaml',
                        required=False,
                        help='text file to get the cluster labels')

    parser.add_argument('-pocket_dir',
                        default='../data/googlenet-dataset/',
                        required=False,
                        help='directory of pockets')

    parser.add_argument('-pop_dir',
                        default='../data/pops-googlenet/',
                        required=False,
                        help='directory of popsa files for sasa feature')

    parser.add_argument('-subcluster_file',
                        default='./pocket_cluster_analysis/results/subclusters_0.yaml',
                        required=False,
                        help='subclusters by chemical reaction of some clusters')

    parser.add_argument('-trained_model_dir',
                        default='../trained_models/pair_selecting_model_1.pt',
                        required=False,
                        help='directory to store the trained model.')

    parser.add_argument('-loss_dir',
                        default='./results/pair_selecting_model_1.json/',
                        required=False,
                        help='directory to store the training losses.')

    return parser.parse_args()


def train():
    """
    Train the model for 1 epoch, then return the mean loss of the data 
    in this epoch.
    Global vars: train_loader, train_size, device, optimizer, model
    batch_interval: number of mini-batch intervals to log loss
    """
    model.train()

    # learning rate decay
    if epoch == lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    total_loss = 0
    num_loss_elements = 0 # total number of pairs used for training in this epoch
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        embedding = model(data)
        loss_mean, loss_sum, loss_shape = loss_function(embedding, data.y)
        loss_mean.backward()
        optimizer.step()

        # the loss is averaged over samples in a mini-batch
        # last incomplete batch is dropped, so just use batch_size
        total_loss += loss_sum.item()
        num_loss_elements += loss_shape[0]
    train_loss = total_loss / num_loss_elements

    return train_loss


def validate():
    """
    Validate the model for 1 epoch, then return the mean loss of the data
    in this epoch. The methodology used is same as training: the hardest
    pairs are selected for validation.
    """
    model.eval()

    total_loss = 0
    num_loss_elements = 0 # total number of pairs used for training in this epoch
    for data in val_loader:
        data = data.to(device)    
        embedding = model(data)
        loss_mean, loss_sum, loss_shape = loss_function(embedding, data.y)

        # last incomplete batch is dropped, so just use batch_size
        total_loss += loss_sum.item()
        num_loss_elements += loss_shape[0]
    val_loss = total_loss / num_loss_elements

    return val_loss


if __name__=="__main__":
    random.seed(666)  # deterministic sampled pockets and pairs from dataset
    print('seed: ', 666)
    args = get_args()
    cluster_file_dir = args.cluster_file_dir
    pocket_dir = args.pocket_dir
    pop_dir = args.pop_dir
    subcluster_file = args.subcluster_file
    with open(subcluster_file) as file:
        subcluster_dict = yaml.full_load(file)

    trained_model_dir = args.trained_model_dir
    loss_dir = args.loss_dir

    num_classes = 10
    print('number of classes:', num_classes)
    cluster_th = 10000  # threshold of number of pockets in a class

    merge_info = [[0, 9], [1, 5], 2, [3, 8], 4, 6, 7]
    print('how to merge clusters: ', merge_info)

    subclustering = False  # whether to further subcluster data according to subcluster_dict
    print('whether to further subcluster data according to chemical reaction: {}'.format(
        subclustering))

    # tunable hyper-parameters
    num_epochs = 3000
    print('number of epochs to train:', num_epochs)
    lr_decay_epoch = 2000
    print('learning rate decay to half at epoch {}.'.format(lr_decay_epoch))

    learning_rate = 0.003
    weight_decay = 0.0005

    batch_size = 96
    print('batch size:', batch_size)
    num_hard_pos_pairs = 128
    num_hard_neg_pairs = 128
    print('number of hardest positive pairs for each mini-batch: ', num_hard_pos_pairs)
    print('number of hardest negative pairs for each mini-batch: ', num_hard_neg_pairs)
    num_workers = os.cpu_count()
    num_workers = int(min(batch_size, num_workers))
    print('number of workers to load data: ', num_workers)

    # margins for the relaxed contrastive loss
    similar_margin = 0.0
    dissimilar_margin = 2.0
    print('similar margin of contrastive loss: {}'.format(similar_margin))
    print('dissimilar margin of contrastive loss: {}'.format(dissimilar_margin))

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')  # detect cpu or gpu
    print('device: ', device)

    # read the original clustered pockets
    clusters = read_cluster_file_from_yaml(cluster_file_dir)

    # select clusters according to rank of sizes and sample large clusters
    clusters = select_classes(clusters, num_classes, cluster_th)

    # merge clusters as indicated in 'merge_info'. e.g., [[0,3], [1,2], 4]
    clusters = merge_clusters(clusters, merge_info)
    num_classes = len(clusters)
    print('number of classes after merging: ', num_classes)

    # replace some clusters with their subclusters
    if subclustering == True:
        clusters, cluster_ids = cluster_by_chem_react(
            clusters, subcluster_dict)
        num_classes = len(clusters)
        print('number of classes after further clustering: ', num_classes)

    # divide the clusters into train, validation and test
    train_clusters, val_clusters, test_clusters = divide_clusters(clusters)
    num_train_pockets = sum([len(x) for x in train_clusters])
    num_val_pockets = sum([len(x) for x in val_clusters])
    num_test_pockets = sum([len(x) for x in test_clusters])
    print('number of pockets in training set: ', num_train_pockets)
    print('number of pockets in validation set: ', num_val_pockets)
    print('number of pockets in test set: ', num_test_pockets)

    # missing popsa files for sasa feature at this moment
    features_to_use = ['x', 'y', 'z',  'r', 'theta', 'phi', 'sasa', 'charge', 'hydrophobicity',
                       'binding_probability', 'sequence_entropy']
    num_features = len(features_to_use)

    train_loader, train_size = pocket_loader_gen(pocket_dir=pocket_dir,
                                                 pop_dir=pop_dir,
                                                 clusters=train_clusters,
                                                 features_to_use=features_to_use,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)

    val_loader, val_size = pocket_loader_gen(pocket_dir=pocket_dir,
                                             pop_dir=pop_dir,
                                             clusters=val_clusters,
                                             features_to_use=features_to_use,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)


    model = SelectiveSiameseNet(num_features=num_features,
        dim=32, train_eps=True, num_edge_attr=1).to(device)
    print('model architecture:')
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)
    print('optimizer:')
    print(optimizer)

    # differentiable, no parameters to train.
    loss_function = SelectiveContrastiveLoss(
        similar_margin=similar_margin, dissimilar_margin=dissimilar_margin, num_pos_pair=num_hard_pos_pairs, num_neg_pair=num_hard_neg_pairs).to(device)
    print('loss function:')
    print(loss_function)


    train_losses = []
    val_losses = []
    best_val_loss = 9999999
    for epoch in range(1, num_epochs+1):
        train_loss = train()
        train_losses.append(train_loss)

        val_loss = validate()
        val_losses.append(val_loss)
        
        print('epoch: {}, train loss: {}, validation loss: {}.'.format(epoch, train_loss, val_loss))

        if  val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), trained_model_dir)

    print('best validation loss {} at epoch {}.'.format(best_val_loss, best_val_epoch))
    
    # write loss history to disk
    results = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(loss_dir, 'w') as fp:
        json.dump(results, fp)

