#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, GCNLinkPred, GATLinkPred, SAGELinkPred
from models.Fed import FedAvg
from models.test import test_img
from graph_data_preprocessing.data_loader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from training.fed_subgraph_lp_trainer import FedSubgraphLPTrainer

def load_data(args, dataset_name):
    if args.dataset not in ["ciao", "epinions"]:
        raise Exception("no such dataset!")

    args.pred_task = "link_prediction"

    args.metric = "MAE"

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ) = load_partition_data(args, args.data_dir, args.num_users)

    dataset = [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ]

    return dataset

def test(model, test_data, device, val=True, metric=mean_absolute_error):
    model.eval()
    model.to(device)
    metric = metric
    mae, rmse, mse = [], [], []

    for batch in test_data:
        batch.to(device)
        with torch.no_grad():
            train_z = model.encode(batch.x, batch.edge_train)
            if val:
                link_logits = model.decode(train_z, batch.edge_val)
            else:
                link_logits = model.decode(train_z, batch.edge_test)

            if val:
                link_labels = batch.label_val
            else:
                link_labels = batch.label_test
            score = metric(link_labels.cpu(), link_logits.cpu())
            mae.append(mean_absolute_error(link_labels.cpu(), link_logits.cpu()))
            rmse.append(mean_squared_error(link_labels.cpu(), link_logits.cpu(), squared = False))
            mse.append(mean_squared_error(link_labels.cpu(), link_logits.cpu()))
    return score, model, mae, rmse, mse

def create_model(args, model_name, feature_dim):
    # print("create_model. model_name = %s" % (model_name))
    if model_name == "gcn":
        model = GCNLinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    elif model_name == "gat":
        model = GATLinkPred(
            feature_dim, args.hidden_size, args.node_embedding_dim, args.num_heads
        )
    elif model_name == "sage":
        model = SAGELinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    else:
        raise Exception("such model does not exist !")
    return model

def local_update(net_glob, train_data):

    net_glob.to(args.device)
    net_glob.train()
    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net_glob.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net_glob.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )

    best_model_params = {}
    for epoch in range(args.local_ep):
        for idx_batch, batch in enumerate(train_data):
            # print(batch)
            batch.to(args.device)
            optimizer.zero_grad()

            z = net_glob.encode(batch.x, batch.edge_train)
            link_logits = net_glob.decode(z, batch.edge_train)
            link_labels = batch.label_train
            loss = F.mse_loss(link_logits, link_labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(copy.deepcopy(loss.item()))
        epoch_loss.append(sum(batch_loss)/len(batch_loss))


    return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # # load dataset and split users
    dataset = load_data(args, args.dataset)
    [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ] = dataset

    net_glob = create_model(args, args.model, feature_dim).to(args.device)
    print(net_glob)
    # copy weights
    w_glob = net_glob.state_dict()

    loss_train = []
    # training
    for round in range(args.epochs):
        loss_locals = []
        epoch_loss = []
        batch_loss = []

        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            train_data = train_data_local_dict[idx]
            w, loss = local_update(net_glob,train_data)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))  

            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
        
            #val
            if val_data_local_dict is not None:
                val_data = val_data_local_dict[idx]
                if args.metric == "MAE":
                    metric_fn = mean_absolute_error 
                test_score, _ , mae, rmse, mse= test(
                    net_glob, train_data, args.device, val=True, metric=metric_fn
                )
                print(
                    "Round = {}: idx = {} , Test score = {} , mae = {}, rmse = {} , mse={}".format(
                        round,  idx, test_score, mae, rmse, mse
                    )
                )


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        #test
        test_score, _ , mae, rmse, mse= test(net_glob, test_data_global, args.device, val=True, metric=mean_absolute_error )
        print("testing: mae = {}, rmse = {} , mse={}".format(mae, rmse, mse))

        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig('./save/fedgnn.png')

