from cgi import print_environ_usage
from tracemalloc import start
from markupsafe import re
import matplotlib
import time

from main_fed_seperateloss import loss_generate

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
import utils.setup_tools as SetupTools
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, GCNLinkPred, GATLinkPred, SAGELinkPred
from models.Fed import FedAvg
from models.test import test_img
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from training.fed_subgraph_lp_trainer import FedSubgraphLPTrainer
import torch.multiprocessing as mp
import ctypes


def generate_loss(loss, losses):
    """
    loss generate functions
    """
    # loss_list = [l.item() for l in losses.values()]
    # loss += (sum(loss_list) / len(loss_list))
    # loss = loss / 2
    return loss


def waiting_all_embeddings(embeddings, dead_time):
    start_time = time.time()
    while True:
        items = []
        for item in embeddings.values():
            items.append(item.item())
        if 0 not in items:
            # print("get all embeddings, continuing decoding...")
            break
        if time.time() - start_time >= 5: break


def merge_embeddings(embedding, embeddings_ids):
    """
    take care of memory storage...
    """
    embeddings = {}
    for idx in embeddings_ids.keys():
        embeddings[idx] = ctypes.cast(id(embeddings_ids[idx]),
                                      ctypes.py_object).value
    #TODO: for each embeddings: shape[x,64]. x is changed in different embeddings. How to merge?

    # math calculation

    return embedding


def local_train(args, idx, model, train_data, losses, embeddings, dead_time=5):
    """
    dead_time: sec, if wait too long means there is no other process 
                training, stop waiting and update by losses that haved.
    """
    local_model = model
    optimizer = SetupTools.get_optimizer(args.client_optimizer, args.lr,
                                         args.wd, local_model)
    local_model.to(args.device)
    local_model.train()
    for idx_batch, batch in enumerate(train_data):
        batch.to(args.device)
        optimizer.zero_grad()
        #forward
        z = local_model.encode(batch.x, batch.edge_train)

        # copy to the share memory and wait for all loss
        start_time = time.time()
        embeddings[idx] += id(z)
        waiting_all_embeddings(embeddings, dead_time=dead_time)

        z = merge_embeddings(z, embeddings)

        link_logits = local_model.decode(z, batch.edge_train)
        link_labels = batch.label_train
        loss = torch.nn.functional.mse_loss(link_logits, link_labels)

        # backward
        loss = generate_loss(loss, losses)
        loss.backward()
        optimizer.step()
        print(f"{idx} loss:{loss}")
    losses[idx] = loss
    # model.load_state_dict(local_model.state_dict())
    model = local_model
    print(f"[training] {idx} loss is {losses[idx]}")


def train(args, train_data_local_dict, val_data_local_dict, net_glob):
    torch.multiprocessing.set_start_method('spawn')
    round_loss = []
    for round in range(args.epochs):
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        models = {}
        losses = {}
        embeddings = {}
        # shared loss and weights
        for idx in idxs_users:
            models[idx] = copy.deepcopy(net_glob)
            losses[idx] = torch.zeros(1).to(args.device)
            embeddings[idx] = torch.zeros(1).to(args.device)
            losses[idx].share_memory_()
            models[idx].share_memory()
            embeddings[idx].share_memory_()
        for epoch in range(args.local_ep):
            # start multiprocess in each client
            epoch_loss = []
            processes = []
            # define multi process
            for idx in idxs_users:
                train_data = train_data_local_dict[idx]
                #multiprocessing all local run
                p = mp.Process(target=local_train,
                               args=(args, idx, models[idx], train_data, losses,
                                     embeddings))
                p.start()
                processes.append(p)
            # run trains
            for p in processes:
                p.join()

            # update net_glob
            for idx in idxs_users:
                w = models[idx].state_dict()
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
            # w_glob = weight_merge(w_locals)
            w_glob = FedAvg(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            epoch_loss.append(sum(losses.values()) / len(losses.values()))
        round_loss.append((sum(epoch_loss) / len(epoch_loss)))
        #val for each round
        for idx in idxs_users:
            if val_data_local_dict is not None:
                val_data = val_data_local_dict[idx]
                if args.metric == "MAE":
                    metric_fn = mean_absolute_error
                test_score, _, mae, rmse, mse = test(net_glob,
                                                     val_data,
                                                     args.device,
                                                     val=True,
                                                     metric=metric_fn)
                print(
                    "[val] Round = {}: idx = {} , Test score = {} , mae = {}, rmse = {} , mse={}"
                    .format(round, idx, test_score, mae, rmse, mse))
        # val for each round

    return net_glob, torch.tensor(round_loss).numpy()


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
                link_labels = batch.label_val
            else:
                link_logits = model.decode(train_z, batch.edge_test)
                link_labels = batch.label_test
            score = metric(link_labels.cpu(), link_logits.cpu())
            mae.append(mean_absolute_error(link_labels.cpu(),
                                           link_logits.cpu()))
            rmse.append(
                mean_squared_error(link_labels.cpu(),
                                   link_logits.cpu(),
                                   squared=False))
            mse.append(mean_squared_error(link_labels.cpu(), link_logits.cpu()))
    return score, model, mae, rmse, mse


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # # load dataset and split users
    dataset = SetupTools.load_data(args=args)
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

    print('log info')

    net_glob = SetupTools.create_model(args, args.model,
                                       feature_dim).to(args.device)
    print(net_glob)

    #train
    model, loss_train = train(args, train_data_local_dict, val_data_local_dict,
                              net_glob)

    #test
    test_score, _, mae, rmse, mse = test(net_glob,
                                         test_data_global,
                                         args.device,
                                         val=True,
                                         metric=mean_absolute_error)
    print("[testing]: mae = {}, rmse = {} , mse={}".format(
        np.mean(mae), np.mean(rmse), np.mean(mse)))

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fedgnn.png')
