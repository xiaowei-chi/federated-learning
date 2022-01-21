from graph_data_preprocessing.data_loader import *
from models.Nets import MLP, CNNMnist, CNNCifar, GCNLinkPred, GATLinkPred, SAGELinkPred


def load_data( args):
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


def create_model( args, model_name, feature_dim):
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


def get_optimizer(client_optimizer, lr, wd, model):
    if client_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wd,
        )

    return optimizer