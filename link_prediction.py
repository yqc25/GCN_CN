import argparse
import torch
import numpy as np
import time
import torch.nn.functional as F
import xlwt
import networkx as nx
import matplotlib.pyplot as plt
import random

import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, add_self_loops, degree, to_undirected, sort_edge_index, \
    from_networkx, to_networkx, to_dense_adj, to_edge_index, dense_to_sparse
from torch_geometric.datasets import AttributedGraphDataset, CitationFull, WebKB, Coauthor, Amazon, Actor
from torch_geometric.loader import DataLoader
from utils import Logger, evaluate_auc
from model import HGCNEncoder, EdgeDecoder, HGAE
from citations import directed_data
from torch_geometric.data import Data


def train(model, data, split_edges, optimizer, lr_scheduler, args, epoch):
    model.train()
    remaining_edges, masked_edges = split_edges['train'].edge_index, split_edges['train'].edge_index
    aug_edge_index, _ = add_self_loops(data.edge_index)
    neg_edges = negative_sampling(
        aug_edge_index,
        num_nodes=data.x.size(0),
        num_neg_samples=masked_edges.view(2, -1).size(1),
    ).view_as(masked_edges)

    for perm in DataLoader(range(masked_edges.size(1)), args.batch_size, shuffle=True):
        optimizer.zero_grad()
        # if args.is_pre is False:
        #     z = model.encoder(data.x, data.adj)
        # else:
        #     z = model.encoder(data.x, remaining_edges)
        if args.new_encoder:
            z = model.encoder(data.x, remaining_edges, data.cns)
        else:
            z = model.encoder(data.x, remaining_edges)
        pos_scores = model.edge_decoder(z, masked_edges[:, perm])
        neg_scores = model.edge_decoder(z, neg_edges[:, perm])

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_edges, batch_size, args):
    model.eval()

    z = model.encoder(data.x, split_edges['train'].edge_index, data.cns)


    aug_edge_index, _ = add_self_loops(data.edge_index)

    pos_valid_edge = split_edges['valid'].pos_edge_label_index
    neg_valid_edge = split_edges['valid'].neg_edge_label_index

    pos_test_edge = split_edges['test'].pos_edge_label_index
    neg_test_edge = split_edges['test'].neg_edge_label_index

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
        edge = pos_valid_edge[:, perm]
        pos_valid_preds += [model.edge_decoder(z, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
        edge = neg_valid_edge[:, perm]
        neg_valid_preds += [model.edge_decoder(z, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
        edge = pos_test_edge[:, perm]
        pos_test_preds += [model.edge_decoder(z, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
        edge = neg_test_edge[:, perm]
        neg_test_preds += [model.edge_decoder(z, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    val_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    val_true = torch.cat([torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)], dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)

    results = evaluate_auc(val_pred, val_true, test_pred, test_true)
    return results


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Facebook')
    parser.add_argument('--encoder_channels', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--encoder_dropout', type=float, default=0.)
    parser.add_argument('--decoder_dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2 ** 18)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=5)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--path_dataset', type=str, default='./dataset/')
    parser.add_argument('--path_model', type=str, default='./models/linkpred.pth')
    parser.add_argument('--per_epoch', type=int, default=5)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)
    if args.directed:
        transform = T.Compose([T.ToDevice(device)])
    else:
        transform = T.Compose([T.ToUndirected(), T.ToDevice(device)])

    if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'BlogCatalog', 'Flickr', 'Facebook', 'PPI', 'Wiki']:
        dataset = AttributedGraphDataset(args.path_dataset, args.dataset)
        data = dataset[0]
        if args.dataset == 'Flickr':
            data.x = data.x.to_dense()
    elif args.dataset == 'Cora_ML':
        data = directed_data(args.path_dataset, args.dataset)
    elif args.dataset == 'DBLP':
        dataset = CitationFull(args.path_dataset, args.dataset)
        data = dataset[0]
    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(args.path_dataset, args.dataset)
        data = dataset[0]
    elif args.dataset in ['Computers', 'Photo']:
        dataset = Amazon(args.path_dataset, args.dataset)
        data = dataset[0]
    elif args.dataset in ['CS', 'Physics']:
        dataset = Coauthor(args.path_dataset, args.dataset)
        data = dataset[0]
    else:
        raise ValueError(args.dataset)
##############################
    # data.x = torch.randn(data.x.size(0), data.x.size(1))
    # seed = int(time.time())
    seed = 10007
    # G = nx.barabasi_albert_graph(2000, 4, seed=seed)
    # G = nx.gnm_random_graph(2000, 8000, seed=seed)
    G = nx.powerlaw_cluster_graph(2000, 4, 0.9, seed=seed)
    # data.edge_index = to_undirected(data.edge_index)
    # G = to_networkx(data)
    G = nx.Graph(G)

    data = from_networkx(G)
    num_nodes = nx.number_of_nodes(G)
    if int(num_nodes * 0.3) > 300:
        num_features = 150
    else:
        num_features = int(num_nodes * 0.3)
    data.x = torch.randn(num_nodes, num_features)
    nodes_pos = torch.randint(0, num_nodes, (num_features,))
    x = torch.zeros(num_nodes, num_features)
    m = 0
    while m < num_nodes:
        n = 0
        while n < num_features:
            if nx.has_path(G, m, nodes_pos[n].item()):
                x[m][n] = nx.shortest_path_length(G, m, nodes_pos[n].item()) / 5
            else:
                x[m][n] = -2
            n += 1
        m += 1
    data.x = x
    # data.x = torch.eye(num_nodes)
###########################
    data = transform(data)

    metric = 'AP'
    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args)
    }

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                        is_undirected=~args.directed,
                                                        split_labels=True,
                                                        add_negative_train_samples=True)(data)

    splits = dict(train=train_data, valid=val_data, test=test_data)

    splits['train'].edge_index, _ = add_self_loops(splits['train'].edge_index, num_nodes=data.x.size(0))
    row, col = splits['train'].edge_index
    cns = torch.zeros_like(row)
    for i in range(row.size(0)):
        nei1 = list(nx.neighbors(G, row[i].item()))
        nei2 = list(nx.neighbors(G, col[i].item()))
        cns[i] = len(set(nei1) & set(nei2)) / len(nei2)
    data.cns = cns.to(device)


    encoder = HGCNEncoder(data.x.size(1), args.encoder_channels)
    edge_decoder = EdgeDecoder(args.encoder_channels, args.hidden_channels, 1, args.decoder_dropout)
    model = HGAE(encoder, edge_decoder).to(device)
    print(model)
    print(device)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(tot_params))

    for run in range(args.runs):

        model.reset_parameters()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            t1 = time.time()
            loss = train(model, data, splits, optimizer, lr_scheduler, args, epoch)
            t2 = time.time()
            if epoch % args.eval_steps == 0:
                results = test(model, data, splits, args.batch_size, args)
                valid_result = results[metric][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), args.path_model)
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_result, test_result = result
                        print(key)
                        print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                              f'Epoch: {epoch:04d} / {args.epochs:02d}, '
                              f'Best_epoch: {best_epoch:02d}, '
                              f'Best_result: {best_valid:.2%}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {valid_result:.2%}, '
                              f'Test: {test_result:.2%}',
                              f'Training Time/epoch: {t2 - t1:.3f}')
                    print('=' * round(140 * epoch / (args.epochs + 1)))
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(args.path_model))
        results = test(model, data, splits, args.batch_size, args)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:04d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == '__main__':
    main()
