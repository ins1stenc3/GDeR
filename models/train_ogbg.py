import os
import argparse
import torch
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from models import GnnNets
from load_dataset import get_TUDataset, upsample
from Configures import data_args, train_args, model_args
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import time
import random
from scipy.optimize import differential_evolution
import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")

from torch_geometric.utils import degree

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def get_degree_array_for_pnaconv(dataset):
    deg_dict = {}
    for data in dataset:
        d = degree(data.edge_index[0], dtype=torch.long)
        for deg in d.tolist():
            if deg in deg_dict:
                deg_dict[deg] += 1
            else:
                deg_dict[deg] = 1

    degrees = sorted(deg_dict.items(), key=lambda x: x[1], reverse=True)
    n = 10  
    top_degrees = degrees[:n]
    degree_array = torch.tensor([deg for deg, count in top_degrees], dtype=torch.float32)

    return degree_array

def balance_line(k, points):
    x, y = points[:, 0], points[:, 1]
    k=k[0]
    above_line = y - k * x > 0
    balance = above_line.sum() - (~above_line).sum()
    return balance.item()  


def joint(model):
    for p in model.model.get_gnn_layers().parameters():
        p.requires_grad = True
    model.model.get_prototype_vectors().requires_grad = True


def append_record(info):
    file_name = './log/'+(model_args.model_name)+'+prot(' +str(model_args.num_prototypes_per_class)+")_"+str(train_args.retain_ratio)+'_cnt='+str(train_args.cnt)
    f = open(file_name, 'a')
    f.write(info)
    f.write('\n')
    f.close()


def compute_degree_distribution(dataset):
    deg_dict = {}
    for data in dataset:
        d = data.degree(data.edge_index[0], dtype=torch.long)
        for degree in d.numpy():
            if degree in deg_dict:
                deg_dict[degree] += 1
            else:
                deg_dict[degree] = 1
    return torch.tensor(sorted(deg_dict.keys()), dtype=torch.float)

def InfoNCEloss(features, temperature=0.07):
    if features.dim() == 1:
        features = features.unsqueeze(1)  
    features = features.float()
    features = F.normalize(features, dim=1)  
    similarity_matrix = torch.matmul(features, features.T)
    similarity_matrix /= temperature
    labels = torch.arange(0, features.size(0), dtype=torch.long).cuda()
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def UniformityLoss(graph_counts, device):
    total_graphs = graph_counts.sum()
    probabilities = graph_counts / total_graphs
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6))  
    max_entropy = torch.log(torch.tensor(len(graph_counts), device=device, dtype=torch.float32))
    normalized_entropy_loss = (max_entropy - entropy) / max_entropy
    return normalized_entropy_loss


def train(model, device, loader, optimizer, task_type, args, output_dim):
    clst = args.clst
    sep = args.sep
    info_lambda = args.info
    protnce_lambda = args.protnce_lambda


    epoch_coordinates = []
    epoch_indices = []
    virtual_label_graph_count = {}

    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, virtual_label, prot_nce_loss, graph_emb, distances = model(batch)

            if virtual_label is not None:
                for label in torch.argmax(virtual_label, dim=1).tolist():
                    virtual_label_graph_count[label] = virtual_label_graph_count.get(label, 0) + 1

            if model_args.enable_prot is True:
                epoch_coordinates.append(distances)
                indices_offset = step * loader.batch_size
                new_indices = torch.arange(distances.size(0), device=model_args.device) + indices_offset
                epoch_indices.append(new_indices)

            optimizer.zero_grad()

            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            
            if model_args.enable_prot is True:     
                if model_args.single_target is True:    
                    prototypes_of_correct_class = torch.t(model.model.get_prototype_class_identity()[:, batch.y.squeeze().long()].bool()).to(model_args.device)
                else:
                    correct_class_indices = torch.argmax(virtual_label, dim=1)
                    prototypes_of_correct_class = torch.t(model.model.get_prototype_class_identity()[:, correct_class_indices.squeeze().long()].bool()).to(model_args.device)

                # Cluster Loss
                cluster_cost = torch.mean(torch.min(distances[prototypes_of_correct_class].view(-1, model_args.num_prototypes_per_class), dim=1)[0])

                # Seperation Loss
                eps = 1e-6
                separation_cost = torch.mean(1.0 / (torch.min(distances[~prototypes_of_correct_class].view(-1, (output_dim-1)*model_args.num_prototypes_per_class), dim=1)[0] + eps))

                # Nce Loss
                if model_args.single_target is False:  
                    prot_nce_cost =   protnce_lambda * prot_nce_loss 
                    info_nce_cost = info_lambda * InfoNCEloss(virtual_label)
                    nce_cost = prot_nce_cost + info_nce_cost
                else:
                    nce_cost = 0.0

            else:
                cluster_cost = 0.00
                separation_cost = 0.00

            if model_args.single_target is False:  
                avg_graphs_per_label = len(virtual_label_graph_count) / output_dim
                graph_counts = torch.tensor(list(virtual_label_graph_count.values()), dtype=torch.float32, device=device)
                uniformity_loss = UniformityLoss(graph_counts, device)
            else:
                uniformity_loss = 0.0

            loss = loss + clst*cluster_cost + sep*separation_cost  + uniformity_loss + nce_cost

            loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()
    
    return epoch_coordinates, epoch_indices, virtual_label_graph_count


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _, _, graph_emb, min_distances = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def pipeline(args):
    clst = args.clst
    sep = args.sep
    model_args.model_name = args.model_name
    model_args.mlp_out_dim = args.mlp_out_dim
    data_args.dataset_name = args.dataset_name
    train_args.learning_rate = args.lr
    train_args.batch_size = args.batch_size
    train_args.weight_decay = args.weight_decay
    train_args.max_epochs = args.max_epochs
    train_args.retain_ratio = args.retain_ratio
    train_args.pruning_epochs = args.pruning_epochs
    train_args.cnt = args.cnt

    print('start loading data====================')
    if model_args.model_name in ['gps']:
        train_args.pre_transform = T.AddRandomWalkPE(walk_length=model_args.pe_dim, attr_name='pe')

    dataset = get_TUDataset(data_args.dataset_name, pre_transform=train_args.pre_transform)

    if dataset[0].edge_attr is None:
        model_args.edge_dim = 0
    else:
        model_args.edge_dim = dataset[0].edge_attr.size(1)
    print('dataset[0]:', dataset[0])
    model_args.deg = get_degree_array_for_pnaconv(dataset)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    print("output_dim:", output_dim)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=train_args.batch_size, shuffle=True, num_workers = train_args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=train_args.batch_size, shuffle=False, num_workers = train_args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=train_args.batch_size, shuffle=False, num_workers = train_args.num_workers)
    dataloader = {
            'train': train_loader,
            'eval': valid_loader,
            'test': test_loader,
    }

    print('start training model==================')
    model = GnnNets(input_dim, output_dim, model_args)
    model.to_device()
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    optimizer = Adam(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")
    best_acc = 0.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    initial_dataset_size = len(dataloader['train'].dataset)
    retain_count = int(initial_dataset_size * train_args.retain_ratio)
    original_train_dataset = dataloader['train'].dataset
    all_rank_scores = torch.zeros(len(original_train_dataset), device=model_args.device)
    global_indices = torch.arange(len(original_train_dataset), device=model_args.device)
    global_indices_cpu = global_indices.cpu()

    evaluator = Evaluator(data_args.dataset_name)

    valid_curve = []
    test_curve = []
    train_curve = []

    append_record("tmux start!!!")
    for epoch in range(0, train_args.max_epochs):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        if model_args.enable_prot is False and epoch >= train_args.pruning_epochs:
            retained_indices = random.sample(range(initial_dataset_size), retain_count)
            pruned_dataset = Subset(original_train_dataset, retained_indices)
            dataloader['train'] = DataLoader(pruned_dataset, batch_size=train_args.batch_size, shuffle=True)

        joint(model)

        epoch_coordinates, epoch_indices, virtual_label_graph_count = train(model, model_args.device, dataloader['train'], optimizer, dataset.task_type, args, output_dim)


        print('Evaluating...')
        train_perf = eval(model, model_args.device, dataloader['train'], evaluator)
        valid_perf = eval(model, model_args.device, dataloader['eval'], evaluator)
        test_perf = eval(model, model_args.device, dataloader['test'], evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        append_record("Epoch {}, Train: {}, Validation: {}, Test: {}".format(epoch, train_perf, valid_perf, test_perf))

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if model_args.enable_prot is True and epoch >= train_args.pruning_epochs:
            epoch_coordinates = torch.cat(epoch_coordinates, 0)
            epoch_indices = torch.cat(epoch_indices, 0)
            epoch_coordinates_np = epoch_coordinates.detach().cpu().numpy()

            bounds = [(-10, 10)]  
            result = differential_evolution(balance_line, bounds, args=(epoch_coordinates_np,))
            if not result.success:
                print("Global optimization did not converge:", result.message)
                return
            else:
                k = torch.tensor(result.x[0], dtype=torch.float32, device=epoch_coordinates.device)

            x = epoch_coordinates[:, 0]
            y = epoch_coordinates[:, 1]
            current_score = torch.abs(y - k * x) / torch.sqrt(k**2 + 1)

            all_rank_scores[epoch_indices] = current_score

            sorted_indices = torch.argsort(all_rank_scores, descending=False)
            top_indices = sorted_indices[:retain_count]

            top_indices_cpu = top_indices.cpu()

            retained_indices = global_indices_cpu[top_indices_cpu].tolist()

            pruned_dataset = Subset(original_train_dataset, retained_indices)
            dataloader['train'] = DataLoader(pruned_dataset, batch_size=train_args.batch_size, shuffle=True)
        

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    append_record("Finished training!")
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    append_record("Best validation score: {}".format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    append_record("Test score: {}".format(test_curve[best_val_epoch]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ProtGNN')
    parser.add_argument('--clst', type=float, default=0.0,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.0,
                        help='separation')
    parser.add_argument('--model_name', type=str, default='gps',
                        help='model name')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='max epochs')
    parser.add_argument('--retain_ratio', type=float, default=0.2,
                        help='retain ratio')
    parser.add_argument('--pruning_epochs', type=int, default=10,
                        help='pruning epochs')
    parser.add_argument('--mlp_out_dim', type=int, default= 0,
                        help='mlp output dim')
    parser.add_argument('--dataset_name', type=str, default='ogbg-molhiv',
                        help='dataset name')
    parser.add_argument('--cnt', type=int, default=1,
                        help='record times')
    parser.add_argument('--info', type=float, default=0.001,
                        help='info nce loss lambda')
    parser.add_argument('--protnce_lambda', type=float, default=0.001,
                        help='proto nce loss lambda')
    args = parser.parse_args()
    pipeline(args)
