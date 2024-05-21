import os
import argparse
import torch
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from models import GnnNets
from load_dataset import get_dataloader, get_TUDataset, upsample
from Configures import data_args, train_args, model_args
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from load_dataset import shuffle, Dataset
from aug import get_class_num
from sklearn.metrics import f1_score
import random
from scipy.optimize import fsolve, differential_evolution
import torch_geometric.transforms as T

import warnings
warnings.filterwarnings("ignore")

from torch_geometric.utils import degree

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

def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


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


def train_GC(args):
    clst = args.clst
    sep = args.sep
    data_args.dataset_name = args.dataset_name
    model_args.model_name = args.model_name
    model_args.mlp_out_dim = args.mlp_out_dim
    train_args.learning_rate = args.lr
    train_args.batch_size = args.batch_size
    train_args.weight_decay = args.weight_decay
    train_args.max_epochs = args.max_epochs
    train_args.retain_ratio = args.retain_ratio
    train_args.pruning_epochs = args.pruning_epochs

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
    output_dim = int(dataset.num_classes)
    print("output_dim:", output_dim)

    if data_args.imb is True:
        c_train_num, c_val_num = get_class_num(data_args.imb_ratio, data_args.num_train, data_args.num_val)
        y = torch.tensor([data.y.item() for data in dataset])
        train_data, val_data, test_data = shuffle(dataset, c_train_num, c_val_num, y)
        if train_args.imb_solve == 'upsampling':
            train_data = upsample(train_data)
            val_data = upsample(val_data)
        Dataset_TMP = Dataset
        train_dataset = Dataset_TMP(train_data, dataset, data_args)
        val_dataset = Dataset_TMP(val_data, dataset, data_args)
        test_dataset = Dataset_TMP(test_data, dataset, data_args)
    
    if data_args.imb is True:
        dataloader = {
            'train': DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True, collate_fn=train_dataset.collate_batch),
            'eval': DataLoader(val_dataset, batch_size=train_args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch),
            'test': DataLoader(test_dataset, batch_size=train_args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch),
        }
    else:
        dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio)

    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

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

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0

    initial_dataset_size = len(dataloader['train'].dataset)
    retain_count = int(initial_dataset_size * train_args.retain_ratio)
    original_train_dataset = dataloader['train'].dataset
    
    all_rank_scores = torch.zeros(len(original_train_dataset), device=model_args.device)
    global_indices = torch.arange(len(original_train_dataset), device=model_args.device)
    global_indices_cpu = global_indices.cpu()

    myck = model_args.model_name+"+visprot("+str(model_args.num_prototypes_per_class)+")_"+str(train_args.retain_ratio)


    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        gnnNets.train()
        epoch_coordinates = []
        epoch_indices = []

        joint(gnnNets)          

        for batch_index, batch in enumerate(dataloader['train']):
            pred, virtual_label, prot_nce_loss, graph_emb, distances = gnnNets(batch)
            if model_args.enable_prot is True:
                epoch_coordinates.append(distances)
                indices_offset = batch_index * dataloader['train'].batch_size
                new_indices = torch.arange(distances.size(0), device=model_args.device) + indices_offset
                epoch_indices.append(new_indices)

            loss = criterion(pred, batch.y)

            if model_args.enable_prot is True:
                # Cluster Loss
                prototypes_of_correct_class = torch.t(gnnNets.model.get_prototype_class_identity()[:, batch.y].bool()).to(model_args.device)
                cluster_cost = torch.mean(torch.min(distances[prototypes_of_correct_class].reshape(-1, model_args.num_prototypes_per_class), dim=1)[0])

                # Seperation Loss
                eps = 1e-6
                separation_cost = torch.mean(1.0 / (torch.min(distances[~prototypes_of_correct_class].reshape(-1, (output_dim-1)*model_args.num_prototypes_per_class), dim=1)[0] + eps))
            else:
                cluster_cost = 0.00
                separation_cost = 0.00

            loss = loss + clst*cluster_cost + sep*separation_cost 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            _, prediction = torch.max(pred, -1)
            loss_list.append(loss.item())
            if data_args.imb is True:
                batch_f1_macro = f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), average='macro')
                acc.append(batch_f1_macro)
            else:
                acc.append(prediction.eq(batch.y).cpu().numpy())

        if data_args.imb is False:
            report_train_acc = np.concatenate(acc, axis=0).mean()
        else:
            report_train_acc = np.mean(acc)
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), report_train_acc))
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.6f} | Ld: {np.average(ld_loss_list):.6f} | "
              f"Acc: {report_train_acc:.6f}")

        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.6f} | Acc: {eval_state['acc']:.6f}")
        append_record("Eval epoch {:2d}, loss: {:.6f}, acc: {:.6f}".format(epoch, eval_state['loss'], eval_state['acc']))

        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
            

        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, myck, eval_state['acc'], is_best)
    
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

            sorted_indices = torch.argsort(current_score, descending=False)
            top_indices = sorted_indices[:retain_count]

            top_indices_cpu = top_indices.cpu()

            retained_indices = global_indices_cpu[top_indices_cpu].tolist()

            pruned_dataset = Subset(original_train_dataset, retained_indices)
            dataloader['train'] = DataLoader(pruned_dataset, batch_size=train_args.batch_size, shuffle=True)
        
        elif epoch >= train_args.pruning_epochs:
            retained_indices = random.sample(range(initial_dataset_size), retain_count)
            pruned_dataset = Subset(original_train_dataset, retained_indices)
            dataloader['train'] = DataLoader(pruned_dataset, batch_size=train_args.batch_size, shuffle=True)

    print(f"The best validation accuracy is {best_acc}.")

    checkpoint = torch.load(os.path.join(ckpt_dir, f'{myck}_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test: | Loss: {test_state['loss']:.6f} | Acc: {test_state['acc']:.6f}")
    append_record("Test, loss: {:.6f}, acc: {:.6f}".format(test_state['loss'], test_state['acc']))

    
def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            pred, virtual_label, prot_nce_loss, graph_emb, distances = gnnNets(batch)
            loss = criterion(pred, batch.y)

            _, prediction = torch.max(pred, -1)
            loss_list.append(loss.item())
            if data_args.imb is True:
                batch_f1_macro = f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), average='macro')
                acc.append(batch_f1_macro)
            else:
                acc.append(prediction.eq(batch.y).cpu().numpy())

        if data_args.imb is False:
            report_eval_acc = np.concatenate(acc, axis=0).mean()
        else:
            report_eval_acc = np.mean(acc)
        eval_state = {'loss': np.average(loss_list),
                      'acc': report_eval_acc}

    return eval_state

def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            pred, virtual_label, prot_nce_loss, graph_emb, distances = gnnNets(batch)
            loss = criterion(pred, batch.y)

            _, prediction = torch.max(pred, -1)
            loss_list.append(loss.item())
            if data_args.imb is True:
                batch_f1_macro = f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), average='macro')
                acc.append(batch_f1_macro)
            else:
                acc.append(prediction.eq(batch.y).cpu().numpy())
        
        if data_args.imb is False:
            report_test_acc = np.average(np.concatenate(acc, axis=0).mean())
        else:
            report_test_acc = np.mean(acc)

    test_state = {'loss': np.average(loss_list),
                  'acc': report_test_acc}

    return test_state, None, None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ProtGNN')
    parser.add_argument('--clst', type=float, default=0.0,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.0,
                        help='separation')
    parser.add_argument('--model_name', type=str, default='pna',
                        help='model name')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='max epochs')
    parser.add_argument('--retain_ratio', type=float, default=0.2,
                        help='retain ratio')
    parser.add_argument('--pruning_epochs', type=int, default=40,
                        help='pruning epochs')
    parser.add_argument('--dataset_name', type=str, default='DHFR',
                        help='dataset name')
    parser.add_argument('--mlp_out_dim', type=int, default=0,
                        help='MLP output dim')
    args = parser.parse_args()
    train_GC(args)
