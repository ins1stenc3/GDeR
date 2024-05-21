import os.path as osp
from torch_geometric.data import DataLoader
import torch
import numpy as np
from torch.utils.data import random_split, Subset
from torch_geometric.datasets import TUDataset, MoleculeNet 
from Configures import data_args
from torch.utils.data import Dataset as BaseDataset
from aug import *
from torch_geometric.data import DataLoader, Batch
from ogb.graphproppred import PygGraphPropPredDataset
import random


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=1):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        print()
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    dataloader = dict()

    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


def get_TUDataset(dataset, pre_transform):
    """
    'PROTEINS', 'REDDIT-BINARY', 'MUTAG', 'PTC_MR', 'DD', 'NCI1', 'DHFR', 'ogbg-molhiv', 'ogbg-molpcba', 'ZINC'
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset', 'TU', data_args.dataset_name)
    if data_args.dataset_name in ['MUTAG', 'PROTEINS', 'DHFR', 'DD', 'NCI1', 'PTC-MR', 'REDDIT-B', 'ENZYMES']:
        dataset = TUDataset(path, name=dataset, pre_transform=pre_transform)
    elif data_args.dataset_name in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(root=path, name=data_args.dataset_name, pre_transform=pre_transform)
    else:
        dataset = MoleculeNet(path, name='ZINC')

    return dataset


class Dataset(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collate_batch(self, batch):
        return Batch.from_data_list(batch)



def shuffle(dataset, c_train_num, c_val_num, y):
    classes = torch.unique(y)

    indices = []

    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index, val_index, test_index = [], [], []

    for i in range(len(classes)):
        train_index.append(indices[classes[i]][:c_train_num[classes[i]]])

        val_index.append(indices[classes[i]][c_train_num[classes[i]]:(
            c_train_num[classes[i]] + c_val_num[classes[i]])])

        test_index.append(indices[classes[i]][(
            c_train_num[classes[i]] + c_val_num[classes[i]]):])

    train_index = torch.cat(train_index, dim=0)
    val_index = torch.cat(val_index, dim=0)
    test_index = torch.cat(test_index, dim=0)

    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    return train_dataset, val_dataset, test_dataset

# def shuffle(dataset, c_train_num, c_val_num, y):
#     classes = torch.unique(y)
#     indices = []

#     for cls in classes:
#         index = torch.nonzero(y == cls).view(-1)
#         index = index[torch.randperm(index.size(0))]
#         indices.append(index)

#     train_index, val_index, test_index = [], [], []

#     for i, cls in enumerate(classes):
#         train_index.append(indices[i][:c_train_num[i]])
#         val_index.append(indices[i][c_train_num[i]:(c_train_num[i] + c_val_num[i])])
#         test_index.append(indices[i][(c_train_num[i] + c_val_num[i]):])

#     train_index = torch.cat(train_index, dim=0)
#     val_index = torch.cat(val_index, dim=0)
#     test_index = torch.cat(test_index, dim=0)

#     train_dataset = dataset[train_index]
#     val_dataset = dataset[val_index]
#     test_dataset = dataset[test_index]

#     return train_dataset, val_dataset, test_dataset



def upsample(dataset):
    y = torch.tensor([dataset[i].y for i in range(len(dataset))])
    classes = torch.unique(y)

    num_class_graph = [(y == i.item()).sum() for i in classes]

    max_num_class_graph = max(num_class_graph)

    chosen = []
    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        up_sample_ratio = max_num_class_graph / num_class_graph[i]
        up_sample_num = int(
            num_class_graph[i] * up_sample_ratio - num_class_graph[i])

        if(up_sample_num <= len(train_idx)):
            up_sample = random.sample(train_idx, up_sample_num)
        else:
            tmp = int(up_sample_num / len(train_idx))
            up_sample = train_idx * tmp
            tmp = up_sample_num - len(train_idx) * tmp

            up_sample.extend(random.sample(train_idx, tmp))

        chosen.extend(up_sample)

    chosen = torch.tensor(chosen)
    extend_data = dataset[chosen]

    data = list(dataset) + list(extend_data)

    return data