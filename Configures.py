import os
import torch
from typing import List


class DataParser():
    def __init__(self):
        super().__init__()
        self.dataset_name = 'ogbg-molhiv'                      # ['MUTAG', 'DHFR', 'ogbg-molhiv', 'ogbg-molpcba', 'ZINC']
        self.dataset_dir = './datasets'

        self.random_split: bool = True
        self.data_split_ratio: List = [120/756, 120/756, 516/756]   # the ratio of training, validation and testing set for random split
        self.seed = 1

        self.imb = False                                 # To control if dataset imbalance
        self.imb_ratio = 0.1                            # Imbalance Ratio
        self.num_train = 150                            
        self.num_val = 150



class ModelParser():
    def __init__(self):
        super().__init__()
        self.device: int = 0
        self.model_name: str = 'pna'
        self.checkpoint: str = './checkpoint'
        self.readout: 'str' = 'max'                    # the graph pooling method
        self.enable_prot = True                        # whether to enable prototype training
        self.num_prototypes_per_class = 1              # the num_prototypes_per_class

        self.deg = None
        self.edge_dim = None

        self.pe_dim = 20
        self.prot_dim = 32
        self.single_target = True

        self.mlp_out_dim = 0


    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass



class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.batch_size = 24
        self.weight_decay = 0
        self.max_epochs = 200
        self.save_epoch = 20
        self.early_stopping = 50000

        self.retain_ratio = 1.0                       # retain ratio
        self.pruning_epochs = 0                       # pruning from epoch >= pruning_epochs
        self.imb_solve: str = 'no'                     # [upsampling, smote, overall_reweight]
        self.visualize = 0                             
        self.pre_transform = None
        self.num_workers = 0
        self.cnt = 1


data_args = DataParser()
model_args = ModelParser()
train_args = TrainParser()

import torch
import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
