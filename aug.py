import numpy as np
from torch_geometric.utils.dropout import dropout_adj


def remove_edge(edge_index, drop_ratio):
    edge_index, _ = dropout_adj(edge_index, p = drop_ratio)

    return edge_index


def drop_node(x, drop_ratio):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_ratio)

    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()

    x[idx_mask] = 0

    return x


def get_class_num(imb_ratio, num_train, num_val):
    c_train_num = [int(imb_ratio * num_train), num_train -
                   int(imb_ratio * num_train)]

    c_val_num = [int(imb_ratio * num_val), num_val - int(imb_ratio * num_val)]

    return c_train_num, c_val_num

# N class
# def get_class_num(imb_ratio, num_train, num_val, num_classes):
#     c_train_num = [int(imb_ratio * num_train) if i == 0 else int((1 - imb_ratio) * num_train / (num_classes - 1)) for i in range(num_classes)]
#     c_val_num = [int(imb_ratio * num_val) if i == 0 else int((1 - imb_ratio) * num_val / (num_classes - 1)) for i in range(num_classes)]

#     return c_train_num, c_val_num