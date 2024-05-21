import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, MLP, GATConv, SAGEConv, GPSConv, PNAConv, GINEConv, BatchNorm, ResGatedGraphConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU, Embedding
from torch_geometric.nn.attention import PerformerAttention
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Dict, Optional

def net_selector(net_name:str):
    if net_name in ["gcn"]:
        return GCN
    elif net_name in ["pna"]:
        return PNA
    elif net_name in ["gps"]:
        return GPS

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout



class GCN(torch.nn.Module):
    def __init__(self, input_dim, model_args):
        super().__init__()
        self.hidden_channels = 384
        self.out_channels = model_args.prot_dim
        self.conv1 = GCNConv(input_dim, self.hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels,
                             normalize=True)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels,
                             normalize=True)
        self.conv4 = GCNConv(self.hidden_channels, self.out_channels,
                             normalize=True)
        self.conv_layers = ModuleList()
        self.conv_layers.append(self.conv1)
        self.conv_layers.append(self.conv2)
        self.conv_layers.append(self.conv3)
        self.conv_layers.append(self.conv4)
        self.readout_layers = get_readout_layers(model_args.readout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = None
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)
        pooled = [readout(x, batch) for readout in self.readout_layers]
        x = torch.cat(pooled, dim=-1)
        return x
    


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

class GPS(torch.nn.Module):
    def __init__(self, input_dim, model_args):
        super().__init__()

        self.channels = 64
        self.pe_dim = 16

        self.node_emb = Linear(input_dim, self.channels - self.pe_dim)
        self.pe_lin = Linear(model_args.pe_dim, self.pe_dim)
        self.pe_norm = BatchNorm1d(model_args.pe_dim)

        self.edge_dim = model_args.edge_dim

        if self.edge_dim != 0:
            self.edge_emb = Linear(self.edge_dim, self.channels)

        self.attn_type = 'multihead'
        self.attn_kwargs = {'dropout': 0.5}
        self.num_layers = 4

        self.conv_layers = ModuleList()

        for _ in range(self.num_layers):
            nn = Sequential(
                Linear(self.channels, self.channels),
                ReLU(),
                Linear(self.channels, self.channels),
            )
            if self.edge_dim == 0:
                conv = GPSConv(self.channels, GINConv(nn), heads=4,
                           attn_type=self.attn_type, attn_kwargs=self.attn_kwargs)
            else:
                # conv = GPSConv(self.channels, ResGatedGraphConv(in_channels=self.channels,out_channels=self.channels,edge_dim=self.channels), heads=4,
                #            attn_type=self.attn_type, attn_kwargs=self.attn_kwargs)
                conv = GPSConv(self.channels,GINEConv(nn), heads=4,
                           attn_type=self.attn_type, attn_kwargs=self.attn_kwargs)
            self.conv_layers.append(conv)

        self.mlp = Sequential(
            Linear(self.channels, model_args.prot_dim)
        )
        self.redraw_projection = RedrawProjection(
            self.conv_layers,
            redraw_interval=1000 if self.attn_type == 'performer' else None)

    def forward(self, data):
        if self.edge_dim != 0:
            edge_attr = data.edge_attr.float()
            edge_attr = self.edge_emb(edge_attr)

        x, pe, edge_index, batch = data.x.float(), data.pe.float(), data.edge_index, data.batch
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        for conv in self.conv_layers:
            if self.edge_dim != 0:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch)
        x = global_mean_pool(x, batch)
        return self.mlp(x)



class PNA(torch.nn.Module):
    def __init__(self, input_dim, model_args):
        super().__init__()

        self.node_dim = 64
        self.edge_dim = model_args.edge_dim
        self.node_emb = Linear(input_dim, self.node_dim)
        self.edge_emb = Linear(self.edge_dim, 16)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.conv_layers = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(4):
            if self.edge_dim != 0:
                conv = PNAConv(in_channels=self.node_dim, out_channels=self.node_dim,
                           aggregators=aggregators, scalers=scalers, deg=model_args.deg,
                           edge_dim=16, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)
            else:
                conv = PNAConv(in_channels=self.node_dim, out_channels=self.node_dim,
                           aggregators=aggregators, scalers=scalers, deg=model_args.deg,
                           edge_dim=None, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.conv_layers.append(conv)
            self.batch_norms.append(BatchNorm(self.node_dim))

        self.mlp = Sequential(Linear(self.node_dim, model_args.prot_dim))

    def forward(self, data):
        if self.edge_dim != 0:
            edge_attr = data.edge_attr.float()
            edge_attr = self.edge_emb(edge_attr)

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.node_emb(x.squeeze())

        for conv, batch_norm in zip(self.conv_layers, self.batch_norms):
            if self.edge_dim != 0:
                x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            else:
                x = F.relu(batch_norm(conv(x, edge_index)))

        x = global_add_pool(x, batch)
        return self.mlp(x)