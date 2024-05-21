import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import net_selector
from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU, Embedding
from Configures import data_args, train_args, model_args

# Prototype Layer 
class PrototypeLayer(nn.Module):
    def __init__(self, output_dim, num_prototypes_per_class, prototype_dim=32, epsilon=1e-4, incorrect_strength=-0.5):
        super(PrototypeLayer, self).__init__()
        self.output_dim = output_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_dim = prototype_dim
        self.epsilon = epsilon
        self.prototype_shape = (output_dim * num_prototypes_per_class, prototype_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.num_prototypes = self.prototype_shape[0]
        self.single_target = model_args.single_target
        self.temperature = 0.07

        indices = torch.arange(self.num_prototypes)
        one_hot_vectors = torch.nn.functional.one_hot(indices // num_prototypes_per_class, num_classes=output_dim)
        self.prototype_class_identity = one_hot_vectors.float()

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        return distance

    def proto_nce_loss(self, distances, proto_labels):
        indices = torch.arange(0, self.num_prototypes, device=distances.device)

        class_indices = proto_labels // self.num_prototypes_per_class * self.num_prototypes_per_class

        pos_mask = (indices >= class_indices[:, None]) & (indices < class_indices[:, None] + self.num_prototypes_per_class)

        logits = torch.full_like(distances, -1e6)  

        logits[pos_mask] = -distances[pos_mask] / self.temperature  

        targets = (proto_labels // self.num_prototypes_per_class * self.num_prototypes_per_class).to(distances.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        return loss


    def forward(self, x):
        distances = self.prototype_distances(x)

        if self.single_target is False:
            _, nearest_prototype_indices = torch.min(distances, dim=1)
            virtual_label = self.prototype_class_identity[nearest_prototype_indices].to(distances.device)
            prot_nce_loss = self.proto_nce_loss(distances, virtual_label)
        else:
            virtual_label = None
            prot_nce_loss = None

        return distances, virtual_label, prot_nce_loss


class GNNPrototypeNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GNNPrototypeNet, self).__init__()
        self.gnn = net_selector(model_args.model_name)(input_dim, model_args)

        self.enable_prot = model_args.enable_prot
        self.output_dim = output_dim  
        self.mlp_out_dim = model_args.mlp_out_dim
        if self.mlp_out_dim == 0:
            self.mlp_out_dim = self.output_dim

        self.mlp = Sequential(
            Linear(model_args.prot_dim, model_args.prot_dim // 2),
            ReLU(),
            Linear( model_args.prot_dim // 2, self.mlp_out_dim),
        )

        self.prototype_layer = PrototypeLayer(self.output_dim, model_args.num_prototypes_per_class, 
                                                  prototype_dim=model_args.prot_dim, 
                                                  epsilon=1e-4, incorrect_strength=-0.5)
    
    def get_gnn_layers(self):
        return self.gnn.conv_layers

    def get_prototype_vectors(self):
        return self.prototype_layer.prototype_vectors
    
    def get_prototype_class_identity(self):
        return self.prototype_layer.prototype_class_identity

    def forward(self, data, protgnn_plus, similarity):
        graph_emb = self.gnn(data) 
        if self.enable_prot:
            distances, virtual_label, prot_nce_loss = self.prototype_layer(graph_emb)
            pred = self.mlp(graph_emb)
            return pred, virtual_label, prot_nce_loss, graph_emb, distances
        else:
            pred = self.mlp(graph_emb)
            return pred, None, [], graph_emb, []