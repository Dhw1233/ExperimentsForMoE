import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from agent_utils import greedy_select_action, select_gpus
INIT = configs.Init


class MLP(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output_layer(x)


class GNNLayer(nn.Module):
    """ Single layer of Graph Neural Network """
    def __init__(self, feature_dim, eps, hidden_dim, num_mlp_layers):
        super(GNNLayer, self).__init__()
        self.mlp = MLP(feature_dim, hidden_dim, feature_dim, num_mlp_layers)
        self.eps = eps

    def forward(self, h, adj):
        # Aggregate neighbor features
        sum_neighbors = torch.bmm(adj, h)
        # Node feature update
        new_h = self.mlp((1 + self.eps) * h + sum_neighbors)
        return new_h


class GNN(nn.Module):
    """ Graph Neural Network consisting of multiple GNN layers """
    def __init__(self, feature_dim, num_layers, hidden_dim, num_mlp_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList([GNNLayer(feature_dim, 0.1, hidden_dim, num_mlp_layers) for _ in range(num_layers)])

    def forward(self, h, adj):
        for layer in self.layers:
            h = layer(h, adj)
        return h


class Expert_Decoder(nn.Module):
    """ Decoder to compute action scores based on combined embeddings """
    def __init__(self, node_dim, graph_dim, hidden_dim, num_mlp_layers):
        super(Expert_Decoder, self).__init__()
        self.mlp = MLP(node_dim + graph_dim, hidden_dim, 1, num_mlp_layers)

    def forward(self, h_node, h_global, mask):
        # Expand h_global and concatenate it with h_node
        h_global_expanded = h_global.unsqueeze(1).expand_as(h_node)
        h_combined = torch.cat([h_node, h_global_expanded], dim=-1)
        print("h_combined = ", h_combined.shape)
        action_scores = self.mlp(h_combined).squeeze(-1)
        # Masking and softmax
        action_scores = action_scores.masked_fill(mask, float('-inf'))
        action_probs = F.softmax(action_scores, dim=1)
        return action_probs


class Expert_Actor(nn.Module):
    """ Integrated model for encoding experts' graph and decoding actions """
    def __init__(self, feature_dim, hidden_dim, num_layers, num_experts, num_mlp_layers,*args,**kwargs):
        super(Expert_Actor, self).__init__()
        self.encoder = GNN(feature_dim, num_layers, hidden_dim, num_mlp_layers)
        self.decoder = Expert_Decoder(feature_dim, feature_dim, hidden_dim, num_mlp_layers)

    def forward(self, node_features, adj_matrix, mask):
        # Encode node features
        node_embeddings = self.encoder(node_features, adj_matrix)
        # Compute global graph embedding
        global_embedding = node_embeddings.mean(dim=1)

        # Decode to get action probabilities
        action_probs = self.decoder(node_embeddings, global_embedding, mask)

        # Sampling or greedy action selection
        if self.training:  # Sampling strategy during training
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
        else:  # Greedy strategy during evaluation
            action = torch.argmax(action_probs, dim=1)

        return action_probs, action



class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,*args,**kwargs):
        super(MLPCritic, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # Aggregate features from neighbors
        x_agg = torch.bmm(adj, x)
        # Transform aggregated features
        x_trans = self.fc(x_agg)
        return self.relu(x_trans)


class MultiLayerGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_layers):
        super(MultiLayerGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # First layer
        self.layers.append(GraphConvolutionLayer(node_feature_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(GraphConvolutionLayer(hidden_dim, output_dim))
        
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, node_features, adj_matrix):
        x = node_features
        for layer in self.layers:
            x = layer(x, adj_matrix)
        
        # Apply batch normalization to the output of the last layer
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class GPU_Encoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_layers):
        super(GPU_Encoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn = MultiLayerGNN(node_feature_dim, hidden_dim, output_dim, num_layers)

    def forward(self, node_features, adj_matrix):
        return self.gnn(node_features, adj_matrix)


class GPU_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gpus, num_mlp_layers):
        super(GPU_Decoder, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, num_gpus, num_mlp_layers)

    def forward(self, gpu_embeddings, h_global, mask):
        h_combined = torch.cat([gpu_embeddings, h_global], dim=-1)
        action_scores = self.mlp(h_combined)
        action_scores = action_scores.masked_fill(mask == 0, float('-inf'))
        return torch.sigmoid(action_scores)  # assuming a binary action space