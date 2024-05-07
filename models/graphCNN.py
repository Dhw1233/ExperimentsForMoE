import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 第一层从input_dim到hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 中间层，全部使用hidden_dim
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层，从hidden_dim到output_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # 更新：确保最后一层的BatchNorm与输出维度匹配
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

    def forward(self, x):
        for layer, bn in zip(self.layers, self.batch_norms):
            x = F.relu(bn(layer(x)))
        return x



class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPActor, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, learn_eps, neighbor_pooling_type, device):
        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        # Setup MLPs for each layer
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for layer in range(num_layers):
            if layer < num_layers - 1:
                # Use hidden_dim as the output dimension for all but the last MLP
                self.mlps.append(MLP(num_mlp_layers, input_dim if layer == 0 else hidden_dim, hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                # Use output_dim for the output dimension of the last MLP
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, output_dim))
                self.batch_norms.append(nn.BatchNorm1d(output_dim))


    def forward(self, node_features, adj_matrices):
        for i, mlp in enumerate(self.mlps):
            node_features = mlp(node_features)
            if i < len(self.batch_norms):
                node_features = self.batch_norms[i](node_features)
            if i < len(self.mlps) - 1:  # No adjacency matrix multiplication in the last layer
                node_features = torch.mm(adj_matrices, node_features)
        node_features_reshaped = node_features.view(64, 32, 1)
        print("GNN encoder output: ", node_features_reshaped.shape)
        return node_features_reshaped

