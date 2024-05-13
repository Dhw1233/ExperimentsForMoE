import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from agent_utils import greedy_select_action, select_gpus
INIT = configs.Init


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.mlp = nn.Linear(in_features, out_features)
        self.mlp_neighbors = nn.Linear(in_features, out_features)
        self.epsilon = nn.Parameter(torch.randn(1))

    def forward(self, x, adj):
        # 节点自身特征的转换
        self_feature = self.mlp((1 + self.epsilon) * x)
        # 邻居特征的加权求和
        neighbor_sum = torch.bmm(adj, x)  # 计算邻居特征的加权和
        neighbor_feature = self.mlp_neighbors(neighbor_sum)  # 通过另一个 MLP 调整邻居特征维度
        # 合并特征
        out = self_feature + neighbor_feature
        return F.relu(out)


class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GraphConvolutionLayer(in_features, hidden_features))
        # 隐藏层
        for _ in range(num_layers - 1):
            self.layers.append(GraphConvolutionLayer(hidden_features, hidden_features))
        # 输出层，输出维度为 expert 数量
        self.layers.append(GraphConvolutionLayer(hidden_features, out_features))

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.relu(layer(x, adj))
        return F.softmax(x, dim=-1)  # 使用 softmax 保证输出是概率分布


class ExpertEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, device):
        super(ExpertEncoder, self).__init__()
        self.gnn = GNN(in_features, hidden_features, out_features, num_layers).to(device)

    def forward(self, expert_nodes, expert_links):
        # expert_nodes 的形状为 [batch_size, num_experts, features]
        # expert_links 的形状为 [batch_size, num_experts, num_experts]
        return self.gnn(expert_nodes, expert_links)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPActor, self).__init__()
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


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
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


class Expert_Actor(nn.Module):
    def __init__(self,
                 n_moe_layer,
                 n_e,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Expert_Actor, self).__init__()
        # expert_select size for problems
        self.n_moe_layer = n_moe_layer
        self.n_e = n_e
        self.device = device
        self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        # gpu size for problems
        self.device = device
        self.encoder = ExpertEncoder(in_features = input_dim, hidden_features = hidden_dim, out_features = output_dim, num_layers = num_layers, device = device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor1 = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)


    def forward(self, 
                expert_nodes, 
                expert_links, 
                graph_pool,
                padded_nei, 
                mask_expert, 
                old_policy=True,
                T=1,
                greedy=True):
        
        h_nodes = self.encoder(expert_nodes=expert_nodes, expert_links=expert_links)
        print("Encoded expert nodes shape:", h_nodes.shape)

        if old_policy:
            
            if greedy:
                probs = h_nodes[:, :, -1]
                masked_probs = probs.clone()
                masked_probs[mask_expert] = float('-inf')
                masked_probs = F.softmax(masked_probs, dim=1)
                _, indices = torch.max(masked_probs, dim=1)
            else:
                print("Greedy Select Expert Needed!\n")

            print("Selected expert indices: batch 0 = ", indices[0], ", batch N = ", indices[1])
            h_pooled = graph_pool(h_nodes)

            return indices, masked_probs, h_pooled


class GPU_Actor(nn.Module):
    def __init__(self, expert_feature_dim, gpu_feature_dim, num_gpus, device):
        super(GPU_Actor, self).__init__()
        self.device = torch.device(device)
        # Feature dimensions setup
        self.expert_feature_dim = expert_feature_dim
        self.gpu_feature_dim = gpu_feature_dim
        self.num_gpus = num_gpus

        # Encoders for the expert and GPU features
        self.expert_feature_encoder = nn.Linear(expert_feature_dim, num_gpus).to(self.device)
        self.gpu_feature_encoder = nn.Linear(gpu_feature_dim, num_gpus).to(self.device)

        # Decoder to combine features and make decisions
        self.decoder = nn.Sequential(
            nn.Linear(num_gpus * 2, num_gpus * 2),  # Combine expert and GPU features
            nn.ReLU(),
            nn.Linear(num_gpus * 2, num_gpus)  # Output a score for each GPU
        ).to(self.device)

    def forward(self, expert_node, expert_links, gpu_nodes, gpu_links, mask_gpu_action):
        
        # Ensure all inputs are PyTorch tensors
        expert_node = torch.as_tensor(expert_node).float().to(self.device)  # Convert to tensor and ensure type is float
        gpu_nodes = torch.as_tensor(gpu_nodes).float().to(self.device)  # Convert and ensure type

        # Encoding
        expert_features = self.expert_feature_encoder(expert_node)  # [batch_size, num_gpus]
        gpu_features = self.gpu_feature_encoder(gpu_nodes.reshape(-1, self.gpu_feature_dim)).view(-1, self.num_gpus, self.num_gpus)
        gpu_features = torch.mean(gpu_features, dim=2)  # [64, 4]

        print("expert_features = ", expert_features.shape, ", gpu_features = ", gpu_features.shape)
        combined_features = torch.cat([expert_features, gpu_features], dim=1)  # Concat along feature dim

        # Decoding to make placement decisions
        decision_logits = self.decoder(combined_features)
        decision_probs_ = F.softmax(decision_logits, dim=1)
        decision_probs_[mask_gpu_action] = 0.0

        decision_probs = torch.sigmoid(decision_logits)
        decision_bool = decision_probs > 0.5

        # Apply mask
        decision_bool[mask_gpu_action] = False

        return decision_bool, decision_probs_



if __name__ == '__main__':
    print('Go home')