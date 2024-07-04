import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from agent_utils import greedy_select_action, select_gpus
INIT = configs.Init

def initialize_weights(model):
    for name, p in model.named_parameters():
        if 'weight' in name:
            if len(p.size()) >= 2:
                nn.init.orthogonal_(p, gain=1)
        elif 'bias' in name:
            nn.init.constant_(p, 0)

class MLP(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        if INIT:
            self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output_layer(x)

class MLPbm(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLPbm, self).__init__()
        # self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        # self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        if INIT:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # for layer in self.layers:
        #     x = self.relu(layer(x))
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.relu(batch_norm(layer(x)))
        return self.output_layer(x)

class MLPtanh(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers and Tanh activation """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLPtanh, self).__init__()
        # self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        # self.output_layer = nn.Linear(hidden_dim, output_dim)
        # self.tanh = nn.Tanh()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        if INIT:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # for layer in self.layers:
        #     x = self.tanh(layer(x))
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.tanh(batch_norm(layer(x)))
        return self.output_layer(x)

class GNNLayer(nn.Module):
    """ Single layer of Graph Neural Network """
    def __init__(self, feature_dim, eps, hidden_dim, output_dim, num_mlp_layers):
        super(GNNLayer, self).__init__()
        self.mlp = MLP(feature_dim, hidden_dim, output_dim, num_mlp_layers)
        self.eps = eps
        if INIT:
            initialize_weights(self)

    def forward(self, h, adj):
        # Aggregate neighbor features
        sum_neighbors = torch.bmm(adj, h)
        # Node feature update
        new_h = self.mlp((1 + self.eps) * h + sum_neighbors)
        return new_h

class GNN(nn.Module):
    """ Graph Neural Network consisting of multiple GNN layers """
    def __init__(self, feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        eps_values = np.random.uniform(0.01, 0.1, num_layers)

        self.layers.append(GNNLayer(feature_dim, eps_values[0], hidden_dim, hidden_dim, num_mlp_layers))
        for i in range(1, num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, eps_values[i], hidden_dim, hidden_dim, num_mlp_layers))
        self.layers.append(GNNLayer(hidden_dim, eps_values[-1], hidden_dim, output_dim, num_mlp_layers))
        if INIT:
            initialize_weights(self)

    def forward(self, h, adj):
        for layer in self.layers:
            h = layer(h, adj)
        return h

class Expert_Encoder(nn.Module):
    """ Encoder for experts using GNN """
    def __init__(self, expert_feature_dim, hidden_dim, output_dim, num_layers, num_mlp_layers):
        super(Expert_Encoder, self).__init__()
        self.gnn = GNN(expert_feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, node_features, adj_matrix):
        node_embeddings = self.gnn(node_features, adj_matrix)
        global_embedding = node_embeddings.mean(dim=1)
        return node_embeddings, global_embedding


class GPU_Encoder(nn.Module):
    """ Encoder for GPUs using GNN """
    def __init__(self, gpu_feature_dim, hidden_dim, output_dim, num_layers, num_mlp_layers):
        super(GPU_Encoder, self).__init__()
        self.gnn = GNN(gpu_feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, gpu_nodes, gpu_links):
        node_embeddings = self.gnn(gpu_nodes, gpu_links[:,:,:,0])
        global_embedding = node_embeddings.mean(dim=1)
        return node_embeddings, global_embedding
    

class Expert_Decoder(nn.Module):
    """ Multi-Layer MLP Actor for decoding actions based on combined embeddings """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Expert_Decoder, self).__init__()
        self.mlp = MLPbm(input_dim, hidden_dim, output_dim, num_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, h_node, h_global, h_pooled_gpu, mask):
        # print(f"Expert_Decoder forward: h_node.shape = {h_node.shape}, h_global.shape = {h_global.shape}, u.shape = {u.shape}")
        # Expand h_global and concatenate it with h_node and u
        h_global_expanded = h_global.unsqueeze(1).expand(-1, h_node.size(1), -1)
        u_expanded = h_pooled_gpu.unsqueeze(1).expand(-1, h_node.size(1), -1)
        h_combined = torch.cat([h_node, h_global_expanded, u_expanded], dim=-1)
        # print(f"h_combined.shape = {h_combined.shape}")

        batch_size, expert_num, feature_dim = h_combined.size()
        h_combined = h_combined.view(batch_size * expert_num, feature_dim)
        action_scores = self.mlp(h_combined).view(batch_size, expert_num, -1).squeeze(-1)

        # action_scores = self.mlp(h_combined).squeeze(-1)
        # Masking and softmax
        action_scores = action_scores.masked_fill(mask, float('-inf')).squeeze(-1)
        action_probs = F.softmax(action_scores, dim=1)
        
        # Sampling or greedy action selection
        if self.training:  # Sampling strategy during training
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
        else:  # Greedy strategy during evaluation
            action = torch.argmax(action_probs, dim=1)

        return action_probs, action


class GPU_Decoder(nn.Module):
    """ Actor to compute GPU action scores based on combined embeddings """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GPU_Decoder, self).__init__()
        self.mlp = MLPtanh(input_dim, hidden_dim, output_dim, num_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, h_node, h_global, h_expert, mask):
        h_global_expanded = h_global.unsqueeze(1).expand(-1, h_node.size(1), -1)
        u_expanded = h_expert.unsqueeze(1).expand(-1, h_node.size(1), -1)
        h_combined = torch.cat([h_node, h_global_expanded, u_expanded], dim=-1)
        batch_size, gpu_num, feature_dim = h_combined.size() 
        h_combined = h_combined.view(batch_size * gpu_num, feature_dim)
        action_scores = self.mlp(h_combined).view(batch_size, gpu_num, -1).squeeze(-1)

        # Masking and softmax
        action_scores = action_scores.masked_fill(mask, float('-inf')).squeeze(-1)

        action_probs = F.softmax(action_scores, dim=1)

        # Determine expand/shrink actions
        if self.training:  # Sampling strategy during training
            distribution = torch.distributions.Categorical(action_probs)
            gpu_chosen = distribution.sample()
        else:  # Greedy strategy during evaluation
            gpu_chosen = torch.argmax(action_probs, dim=1)
        return action_probs, gpu_chosen

class EXPERT_ACTOR(nn.Module):
    def __init__(self,expert_feature_dim, 
                 gpu_feature_dim,
                 hidden_dim, 
                 expert_output_dim,
                 gpu_output_dim, 
                 num_layers, 
                 num_encoder_layers,
                 num_decoder_layers,
                 num_critic_layers,
                 num_experts,
                 num_gpus,
                 old_policy,
                 *args,
                 **kwargs):
        super(EXPERT_ACTOR,self).__init__()

        self.old_policy = old_policy

        self.ep_encoder = Expert_Encoder(expert_feature_dim=expert_feature_dim, hidden_dim=hidden_dim,
                                          output_dim=expert_output_dim, num_layers=num_layers,
                                            num_mlp_layers=num_encoder_layers)
        
        self.gp_encoder = GPU_Encoder( gpu_feature_dim=gpu_feature_dim, hidden_dim=hidden_dim,
                                       output_dim=gpu_output_dim, num_layers=num_layers,
                                         num_mlp_layers=num_encoder_layers)
        
        self.ep_decoder = Expert_Decoder(input_dim=2*expert_output_dim+gpu_output_dim, hidden_dim=hidden_dim,
                                        output_dim=1, num_layers = num_decoder_layers)
        
        
        self.critic = MLPCritic(num_layers=num_critic_layers,
                                input_dim=num_experts*expert_output_dim+num_gpus*gpu_output_dim,
                                hidden_dim=hidden_dim,
                                output_dim=1)
        
    def forward(self,ep_nodes, ep_links,gp_nodes, gp_links,mask_ep,mask_gp,old_policy):

        h_ep_node,h_ep_global = self.ep_encoder(ep_nodes,ep_links)

        h_gp_node,h_gp_global = self.gp_encoder(gp_nodes,gp_links)
        
        if  old_policy:
            ep_probs,ep_index = self.ep_decoder(h_ep_node,h_ep_global,h_gp_global,mask_ep)
            
            return ep_probs,ep_index
        
        else:
            ep_probs,ep_index = self.ep_decoder(h_ep_node,h_ep_global,h_gp_global,mask_ep)

            val = self.critic(torch.cat([h_ep_node.view(-1,h_ep_node.size(1)*h_ep_node.size(2)),h_gp_node.view(-1,h_gp_node.size(1)*h_gp_node.size(2))],dim=1))

            return ep_probs,ep_index,val

class GPU_ACTOR(nn.Module):
    def __init__(self,expert_feature_dim, 
                 gpu_feature_dim,
                 hidden_dim, 
                 expert_output_dim,
                 gpu_output_dim, 
                 num_layers, 
                 num_encoder_layers,
                 num_decoder_layers,
                 num_critic_layers,
                 num_experts,
                 num_gpus,
                 old_policy,
                 *args,
                 **kwargs):
        super(GPU_ACTOR,self).__init__()

        self.old_policy = old_policy

        self.ep_encoder = Expert_Encoder(expert_feature_dim=expert_feature_dim, hidden_dim=hidden_dim,
                                          output_dim=expert_output_dim, num_layers=num_layers,
                                            num_mlp_layers=num_encoder_layers)
        
        self.gp_encoder = GPU_Encoder( gpu_feature_dim=gpu_feature_dim, hidden_dim=hidden_dim,
                                       output_dim=gpu_output_dim, num_layers=num_layers,
                                         num_mlp_layers=num_encoder_layers)
        
        self.gp_decoder = GPU_Decoder(input_dim=2*gpu_output_dim+expert_output_dim,hidden_dim=hidden_dim,
                                    output_dim=1,num_layers=num_decoder_layers)
        
        self.ac_decoder = MLPCritic(num_layers=1,
                                    input_dim=gpu_output_dim,
                                    hidden_dim=hidden_dim,
                                    output_dim=2)
        
        self.critic = MLPCritic(num_layers=num_critic_layers,
                                input_dim=num_experts*expert_output_dim+num_gpus*gpu_output_dim,
                                hidden_dim=hidden_dim,
                                output_dim=1)
        
    def forward(self,ep_nodes, ep_links,gp_nodes, gp_links,mask_ep,mask_gp,ep_index,old_policy):

        h_ep_node,h_ep_global = self.ep_encoder(ep_nodes,ep_links)

        h_gp_node,h_gp_global = self.gp_encoder(gp_nodes,gp_links)
        
        ep_index = ep_index.unsqueeze(1).unsqueeze(1).expand(-1,-1,h_ep_node.size(2))

        gp_probs,gp_index = self.gp_decoder(h_gp_node,h_gp_global,torch.gather(h_ep_node,1,ep_index).squeeze(1),mask_gp)

        chosen_prob = F.softmax(self.ac_decoder(torch.gather(h_gp_node,1,gp_index)),dim=1)

        distribution = torch.distributions.Categorical(chosen_prob)

        acts = distribution.sample()
        
        return gp_probs,gp_index,chosen_prob,acts
        

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
        if INIT: 
            initialize_weights(self)

    def forward(self, x):
        return self.model(x)