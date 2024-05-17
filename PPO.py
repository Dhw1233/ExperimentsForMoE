import torch 
from Params import configs
from models.PPO_Actor1 import Expert_Actor, GPU_Encoder, MLPCritic
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import numpy as np
from memory import Memory
from torch.distributions.categorical import Categorical
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):

class PPO:
    #PPO的初始化
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_moe_layer, # number of moe layers
                 n_e, # the total number of experts 
                 n_g, # number of gpus
                 GCN_num_layers, # for GCNN, number of layers in the neural networks (INCLUDING the input layer), 每一层可能使用一个 MLP 来处理节点的特征
                 neighbor_pooling_type, 
                 input_dim,
                 hidden_dim,
                 GCN_num_mlp_layers_feature_extract, # for GCNN, number of layers in mlps (EXCLUDING the input layer), 指定了每个 MLP 的内部层数
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.policy_expert = Expert_Actor(
                                    feature_dim = configs.expert_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    num_layers = configs.num_layers, 
                                    num_experts = configs.n_e, 
                                    num_mlp_layers = configs.num_mlp_layers_feature_extract).to(device)
        self.gpu_encoder = GPU_Encoder(
                                    node_feature_dim = configs.gpu_feature_dim, 
                                    hidden_dim = configs.hidden_dim, 
                                    output_dim = 1, 
                                    num_layers = configs.num_layers
        ).to(device)
        self.policy_critic = MLPCritic(num_layers = num_mlp_layers_critic, 
                                        input_dim = configs.output_dim + configs.n_g, # expert + gpu array
                                        hidden_dim = configs.hidden_dim, 
                                        output_dim = 1).to(device)

        self.policy_old_expert = deepcopy(self.policy_expert)
        self.policy_old_gpu = deepcopy(self.gpu_encoder)

        self.policy_old_expert.load_state_dict(self.policy_expert.state_dict())
        self.policy_old_gpu.load_state_dict(self.gpu_encoder.state_dict())

        self.expert_optimizer = torch.optim.Adam(self.policy_expert.parameters(), lr=lr)
        self.gpu_optimizer = torch.optim.Adam(self.gpu_encoder.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.policy_critic.parameters(), lr=lr)

        self.expert_scheduler = torch.optim.lr_scheduler.StepLR(self.expert_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
        self.gpu_scheduler = torch.optim.lr_scheduler.StepLR(self.gpu_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)

        self.MSE = nn.MSELoss()

    #PPO的更新
    def update(self, memories, epoch):
        '''self.policy_expert.train()
        self.policy_gpu.train()
        self.policy_critic.train()'''

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []
        # 计算折扣奖励并进行标准化
        for rewards_list, dones_list in zip(memories.env_rewards, memories.env_done):
            rewards = []
            discounted_reward = 0

            rewards_list = rewards_list.squeeze() # 转换为一维张量
            dones_list = dones_list.squeeze()

            for reward, is_terminal in zip(reversed(rewards_list), reversed(dones_list)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0).squeeze()
        
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            
            expert_log_prob = []
            gpu_log_prob = []
            val = []
            gpu_select = []
            entropies = []
            expert_entropy = []
            gpu_entropy = []
            
            expert_log_old_prob = memories.expert_logprobs[0]
            gpu_log_old_prob = memories.gpu_logprobs[0]

            for i in range(len(memories.expert_node_fea)):
                env_expert_nodes = memories.expert_node_fea[i]
                env_expert_links = memories.expert_link_fea[i]
                env_gpu_nodes = memories.gpu_node_fea[i]
                env_gpu_links = memories.gpu_link_fea[i]
                old_expert = memories.expert_selection[i]
                env_mask_expert = memories.mask_expert[i]
                env_mask_gpu = memories.mask_gpu[i]

                expert_selects,expert_probs = self.policy_expert(
                                                    node_features = memories.expert_node_fea[i],
                                                    adj_matrix = memories.expert_link_fea[i],
                                                    env_mask_expert = memories.mask_expert[i]
                                                    )
                selected_expert_features = env_expert_nodes[:, expert_indices[0], :] # torch.Size([64, 2])
                selected_expert_links = env_expert_links[:, expert_indices[0], :] # torch.Size([64, 32])

                gpu_bool_array, gpu_prob = self.policy_gpu(
                                                expert_node = selected_expert_features, 
                                                expert_links = selected_expert_links,
                                                gpu_nodes = env_gpu_nodes, 
                                                gpu_links = env_gpu_links, 
                                                pooling_type = configs.graph_pool_type,
                                                mask_gpu_action = env_mask_gpu)
                print("\nexpert_prob[batch 0] = ", expert_prob[0], "\ngpu_prob[batch 0] = ", gpu_prob[0], "\n")
                # Combine (action_e, action_g)
                critic_input = torch.cat([h_pooled, gpu_bool_array.float()], dim=1)
                v = self.policy_critic(critic_input)
                val.append(v)

                # Calculate the log probabilities
                expert_log_prob.append(torch.log(expert_prob + 1e-10))
                gpu_log_prob.append(torch.log(gpu_prob + 1e-10))

                # Calculate the entropies
                expert_dist = Categorical(expert_prob)
                expert_entropy.append(expert_dist.entropy())

                gpu_dist = Categorical(gpu_prob)
                gpu_entropy.append(gpu_dist.entropy())

            # Convert lists to tensors
            expert_log_prob, expert_log_old_prob = torch.cat(expert_log_prob, dim=0), torch.cat(expert_log_old_prob, dim=0) # torch.Size([64, 32])
            gpu_log_prob, gpu_log_old_prob = torch.cat(gpu_log_prob, dim=0), torch.cat(gpu_log_old_prob, dim=0) # torch.Size([64, 4])
            
            val = torch.cat(val).squeeze() # torch.Size([64])
            expert_entropy = torch.cat(expert_entropy).squeeze() # torch.Size([64])
            gpu_entropy = torch.cat(gpu_entropy).squeeze() # torch.Size([64])

            # Compute advantages
            advantages = rewards_all_env - val.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            expert_loss_sum = torch.zeros(1, device=device)
            gpu_loss_sum = torch.zeros(1, device=device)
            value_loss_sum = torch.zeros(1, device=device)
            for j in range(configs.batch_size):
                expert_ratios = torch.exp(expert_log_prob[j] - expert_log_old_prob[j].detach()) # torch.Size([32])
                gpu_ratios = torch.exp(gpu_log_prob[j] - gpu_log_old_prob[j].detach()) #  torch.Size([4])

                expert_surr1 = expert_ratios * advantages[j] # torch.Size([32])
                expert_surr2 = torch.clamp(expert_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j] # torch.Size([32])
                expert_loss = -1 * torch.min(expert_surr1, expert_surr2) - entloss_coef * expert_entropy[j]
                expert_loss_sum += expert_loss.sum() # torch.Size([1])

                gpu_surr1 = gpu_ratios * advantages[j]
                gpu_surr2 = torch.clamp(gpu_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j]
                gpu_loss = -1 * torch.min(gpu_surr1, gpu_surr2) - entloss_coef * gpu_entropy[j]
                gpu_loss_sum += gpu_loss.sum() # torch.Size([1])

                value_loss = self.MSE(val[j], rewards_all_env[j])
                value_loss_sum += value_loss # torch.Size([1])

            # Calculate the total loss
            total_expert_loss = ploss_coef * expert_loss_sum / configs.batch_size
            total_gpu_loss = ploss_coef * gpu_loss_sum / configs.batch_size
            total_value_loss = vloss_coef * value_loss_sum / configs.batch_size

            # take gradient step, scheduler.step()
            self.expert_optimizer.zero_grad()
            total_expert_loss.backward(retain_graph=True)
            self.expert_optimizer.step()

            self.gpu_optimizer.zero_grad()
            total_gpu_loss.backward(retain_graph=True)
            self.gpu_optimizer.step()

            self.value_optimizer.zero_grad()
            total_value_loss.backward()
            self.value_optimizer.step()

            # Copy new weights into old policy
            self.policy_old_expert.load_state_dict(self.policy_expert.state_dict())
            self.policy_old_gpu.load_state_dict(self.policy_gpu.state_dict())
            
            if configs.decayflag:
                self.expert_scheduler.step()
                self.gpu_scheduler.step()
                self.value_scheduler.step()

            return expert_loss_sum.mean().item(), gpu_loss_sum.mean().item(), value_loss_sum.mean().item()
    
