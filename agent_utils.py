from torch.distributions.categorical import Categorical
import numpy as np
import torch

def vanilla_placement(n_layers, experts_per_layer, number_of_gpus):
        # 生成每层专家的列表
        experts = np.arange(n_layers * experts_per_layer)
        # 随机打乱并均匀分布
        np.random.shuffle(experts)
        # 将专家均匀分配到每个GPU
        return np.array_split(experts, number_of_gpus)

def select_gpus(p, prob_high, prob_low):
    gpu_selection = torch.full_like(p, -1, dtype=torch.int)  # -1 表示不确定
    gpu_selection[p > prob_high] = 1
    gpu_selection[p < prob_low] = 0
    return gpu_selection

# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# select expert_select method for test
def greedy_select_action(p, candidate, expert_nodes, expert_links):

    _, index = p.squeeze(-1).max(1)
    expert_select = []
    expert_node = []
    expert_adj = []
    for i in range(index.size(0)):
        a = candidate[i][index[i]]
        expert_select.append(a)

        b = expert_nodes[i][index[i]] # historical popularity、current token load
        expert_node.append(b)

        c = expert_links[i][index[i],:]['affinity'] # expert affinity
        expert_adj.append(c)
    expert_select = torch.stack(expert_select, 0)
    expert_node = torch.stack(expert_node, 0)
    return expert_select, expert_node, expert_adj

# select expert_select method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
