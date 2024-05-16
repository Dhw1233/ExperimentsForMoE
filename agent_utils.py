#较为完善
from torch.distributions.categorical import Categorical
import numpy as np
import torch
#一开始每个gpu上只有一个expert
def vanilla_placement(num_expert, num_gpu):
    placement = [[False for _ in range(num_gpu)] for _ in range(num_expert)]
    for i in range(min(num_expert, num_gpu)):
        placement[i][i] = True
    return placement
#按照给定的概率进行gpu的选择，高于某个概率值选，低于则扔，高不成低不就的为不确定状态
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
