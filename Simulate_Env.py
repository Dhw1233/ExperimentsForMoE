import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance import override
from Params import configs
from agent_utils import vanilla_placement
from copy import deepcopy
import torch
import random
import matplotlib.pyplot as plt
import time
class Simulate_Env(gym.Env, EzPickle):
    def __init__(self,
                 n_moe_layer,
                 n_e,
                 n_g):
        EzPickle.__init__(self)

        self.step_count = 0
        self.n_moe_layer = n_moe_layer
        self.number_of_experts = n_e
        self.number_of_gpus = n_g
        # the task id for first column
        self.candidate = []
        #self.expert_affinity = getExpertAffinity # 专家亲和力矩阵 待完成！！


    @override
    def reset(self, data):
        # 重置各类计数器和矩阵，为新的环境做准备
        # data (n_sample, n_expert, n_expert)
        self.batch_size = data.shape[0]

        experts_per_layer = self.number_of_experts // self.n_moe_layer

        self.step_count = 0
        # 跟踪各专家对GPU资源的分配状态，-1 代表未分配
        self.history_expert_gpu = np.zeros((self.batch_size, self.number_of_experts, self.number_of_gpus), dtype=bool)
        self.history_gpu_expert = np.zeros((self.batch_size, self.number_of_gpus, self.number_of_experts), dtype=bool)
        self.expert_token = data.astype(np.single) # single单精度浮点数

        self.posRewards = np.zeros(self.batch_size)
        self.expert_links = []
        self.expert_adj = []
        # 生成每个GPU上放置的专家id的列表
        vanilla_p = vanilla_placement(self.number_of_experts, self.number_of_gpus) # (n_expert, n_gpu)

        # initialize expert_links[batchsize, n_e, n_e, fea_dim] : traffic
        self.expert_links = np.zeros((self.batch_size, self.number_of_experts, self.number_of_experts)) # expert traffic
        self.expert_adj = np.zeros((self.batch_size, self.number_of_experts, self.number_of_experts)) # expert affinity
        for k in range(self.batch_size):
            for layer in range(self.n_moe_layer - 1):
                start_index = layer * experts_per_layer
                end_index = start_index + experts_per_layer

                down_start_index = (layer + 1) * experts_per_layer
                down_end_index = down_start_index + experts_per_layer

                random_weights = np.random.rand(experts_per_layer, experts_per_layer)
                row_sums = random_weights.sum(axis=1, keepdims=True)
                normalized_weights = random_weights / row_sums
                for i in range(experts_per_layer):
                    for j in range(experts_per_layer):
                        self.expert_adj[k, start_index + i, down_start_index + j] = normalized_weights[i, j]
                        self.expert_links[k, start_index + i, down_start_index + j] = data[k, start_index + i, down_start_index + j]
        self.expert_links = torch.tensor(self.expert_links, dtype=torch.float32)
        self.expert_adj = torch.tensor(self.expert_adj, dtype=torch.float32)

        # 根据 vanilla_p 初始化 history_expert_gpu、history_gpu_expert
        for k in range(self.batch_size):
            for expert_id in range(self.number_of_experts):
                for gpu_id in range(self.number_of_gpus):
                    if vanilla_p[expert_id][gpu_id]:
                        self.history_expert_gpu[k][expert_id][gpu_id] = True
                        self.history_gpu_expert[k][gpu_id][expert_id] = True
        
        # initialize self.gpu_links matrix : traffic rate
        self.gpu_links = np.zeros((self.batch_size, self.number_of_gpus, self.number_of_gpus))
        self.gpu_total_traffic = np.zeros((self.batch_size, self.number_of_gpus))
        for k in range(self.batch_size):
            # 先计算每个 GPU 的总发送数据量
            for gpu_id in range(self.number_of_gpus):
                for other_gpu_id in range(self.number_of_gpus):
                    if gpu_id != other_gpu_id:
                        for expert_id in range(self.number_of_experts):
                            for other_expert_id in range(self.number_of_experts):
                                if self.history_gpu_expert[k][gpu_id][expert_id] and self.history_gpu_expert[k][other_gpu_id][other_expert_id]:
                                    traffic = data[k, expert_id, other_expert_id]
                                    self.gpu_links[k, gpu_id, other_gpu_id] += traffic
                                    self.gpu_total_traffic[k, gpu_id] += traffic
        # 然后计算比例
        for k in range(self.batch_size):
            for gpu_id in range(self.number_of_gpus):
                if self.gpu_total_traffic[k, gpu_id] > 0:  # 避免除以0
                    for other_gpu_id in range(self.number_of_gpus):
                        self.gpu_links[k, gpu_id, other_gpu_id] /= self.gpu_total_traffic[k, gpu_id]
        self.gpu_links = torch.tensor(self.gpu_links, dtype=torch.float32)

        # 初始化 专家的token数据量
        self.current_token = np.zeros((self.batch_size, self.number_of_experts))
        for k in range(self.batch_size):
            for i in range(self.number_of_experts):
                total_tokens = np.sum(data[k, i, :])  # 专家i发送给所有其他专家的token总数
                self.current_token[k, i] = total_tokens

        # 初始化 expert_nodes: historical popularity、current token load
        self.history_popularity = np.random.rand(self.batch_size, self.number_of_experts).astype(float)
        self.expert_nodes = np.concatenate(
            [   self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)],
            axis=2  # 沿最后一个维度拼接
        )
        self.expert_nodes = torch.tensor(self.expert_nodes, dtype=torch.float32)

        # 初始化 GPU nodes: compute speed、utilization、available memory
        compute_speed = np.random.uniform(low=0.5, high=2.0, size=(self.batch_size, self.number_of_gpus))
        utilization = np.random.uniform(low=0.1, high=0.9, size=(self.batch_size, self.number_of_gpus))
        total_memory = np.random.uniform(low=8, high=16, size=(self.batch_size, self.number_of_gpus))  # 假设内存范围在 8GB 到 16GB 之间
        used_memory = total_memory * utilization
        available_memory = total_memory - used_memory

        self.gpu_nodes = np.concatenate(
            [compute_speed.reshape(self.batch_size, self.number_of_gpus, 1), 
            utilization.reshape(self.batch_size, self.number_of_gpus, 1), 
            available_memory.reshape(self.batch_size, self.number_of_gpus, 1)], 
            axis=-1)
        self.gpu_nodes = torch.tensor(self.gpu_nodes, dtype=torch.float32)

        # initialize self.mask_expert, mask out current traffic < 200
        self.mask_expert = self.current_token < 200
        # initialize self.mask_gpu, mask out utilization > 0.9
        self.mask_gpu = utilization > 0.9

        self.initQuality = np.ones(self.batch_size)

        return self.expert_links, self.expert_nodes, self.expert_adj, self.gpu_links, self.gpu_nodes, self.mask_expert, self.mask_gpu


    def done(self):
        return np.all(self.mask_gpu)

    @override
    def step(self, expert_indices, gpu_bool_array, data, gantt_plt=None):
        # 执行动作，维护和更新环境状态，计算奖励
        t1 = time.time()
        rewards, gpu_done = [],[]
        for i in range(self.batch_size):
            expert_selected = expert_indices[i]
            gpu_selected = gpu_bool_array[i]

            # update mask_expert :
            popularity_threshold = 0.1 # 假设我们根据 history_popularity 小于某个阈值来决定是否屏蔽
            for expert in range(self.number_of_experts):
                if self.history_popularity[i, expert] < popularity_threshold:
                    self.mask_expert[i, expert] = True
                else:
                    self.mask_expert[i, expert] = False
            # 检查 expert_selected 是否被屏蔽，如果是则跳过本次循环
            if self.mask_expert[i, expert_selected]:
                rewards.append(0)
                done = self.done()
                gpu_done.append(done)
                continue
            
            previous_traffic = np.sum(self.gpu_links[i, :, :, 1])
            done = False
            '''若一个专家有多个 expert replica，还需要根据 token routing split 进行更新，待实现'''
            # redundant expert_gpu action makes no effect
            for j in range(self.number_of_gpus):
                if self.history_expert_gpu[i][expert_selected][j] == gpu_selected[j]:
                    continue
                # 检查 gpu_selected 是否被屏蔽，如果是则跳过本次循环
                if self.mask_gpu[i, j]:
                    continue
                # UPDATE BASIC INFO 
                if i == 0:
                    self.step_count += 1
                
                previous_placement = self.history_expert_gpu[i, expert_selected, j] # before update 
                
                # update gpu_nodes : compute speed(stable), utilization, available memory
                old_utilization = deepcopy(self.gpu_nodes[i, j, 1])
                old_available_memory = deepcopy(self.gpu_nodes[i, j, 2])

                token_load = self.current_token[i, expert_selected]
                if previous_placement == 1:
                    old_utilization -= token_load / self.current_token[i].sum() # 这里需要改为实际利用率
                    old_available_memory += token_load
                if gpu_selected[j] == 1:
                    old_utilization += token_load / self.current_token[i].sum()
                    old_available_memory -= token_load
                
                if (old_utilization > 0.9) or (old_available_memory < 0): # 如果超过利用率或者内存不足，不更新
                    break
                self.history_expert_gpu[i,expert_selected, j] = gpu_selected[j] # update expert_gpu！！！

                self.gpu_nodes[i, j, 1] = np.clip(old_utilization, 0, 1)  # 更新utilization并确保不超过100%
                self.gpu_nodes[i, j, 2] = np.clip(old_available_memory, 0, None)  # 更新available_memory并确保不为负

                # update gpu_links: bandwidth(stable), token traffic
                for k in range(self.number_of_gpus):
                    if k != j:
                        if previous_placement == 1:
                            self.gpu_links[i, j, k, 1] -= self.current_token[i, expert_selected]
                            self.gpu_links[i, k, j, 1] -= self.current_token[i, expert_selected]
                        if gpu_selected[j] == 1:
                            self.gpu_links[i, j, k, 1] += self.current_token[i, expert_selected]
                            self.gpu_links[i, k, j, 1] += self.current_token[i, expert_selected]
                
                # update mask_gpu
                self.mask_gpu[i, j] = self.gpu_nodes[i, j, 1] > 0.9
                done = self.done()
                
            # update expert_nodes : current token load(already updated), history popularity
            for expert in range(self.number_of_experts):
                self.history_popularity[i, expert] = self.history_popularity[i, expert] * 0.9 + self.current_token[i, expert] * 0.1

            # update expert_links :
            alpha = 0.6  # 调节当前数据和历史数据的权重
            for ii in range(self.number_of_experts):
                total_tokens = np.sum(data[i, ii, :])
                if total_tokens > 0:
                    for j in range(self.number_of_experts):
                        current_ratio = data[i, ii, j] / total_tokens
                        self.expert_links[i, ii, j] = alpha * current_ratio + (1 - alpha) * self.expert_links[i, ii, j]
                else:
                    self.expert_links[i, ii, :] = self.expert_links[i, ii, :] * (1 - alpha)
            
            # update rewards : 如果跨GPU传输的数据量减少，则给予正奖励
            current_traffic = np.sum(self.gpu_links[i, :, :, 1])
            reward = 0
            if current_traffic < previous_traffic:
                reward = previous_traffic - current_traffic
            rewards.append(reward)
            gpu_done.append(done)

        self.expert_nodes = np.concatenate([
            self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
            self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)
        ], axis=2)

        t2 = time.time()
        dur_time = t2-t1
        print('env step() : dur_time', dur_time)
        return self.expert_nodes, self.expert_links, self.gpu_nodes, self.gpu_links, self.mask_expert, self.mask_gpu, dur_time, gpu_done, rewards


