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

def calculate_workload_ratios(capacities, bandwidths, local_gpu_id, expert_gpu_ids):
    total_capacity = sum(capacities[gpu_id] for gpu_id in expert_gpu_ids)
    capacity_ratios = [capacities[gpu_id] / total_capacity for gpu_id in expert_gpu_ids]
    
    total_bandwidth = sum(bandwidths[local_gpu_id][gpu_id] for gpu_id in expert_gpu_ids)
    bandwidth_ratios = [bandwidths[local_gpu_id][gpu_id] / total_bandwidth for gpu_id in expert_gpu_ids]
    
    # 综合考虑容量比例和带宽比例
    combined_ratios = [(capacity_ratios[i] + bandwidth_ratios[i]) / 2 for i in range(len(expert_gpu_ids))]
    return combined_ratios

def calculate_expert_gpu(expert_id,n_g,n_e):
    return expert_id*n_g // n_e

def calculate_token_gpu(token_id,n_g,n_token):
    return token_id*n_g // n_token

def calculate_reward(replica_p,history_expert_gpu,expert_gradsize,expert_size,token_size,gpu_links,gpu_nodes,expert_nodes):
    return random.random()
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


    @override
    def reset(self, data,token_num,expert_size,expert_gradsize,token_size):
        # 重置各类计数器和矩阵，为新的环境做准备
        #data (sample_cnt, n_expert, n_expert)
        self.batch_size = data.shape[0]
        self.expert_size = expert_size
        self.expert_gradsize = expert_gradsize
        self.token_size = token_size
        self.token_num = token_num
        experts_per_layer = self.number_of_experts // self.n_moe_layer

        self.step_count = 0
        # 跟踪各专家对GPU资源的分配状态，-1 代表未分配
        self.history_expert_gpu = np.zeros((self.batch_size, self.number_of_experts, self.number_of_gpus), dtype=bool)

        self.expert_token = data.astype(np.single)#single单精度浮点数、

        self.posRewards = np.zeros(self.batch_size)

        self.replica_p = []
        for i in range(self.batch_size):
            dict1 = {}
            for expert in range(self.number_of_experts):
                dict1[expert] = []
            self.replica_p.append(dict1)

        self.expert_links = []
        # initialize expert_links matrix 专家亲和力矩阵，只需要考虑下一层！
        for i in range(self.batch_size):
            links_matrix = np.zeros((self.number_of_experts, self.number_of_experts), dtype=float)
            for layer in range(self.n_moe_layer - 1):
                for token in range(self.token_num):
                    links_matrix[data[i,token,layer],data[i,token,layer+1]] += 1
            self.expert_links.append(links_matrix)
        expert_links_array = np.array(self.expert_links)
        self.expert_links = torch.tensor(expert_links_array, dtype=torch.float32)

        #初始化所有expert对应的gpu
        vanilla_p = vanilla_placement(self.n_moe_layer, self.number_of_experts / self.n_moe_layer, self.number_of_gpus)
        for gpu_id, experts_in_gpu in enumerate(vanilla_p):
            for expert_id in experts_in_gpu:
                self.history_expert_gpu[:, expert_id, gpu_id] = True
        
        # initialize self.gpu_links matrix : bandwidth , token traffic
        # initialize self.current_token : token_number processed by every expert
        self.gpu_links = np.zeros((self.batch_size, self.number_of_gpus, self.number_of_gpus, 2))
        self.current_token = np.zeros((self.batch_size, self.number_of_experts))
        for k in range(self.batch_size):
            # Initialize bandwidth with random values
            for i in range(self.number_of_gpus):
                for j in range(i + 1, self.number_of_gpus):
                    bandwidth = np.random.rand() * 1000  # 带宽范围在 0 到 1000 之间
                    self.gpu_links[k, i, j, 0] = bandwidth
                    self.gpu_links[k, j, i, 0] = bandwidth

            # Initialize data transfer traffic
            for layer in range(self.n_moe_layer):
                for token in range(self.token_num):
                    expert = data[i,token,layer]
                    token_gpu = calculate_token_gpu(token,self.number_of_gpus,self.token_num)
                    expert_gpu = calculate_expert_gpu(expert,self.number_of_gpus,self.number_of_experts)
                    self.gpu_links[k,token_gpu,expert_gpu] += 1
                    self.current_token[k,expert] += 1
                        
        print("Initialize GPU_links Success!\n", "bandwidth[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][0], ", token traffic[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][1], "\n")
        

        # 随机初始化专家的历史流行度
        self.history_popularity = np.random.rand(self.batch_size, self.number_of_experts).astype(float)

        # 初始化 self.expert_nodes: historical popularity、current token load
        self.expert_nodes = np.concatenate(
            [   self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)],
            axis=2  # 沿最后一个维度拼接
        )
        print("Initialize self.expert_nodes Success!\n", "current_token: ", self.expert_nodes[0][0][0], ", history_popularity: ", self.expert_nodes[0][0][1], "\n")

        # 随机初始化 GPU nodes: compute speed、utilization、available memory
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
        print("Initialize self.gpu_nodes Success!\n", "compute_speed: ", self.gpu_nodes[0][0][0], ", utilization: ", self.gpu_nodes[0][0][1], ", available_memory: ", self.gpu_nodes[0][0][2], "\n")

        # initialize self.mask_expert, mask out current traffic < 200
        self.mask_expert = self.current_token < 2
        # initialize self.mask_gpu, mask out utilization > 0.9
        self.mask_gpu = utilization > 0.9

        self.initQuality = np.ones(self.batch_size)

        return self.expert_links, self.expert_nodes, self.gpu_links, self.gpu_nodes, self.mask_expert, self.mask_gpu


    def done(self):
        return np.all(self.mask_gpu)

    @override
    def step(self, expert_index, gpu_index,act_index, data, gantt_plt=None):
        # 执行动作，维护和更新环境状态，计算奖励
        t1 = time.time()
        rewards, gpu_done = [],[]
        self.gpu_links = np.zeros((self.batch_size, self.number_of_gpus, self.number_of_gpus, 2))
        self.current_token = np.zeros((self.batch_size, self.number_of_experts))
        for i in range(self.batch_size):
            #calculate replica_list
            expert_selected = expert_index[i]
            gpu_selected = gpu_index[i]
            act_selected = act_index[i]
            links_matrix = np.zeros((self.number_of_experts, self.number_of_experts), dtype=float)
            # print("actions:",expert_selected,gpu_selected,act_selected)
            if act_selected == 1:
                self.replica_p[i][expert_selected].append(gpu_selected)
            else:
                if gpu_selected in self.replica_p[i][expert_selected]:
                    self.replica_p[i][expert_selected].remove(gpu_selected)

            # update expert_node and gpu_links,LP to be realized,待实现
            for layer in range(self.n_moe_layer):
                for token in range(self.token_num):
                    expert = data[i,token,layer]
                    token_gpu = calculate_token_gpu(token,self.number_of_gpus,self.token_num)
                    expert_gpu = calculate_expert_gpu(expert,self.number_of_gpus,self.number_of_experts)
                    self.gpu_links[i,token_gpu,expert_gpu] += 1
                    self.current_token[i,expert] += 1
            
            for expert in range(self.number_of_experts):
                self.history_popularity[i, expert] = self.history_popularity[i, expert] * 0.9 + self.current_token[i, expert] * 0.1

            # update expert_links :
            alpha = 0.6  # 调节当前数据和历史数据的权重
            for layer in range(self.n_moe_layer-1):
                for token in range(self.token_num):
                    links_matrix[data[i,token,layer],data[i,token,layer+1]] += 1
                    
            for ii in range(self.number_of_experts):
                for j in range(self.number_of_experts):
                    if ii != j:
                        self.expert_links[i, ii, j] = alpha * links_matrix[ii,j] + (1 - alpha) * self.expert_links[i, ii, j]

            # update mask_expert :
            popularity_threshold = 0.1 # 假设我们根据 history_popularity 小于某个阈值来决定是否屏蔽
            for expert in range(self.number_of_experts):
                if self.history_popularity[i, expert] < popularity_threshold:
                    self.mask_expert[i, expert] = True
                else:
                    self.mask_expert[i, expert] = False

            self.expert_nodes = np.concatenate([
                        self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                        self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)
                    ], axis=2)
            
            # update gpu_links
            capicities = self.gpu_nodes[i,:,1]
            available_memory = self.gpu_nodes[i,:,1]
            bandwith = self.gpu_links[i,:,:,0]
            # calculate gpu nodes
            for j in range(self.number_of_gpus):                
                # update gpu_nodes : compute speed(stable), utilization, available memory
                old_utilization = deepcopy(self.gpu_nodes[i, j, 1])
                old_available_memory = deepcopy(self.gpu_nodes[i, j, 2])
                if (old_utilization > 0.9) or (old_available_memory < 0): # 如果超过利用率或者内存不足，不更新
                    break
                self.gpu_nodes[i, j, 1] = np.clip(old_utilization, 0, 1)  # 更新utilization并确保不超过100%
                self.gpu_nodes[i, j, 2] = np.clip(old_available_memory, 0, None)  # 更新available_memory并确保不为负
                # update mask_gpu
                self.mask_gpu[i, j] = self.gpu_nodes[i, j, 1] > 0.9
            reward  = calculate_reward(self.replica_p[i],self.history_expert_gpu[i],self.expert_gradsize,self.expert_size,
                                      self.token_size,self.gpu_links[i],self.gpu_nodes[i],self.expert_nodes[i])
            rewards.append(reward)

        t2 = time.time()
        dur_time = t2-t1
        print('env step() : dur_time', dur_time)
        return self.expert_nodes, self.expert_links, self.gpu_nodes, self.gpu_links, self.mask_expert, self.mask_gpu, dur_time, rewards


