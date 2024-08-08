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

def calculate_expert_gpu(expert_id,n_g,n_e):
    return expert_id*n_g // n_e

def calculate_token_gpu(token_id,n_g,n_token):
    return token_id*n_g // n_token

def calculate_reward(replica_p,n_g,n_e,expert_gradsize,link,token_size,gpu_links,expert_nodes,Tsend):
    '''
    replica_p:[n_e,list],replica_list
    expert_size:the size of expert parameters
    gpu_links:[n_g,n_g,2],bandwidth and gpu traffic
    gpu_nodes:[n_g,3],computation eff,memory

    '''
    throughput = 0.6
    Ta2a = max([sum(gpu_links[:,g,1]/gpu_links[:,g,0]) for g in range(n_g)])
    Tmfc = max([expert_nodes[e,0]*token_size/throughput for e in range(n_e)])
    Tsend = Tsend
    ep_gp = n_e // n_g
    gpu_cnt = np.zeros((n_g,))
    for ep in range(n_e):
        for gp in replica_p[ep]:
            gpu_cnt[gp] += 1
    max_gp = 0
    gp_s = 0
    for gp in range(n_g):
        if gpu_cnt[gp]>max_gp:
            gp_s = gp
            max_gp = gpu_cnt[gp]
    Tallreduce = expert_gradsize*(2*(ep_gp-1)/ep_gp)*(max_gp+ep_gp)*n_g/sum(gpu_links[gp_s,:,0])

    return 4*Ta2a+3*Tmfc+Tsend+Tallreduce

def calculate_gpu_link(tk_expert,n_e,n_g,token_size,bandwidth,compu,placepolicy,M=50):
    '''
    tk_expert:[n_g,n_e],the total number of token transmitted from gpu_i to expert_j 
    n_e: number of experts
    n_g: number of gpus
    token_size: the size of one token
    bandwidth:[n_g,n_g],bandwidth between two gpus
    compu: the computation speed of one gpu 
    placepolicy: expert replica list
    '''
    gpulink = np.zeros((n_g,n_g))
    gpuexpertlink = np.zeros((n_g,n_e,n_g))
    finallink = np.zeros((n_g,n_e,n_g))
    minn = 0x7fffff
    for _ in range(5):
        #generate gpuexpertlink
        for expert in range(n_e):
            gpu = calculate_expert_gpu(expert,n_g,n_e)
            candidate_gpu = placepolicy[expert] + [gpu]
            # print(gpu)
            size = len(candidate_gpu)
            for gpu_id in range(n_g):
                random_numbers = np.random.rand(size)
                probabilities = random_numbers / random_numbers.sum()
                for index in range(size):
                    gpuexpertlink[gpu_id,expert,candidate_gpu[index]] += tk_expert[gpu_id,expert]*probabilities[index]

        goal = max([sum([sum([gpuexpertlink[g,e,g_]*token_size/bandwidth[g,g_] for g in range(n_g)]) for e in range(n_e)])+
                    max([sum([gpuexpertlink[g,e,g_]/compu[g] for g in range(n_g)]) for e in range(n_e)]) for g_ in range(n_g)])
        if goal<minn:
            #generate gpulink
            minn = goal
            gpulink = np.sum(gpuexpertlink,axis=1)
            finallink = gpuexpertlink

    return gpulink,finallink

class Simulate_Env:
    def __init__(self,
                 n_moe_layer,
                 n_e,
                 n_g):
        EzPickle.__init__(self)

        self.step_count = 0
        self.n_moe_layer = n_moe_layer
        self.number_of_experts = n_e // n_g
        self.number_of_gpus = n_g
        # the task id for first column
        self.candidate = []


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


        self.posRewards = np.zeros(self.batch_size)

        self.replica_p = []
        for i in range(self.batch_size):
            dict1 = {}
            for expert in range(self.number_of_experts):
                dict1[expert] = []
            self.replica_p.append(dict1)

        self.expert_links = []
        # initialize expert_links matrix 专家亲和力矩阵，只需要考虑一层的情况
        for i in range(self.batch_size):
            links_matrix = np.ones((self.number_of_experts, self.number_of_experts), dtype=float)
            self.expert_links.append(links_matrix)
        
        expert_links_array = np.array(self.expert_links)
        
        self.expert_links = torch.tensor(expert_links_array, dtype=torch.float32)

        #初始化所有expert对应的gpu
        vanilla_p = vanilla_placement(self.n_moe_layer, self.number_of_experts, self.number_of_gpus)
        for gpu_id, experts_in_gpu in enumerate(vanilla_p):
            for expert_id in experts_in_gpu:
                self.history_expert_gpu[:, expert_id, gpu_id] = True
        # initialize self.gpu_links matrix : bandwidth , token traffic
        # initialize self.current_token : token_number processed by every expert
        self.gpu_links = np.zeros((self.batch_size, self.number_of_gpus * 3, self.number_of_gpus * 3, 2))
        self.current_token = np.zeros((self.batch_size,self.number_of_experts))
        bandwidth = 650000000.0
        for k in range(self.batch_size):
            # Initialize bandwidth with random values
            for i in range(self.number_of_gpus * 3):
                for j in range(i + 1, self.number_of_gpus * 3): 
                    self.gpu_links[k, i, j, 0] = bandwidth
                    self.gpu_links[k, j, i, 0] = bandwidth
                    
            # Initialize data transfer traffic
            
            for token in range(self.token_num):
                expert = data[token]
                token_gpu = calculate_token_gpu(token,self.number_of_gpus,self.token_num)
                expert_gpu = calculate_expert_gpu(expert,self.number_of_gpus,self.number_of_experts)
                self.gpu_links[k,token_gpu,expert_gpu] += 1
                self.current_token[k,expert] += 1
                        
        print("Initialize GPU_links Success!\n", "bandwidth[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][0], ", token traffic[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][1], "\n")
        
        # 随机初始化专家的历史流行度
        self.history_popularity = np.zeros((self.batch_size, self.number_of_experts)).astype(float)

        # 初始化 self.expert_nodes: historical popularity、current token load
        self.expert_nodes = np.concatenate(
            [   self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)],
            axis=-1  # 沿最后一个维度拼接
        )

        print("Initialize self.expert_nodes Success!\n", "current_token: ", self.expert_nodes[0][0][0], ", history_popularity: ", self.expert_nodes[0][0][1], "\n")

        # 随机初始化 GPU nodes: compute speed、utilization、available memory
        compute_speed = np.random.uniform(low=0.5, high=2.0, size=(self.batch_size, self.number_of_gpus))
        lower_bound = self.number_of_experts // self.number_of_gpus
        utilization_expert = np.random.uniform(low=lower_bound, high=lower_bound * self.number_of_experts, size=(self.batch_size, self.number_of_gpus))
         # 假设内存范围在 8GB 到 16GB 之间

        self.gpu_nodes = np.concatenate(
            [compute_speed.reshape(self.batch_size, self.number_of_gpus, 1), 
            utilization_expert.reshape(self.batch_size, self.number_of_gpus, 1)], 
            axis=-1)
        print("Initialize self.gpu_nodes Success!\n", "compute_speed: ", self.gpu_nodes[0][0][0], ", utilization: ", self.gpu_nodes[0][0][1], ", available_memory: ", self.gpu_nodes[0][0][2], "\n")


        self.initQuality = np.ones(self.batch_size)

        return self.expert_links, self.expert_nodes, self.gpu_links, self.gpu_nodes


    def done(self):
        return np.all(self.mask_gpu)

    def step(self, expert_index, gpu_index, data, gantt_plt=None):
        # 执行动作，维护和更新环境状态，计算奖励
        t1 = time.time()
        rewards, gpu_done = [],[]
        act_index = gpu_index // 3
        gpu_index = gpu_index % 3
    # for each batch we generate a env step
        for i in range(self.batch_size):
            #calculate replica_list
            expert_selected = expert_index[i]
            gpu_selected = gpu_index[i]
            act_selected = act_index[i]
            links_matrix = np.ones((self.number_of_experts, self.number_of_experts), dtype=float)
            # print("actions:",expert_selected,gpu_selected,act_selected)
            # update expert_links :
            alpha = 0.6  # 调节当前数据和历史数据的权重
            if act_selected == 1:
                self.replica_p[i][expert_selected].append(gpu_selected)
                expert_gpu = calculate_expert_gpu(expert_selected,self.number_of_gpus,self.number_of_experts)
                Tsend = self.expert_size / self.gpu_links[i,expert_gpu,gpu_selected,0]
            elif act_selected == 2:
                if gpu_selected in self.replica_p[i][expert_selected]:
                    self.replica_p[i][expert_selected].remove(gpu_selected)
                Tsend = 0
            # update expert_node and gpu_links,LP to be realized,待实现
            # 1.计算每个expert在每个gpu上得到的token个数
            # 2.通过解线性规划求得跨gpu的传输量
            tk_gpu = np.zeros((self.number_of_gpus,self.number_of_experts))
            
            
            self.gpu_links[i,:,:,1],link = calculate_gpu_link(tk_gpu,self.number_of_experts,self.number_of_gpus,self.token_size,self.gpu_links[i,:,:,0],
                                                         self.gpu_nodes[i,:,0],self.replica_p[i],50)

            for expert in range(self.number_of_experts):
                self.history_popularity[i, expert] = self.history_popularity[i, expert] * 0.9 + self.current_token[i, expert] * 0.1

            for layer in range(self.n_moe_layer):
                            for token in range(self.token_num):
                                expert = data[i,token,layer]
                                token_gpu = calculate_token_gpu(token,self.number_of_gpus,self.token_num)
                                expert_gpu = calculate_expert_gpu(expert,self.number_of_gpus,self.number_of_experts)
                                tk_gpu[token_gpu,expert] += 1
                                self.current_token[i,expert] += 1

            self.expert_nodes = np.concatenate([
                        self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                        self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)
                    ], axis=2)
            
            # update gpu_links
            utilize_expert = self.gpu_nodes[i,:,1]
            # calculate gpu nodes
            for j in range(self.number_of_gpus):                
                self.gpu_nodes[i, j, 1] = utilize_expert[j] + (gpu_index == j and act_index == 1)
                # update mask_gpu
            reward  = calculate_reward(self.replica_p[i],self.number_of_gpus,self.number_of_experts,self.expert_gradsize,link,
                                      self.token_size,self.gpu_links[i],self.expert_nodes[i],Tsend)
            rewards.append(reward)

        t2 = time.time()
        dur_time = t2-t1
        print('env step() : dur_time', dur_time)
        return self.expert_nodes, self.expert_links, self.gpu_nodes, self.gpu_links, dur_time, rewards


