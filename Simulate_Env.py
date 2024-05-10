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
        #data (batch_size, n_expert, n_expert)
        self.batch_size = data.shape[0]

        experts_per_layer = self.number_of_experts // self.n_moe_layer

        self.step_count = 0
        # 跟踪各专家对GPU资源的分配状态，-1 代表未分配
        self.history_expert_gpu = np.zeros((self.batch_size, self.number_of_experts, self.number_of_gpus), dtype=bool)

        self.expert_token = data.astype(np.single)#single单精度浮点数、

        self.posRewards = np.zeros(self.batch_size)
        self.expert_links = []
        # initialize expert_links matrix 专家亲和力矩阵，只需要考虑下一层！
        for i in range(self.batch_size):
            # 创建全零矩阵，大小为总专家数 x 总专家数
            links_matrix = np.zeros((self.number_of_experts, self.number_of_experts), dtype=float)
            # 建立每层专家之间的连接
            for layer in range(self.n_moe_layer - 1):
                start_index = layer * experts_per_layer
                end_index = start_index + experts_per_layer

                down_start_index = (layer + 1) * experts_per_layer
                down_end_index = down_start_index + experts_per_layer
                
                # 生成随机数并进行归一化
                random_weights = np.random.rand(experts_per_layer, experts_per_layer)
                row_sums = random_weights.sum(axis=1, keepdims=True)
                normalized_weights = random_weights / row_sums  # 归一化权重

                links_matrix[start_index:end_index, down_start_index:down_end_index] = normalized_weights
            self.expert_links.append(links_matrix)
        # print("Initialize Expert_links Success!\n", "affinity: ", self.expert_links[0][0,:], "\n")
        expert_links_array = np.array(self.expert_links)
        self.expert_links = torch.tensor(expert_links_array, dtype=torch.float32)

        vanilla_p = vanilla_placement(self.n_moe_layer, self.number_of_experts / self.n_moe_layer, self.number_of_gpus)
        for gpu_id, experts_in_gpu in enumerate(vanilla_p):
            for expert_id in experts_in_gpu:
                self.history_expert_gpu[:, expert_id, gpu_id] = True
        # print("Initialize history_expert_gpu Success!\n", "history_expert_gpu[batch 0][expert 0]: ", self.history_expert_gpu[0][0], "\n")

        # initialize self.gpu_links matrix : bandwidth , token traffic
        self.gpu_links = np.zeros((self.batch_size, self.number_of_gpus, self.number_of_gpus, 2))
        for k in range(self.batch_size):
            # Initialize bandwidth with random values
            for i in range(self.number_of_gpus):
                for j in range(i + 1, self.number_of_gpus):
                    bandwidth = np.random.rand() * 1000  # 带宽范围在 0 到 1000 之间
                    self.gpu_links[k, i, j, 0] = bandwidth
                    self.gpu_links[k, j, i, 0] = bandwidth
            # Initialize data transfer traffic
            for i in range(self.number_of_gpus):
                for j in range(self.number_of_gpus):
                    if i != j:
                        traffic = 0
                        # Calculate token traffic from GPU i to GPU j
                        for expert_i in vanilla_p[i]:
                            for expert_j in vanilla_p[j]:
                                traffic += data[k, int(expert_i), int(expert_j)]
                        self.gpu_links[k, i, j, 1] = traffic
        print("Initialize GPU_links Success!\n", "bandwidth[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][0], ", token traffic[batch 0][gpu 0][gpu 1]: ", self.gpu_links[0][0][1][1], "\n")
        
        # 初始化专家的token数据量
        self.current_token = np.zeros((self.batch_size, self.number_of_experts))
        # 计算每个专家在每个批次中处理的总token数量
        for k in range(self.batch_size):
            for i in range(self.number_of_experts):
                total_tokens = np.sum(data[k, i, :])  # 专家i发送给所有其他专家的token总数
                self.current_token[k, i] = total_tokens

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
        self.mask_expert = self.current_token < 200
        # initialize self.mask_gpu, mask out utilization > 0.9
        self.mask_gpu = utilization > 0.9

        self.initQuality = np.ones(self.batch_size)

        return self.expert_links, self.expert_nodes, self.gpu_links, self.gpu_nodes, self.mask_expert, self.mask_gpu


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



class GANTT():
    def __init__(self,total_n_job,number_of_gpus):
        super(GANTT, self).__init__()

        self.total_n_job = total_n_job
        self.number_of_gpus = number_of_gpus
        self.initialize_plt()
    def colour_gen(self,n):
        '''
        为工件生成随机颜色
        :param n: 工件数
        :return: 颜色列表
        '''
        color_bits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        colours = []
        random.seed(234)
        for i in range(n):
            colour_bits = ['#']
            colour_bits.extend(random.sample(color_bits, 6))
            colours.append(''.join(colour_bits))
        return colours
    def initialize_plt(self):
        plt.figure(figsize=((self.total_n_job * 1.5, self.number_of_gpus)))
        y_value = list(range(1, 21))

        plt.xlabel('Makespan', size=20, fontdict={'family': 'SimSun'})
        plt.ylabel('机器号', size=20, fontdict={'family': 'SimSun'})
        plt.yticks(y_value, fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)


    def gantt_plt(self,job, operation, mach_a, start_time, dur_a,number_of_experts):
        '''
        绘制甘特图
        :param job: 工件号
        :param operation: 工序号
        :param mach_a: 机器号
        :param start_time: 开始时间
        :param dur_a: 加工时间
        :param colors: 颜色列表
        '''
        colors = self.colour_gen(number_of_experts)
        plt.barh(mach_a + 1, dur_a, 0.5, left=start_time, color=colors[job])
        plt.text(start_time + dur_a / 10, mach_a + 0.9, 'J%s\nO%s' % (job + 1, operation + 1), size=6)