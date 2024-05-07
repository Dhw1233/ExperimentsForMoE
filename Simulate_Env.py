import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance import override
from updateEndTimeLB import calEndTimeLB,calEndTimeLBm
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getExpertAffinity
from agent_utils import vanilla_placement
from copy import deepcopy
import torch
import random
import matplotlib.pyplot as plt
import time
from min_job_machine_time import min_job_mch,min_mch_job,min_job_mch1
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

    def done(self):
        return np.all(self.mask_gpu)

    @override
    def step(self, action, mch_a, gantt_plt=None):
        # 执行动作，维护和更新环境状态，计算奖励
        # 删除done相关的
        # action is a int 0 - 224 for 15x15 for example
        time1 = time.time()
        feas, rewards, dones,masks,mch_masks = [],[], [], [],[]
        mch_spaces, mchForJobSpaces = [],[]
        for i in range(self.batch_size):
            # redundant action makes no effect 多余的动作无效
            if action[i] not in self.history_expert_gpu[i]: # 记录已经被调度的任务序列

                # UPDATE BASIC INFO:
                row = action[i] // self.number_of_gpus#取整除
                col = action[i] % self.number_of_gpus#取余数
                if i == 0:
                    self.step_count += 1
                self.history_popularity[i,row, col] = 1

                self.dur_a = self.expert_token[i,row, col,mch_a[i]]
                #action time
                self.history_expert_gpu[i][np.where(self.history_expert_gpu[i]<0)[0][0]] = action[i]

                self.m[i][row][col]=mch_a[i]
                # UPDATE STATE:
                # permissible left shift 允许向左移动

                startTime_a, flag = permissibleLeftShift(a=action[i], mch_a=mch_a[i], durMat=self.comp_time[i], mchMat=self.m[i],
                                                         mchsStartTimes=self.mchsStartTimes[i], opIDsOnMchs=self.opIDsOnMchs[i],mchEndTime=self.mchsEndTimes[i])
                self.flags.append(flag)
                if gantt_plt != None:
                    gantt_plt.gantt_plt(row, col, mch_a.cpu().numpy(), startTime_a, self.dur_a,
                                    self.number_of_experts)
                # update candidate or mask_expert
                if action[i] not in self.last_col[i]:
                    self.candidate[i,action[i] // self.number_of_gpus] += 1
                else:
                    self.mask_expert[i,action[i] // self.number_of_gpus] = 1

                self.temp1[i,row, col] = startTime_a + self.dur_a#完工时间

                #temp1.shape()
                self.LBs[i] = calEndTimeLB(self.temp1[i], self.input_min[i],self.input_mean[i])

                self.current_token[i] = calEndTimeLBm(self.temp1[i],self.input_min[i])


                #self.LBs为所有task最快的完工时间
                # expert_links matrix
                precd, succd = self.getNghbs(action[i], self.opIDsOnMchs[i])

                self.expert_links[i, action[i]] = 0
                self.expert_links[i, action[i], action[i]] = 1
                if action[i] not in self.first_col[i]:
                    self.expert_links[i, action[i], action[i] - 1] = 1
                self.expert_links[i, action[i], precd] = 1
                self.expert_links[i, succd, action[i]] = 1

                '''if action[i] not in self.first_col[i]:
                    self.expert_links[i,action[i]-1, action[i]] = 0
                self.expert_links[i, precd,action[i]] = 0
                self.expert_links[i, action[i],succd] = 0'''
                done = self.done()
                #min_job_mch(gpu_token, mchsEndTimes, number_of_gpus, expert_token, temp, first_col)
                mch_space,mchForJobSpace,mask1,mch_mask = min_job_mch(self.gpu_token[i],self.run_time[i],self.mchsEndTimes[i],self.number_of_gpus,self.comp_time[i],self.temp1[i],self.candidate[i],self.mask_expert[i],done,self.mask_gpu[i])

                mch_spaces.append(mch_space)
                mchForJobSpaces.append(mchForJobSpace)
                masks.append(mask1)
                mch_masks.append(mch_mask)
                #print('action_space',mchForJobSpaces,'mchspace',mch_space)

            # prepare for return
            #-------------------------------------------------------------------------------------
            '''expert_nodes = np.concatenate((self.LBs[i].reshape(-1, 2)/configs.et_normalize_coef,
                                  self.history_popularity[i].reshape(-1, 1)), axis=-1)'''
            #----------------------------------------------------------------------------------------

            '''expert_nodes = np.concatenate((self.expert_token[i].reshape( -1, self.number_of_gpus)/configs.et_normalize_coef,
                                  self.history_popularity[i].reshape( -1, 1)), axis=-1)'''
#--------------------------------------------------------------------------------------------------------------------

            '''expert_nodes = self.current_token[i].reshape(-1, 1) / configs.et_normalize_coef'''
            expert_nodes = np.concatenate((self.current_token[i].reshape(-1, 1) / configs.et_normalize_coef,
                                  #np.expand_dims(self.run_time[i], 1).repeat(self.number_of_gpus, axis=1).reshape(
                                      #self.n_moe_layer, 1)/configs.et_normalize_coef,
                                  self.history_popularity[i].reshape( -1, 1)), axis=-1)

            feas.append(expert_nodes)


            '''reward = self.mchsEndTimes[i][mch_a[i]].max()-self.up_mchendtime[i][mch_a[i]].max()-self.dur_a


            if reward < 0.00001:
                reward = 0
            self.up_mchendtime = np.copy(self.mchsEndTimes)
            for b,c in zip(self.up_mchendtime[i],range(self.number_of_gpus)):
                self.up_mchendtime[i][c] = [0 if i < 0 else i for i in b]
            rewards.append(reward)'''
            reward = -(self.current_token[i].max() - self.max_endTime[i]) # 时间缩短，则奖励为正
            if reward == 0:
                reward = configs.rewardscale
                self.posRewards[i] += reward
            rewards.append(reward)
            self.max_endTime[i] = self.current_token[i].max()

            dones.append(done)


        t2 = time.time()
        mch_masks = np.array(mch_masks)

        #print('t2',t2-t1)
        return self.expert_links, np.array(feas), rewards, dones, self.candidate, masks,mchForJobSpaces,self.mask_gpu,self.gpu_token,self.run_time

    @override
    def reset(self, data):
        # 重置各类计数器和矩阵，为新的环境做准备
        #data (batch_size, n_expert, n_expert)
        self.batch_size = data.shape[0]

        experts_per_layer = self.number_of_experts // self.n_moe_layer

        self.step_count = 0
        # 跟踪各专家对GPU资源的分配状态，-1 代表未分配
        self.m = -1 * np.ones((self.batch_size,self.number_of_experts,self.number_of_gpus), dtype=np.int32)

        self.expert_token = data.astype(np.single)#single单精度浮点数
        # record action history，跟踪每个专家在每个 GPU 上的历史分配情况
        self.history_expert_gpu = -1 * np.ones((self.batch_size,self.number_of_experts*self.number_of_gpus),dtype=np.int32)

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
        # initialize gpu_links matrix : bandwidth , token traffic
        self.gpu_links = []
        for k in range(self.batch_size):
            gpu_links_matrix = np.zeros((self.number_of_gpus, self.number_of_gpus), dtype=[('bandwidth', 'f4'), ('traffic', 'f4')])
            # 初始化带宽为随机值
            for i in range(self.number_of_gpus):
                for j in range(i + 1, self.number_of_gpus):
                    bandwidth = np.random.uniform(1.0, 1.5)
                    gpu_links_matrix[i, j]['bandwidth'] = gpu_links_matrix[j, i]['bandwidth'] = bandwidth
            # 初始化数据传输量
            for i in range(self.number_of_gpus):
                for j in range(self.number_of_gpus):
                    if i != j:
                        traffic = 0
                        # 计算从GPU i到GPU j的专家间的token传输量
                        for expert_i in vanilla_p[i]:
                            for expert_j in vanilla_p[j]:
                                traffic += data[k, int(expert_i), int(expert_j)]
                        gpu_links_matrix[i, j]['traffic'] = traffic
            self.gpu_links.append(gpu_links_matrix)
        print("Initialize Expert_links(affinity) Success!\n", "bandwidth: ", self.gpu_links[0][0,1]['bandwidth'], ", traffic: ", self.gpu_links[0][0,1]['traffic'], "\n")

        # 初始化专家的token数据量
        self.current_token = np.zeros((self.batch_size, self.number_of_experts))
        # 计算每个专家在每个批次中处理的总token数量
        for k in range(self.batch_size):
            for i in range(self.number_of_experts):
                total_tokens = np.sum(data[k, i, :])  # 专家i发送给所有其他专家的token总数
                self.current_token[k, i] = total_tokens

        # 随机初始化专家的历史流行度
        self.history_popularity = np.random.rand(self.batch_size, self.number_of_experts).astype(float)

        # 初始化 expert_nodes: historical popularity、current token load
        self.expert_nodes = np.concatenate(
            [   self.current_token.reshape(self.batch_size, self.number_of_experts, 1),
                self.history_popularity.reshape(self.batch_size, self.number_of_experts, 1)],
            axis=2  # 沿最后一个维度拼接
        )
        print("Initialize expert_nodes Success!\n", "current_token: ", self.expert_nodes[0][0][0], ", history_popularity: ", self.expert_nodes[0][0][1], "\n")

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
        print("Initialize gpu_nodes Success!\n", "compute_speed: ", self.gpu_nodes[0][0][0], ", utilization: ", self.gpu_nodes[0][0][1], ", available_memory: ", self.gpu_nodes[0][0][2], "\n")

        # initialize mask_expert, mask out current traffic < 200
        self.mask_expert = self.current_token < 200
        # initialize mask_gpu, mask out utilization > 0.9
        self.mask_gpu = utilization > 0.9
        print("Initialize Mask_Expert: ", self.mask_expert[0], "\n")
        print("Initialize Mask_GPU ", self.mask_gpu[0], "\n")

        self.initQuality = np.ones(self.batch_size)

        return self.expert_links, self.expert_nodes, self.gpu_links, self.gpu_nodes, self.mask_expert, self.mask_gpu, #self.candidate


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