import torch


class Memory:
    # 待修改！
    def __init__(self):
        self.expert_node_fea = [] # expert 特征矩阵
        self.expert_link_fea = [] # expert 邻接矩阵
        self.expert_selection = [] # 上一次调度的 expert indice
        self.mask_expert = []
        self.expert_logprobs = [] # expert 的 log 概率  

        self.gpu_node_fea = [] # gpu 特征矩阵
        self.gpu_link_fea = [] # gpu 邻接矩阵
        self.gpu_selection = []
        self.mask_gpu = []
        self.gpu_logprobs = [] # gpu 的 log 概率
        
        self.env_rewards = [] # 奖励
        self.env_done = []


    def clear_memory(self):
        del self.expert_link_fea[:]
        del self.expert_node_fea[:]
        del self.expert_selection[:]
        del self.mask_expert[:]
        del self.expert_logprobs[:]

        del self.gpu_node_fea[:]
        del self.gpu_link_fea[:]
        del self.gpu_selection[:]
        del self.mask_gpu[:]
        del self.gpu_logprobs[:]
        
        del self.env_rewards[:]
        del self.env_done[:]
