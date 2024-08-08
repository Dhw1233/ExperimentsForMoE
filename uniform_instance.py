import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import random
from rl_data2 import generate_load_change
class Simulate_Dataset(Dataset):
    '''
    生成的模拟数据为3维
    [样本数,token数,层数]
    输入参数: n_moe_layer 层数, n_e_per_layer 每一层专家数
    第1层第1个专家的id为 0, 最后一层最后一个专家id为 n_moe_layer * n_e_per_layer - 1
    生成的数据样本表示了: 在第k个样本中, 进行m条经验的获取,每个token在这一层被分配给的expert。
    '''

    def __init__(self, n_e_per_layer,n_e,n_g, n_moe_layer, simu_tokens,batch_num,batch_size, time_steps, seed=None):
        super(Simulate_Dataset, self).__init__()

        num_expert = n_e // n_g
        load_history = []
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        # 总专家数
        total_experts = int(n_moe_layer * n_e_per_layer)
        tokens = np.zeros((batch_size * batch_num, time_steps,simu_tokens),dtype=int)
        
        initial_loads = np.random.rand(num_expert)
        initial_loads /= np.sum(initial_loads)
        load_history.append(initial_loads)

        for batch in range(batch_size*batch_num):
            # genertate t's ratio
            for step in range(time_steps):  #为所有层生成token路由
                
                previous_loads = load_history[-1]
                
                load_changes = generate_load_change(batch % num_expert, num_expert)
                
                new_loads = previous_loads + load_changes

                new_loads = np.clip(new_loads,0,None)

                new_loads /= np.sum(new_loads)

                load_history[-1] = new_loads
                
                for token in range(simu_tokens):
            
                    tokens[batch,step,token] = np.random.choice(len(new_loads), p=new_loads)

        self.data = tokens
        
        self.size = len(self.data)

    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def override(fn):
    """
    override decorator
    """
    return fn
