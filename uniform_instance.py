import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import random

class Simulate_Dataset(Dataset):
    '''
    生成的模拟数据为4维
    [样本数,抽样状态数,token数, 层数]
    输入参数: n_moe_layer 层数, n_e_per_layer 每一层专家数
    第1层第1个专家的id为 0, 最后一层最后一个专家id为 n_moe_layer * n_e_per_layer - 1
    生成的数据样本表示了: 在第k个样本中, 进行m条经验的获取,每个token在这一层被分配给的expert。
    '''

    def __init__(self, n_e_per_layer,n_e,n_g, n_moe_layer, simu_tokens,batch_num,batch_size, num_samples, seed=None):
        super(Simulate_Dataset, self).__init__()

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        # 总专家数
        total_experts = int(n_moe_layer * n_e_per_layer)
        tokens = np.zeros((int(batch_size*batch_num),int(num_samples), simu_tokens, n_moe_layer), dtype=int)

        for batch in range(batch_size*batch_num):
        # 只在相邻层之间生成token数量
            for layer in range(n_moe_layer):  #为所有层生成token路由
                candidate_expert = []
                for g in range(n_g):
                    expert = [_ for _ in range((n_e//n_g)*g+layer*(n_e_per_layer//n_g),(n_e//n_g)*g+(n_e_per_layer//n_g)*(layer+1))]
                    candidate_expert.extend(expert)
                # 为当前层和下一层之间的每对专家生成token数量
                # print(f"layer:{layer},experts:{candidate_expert}")
                for token in range(simu_tokens):
                    tokens[batch,:,token,layer] = random.choice(candidate_expert)

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
