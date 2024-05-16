import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel


class Simulate_Dataset(Dataset):
    '''
    生成的模拟数据为3维
    [样本数, 专家数, 专家数]
    输入参数: n_moe_layer 层数, n_e_per_layer 每一层专家数
    第1层第1个专家的id为 0, 最后一层最后一个专家id为 n_moe_layer * n_e_per_layer - 1
    生成的数据样本表示了: 在第k个样本中, 专家i到专家j需要路由的token数量。
    '''

    def __init__(self, n_e_per_layer, n_moe_layer, simu_tokens, num_samples=1000000, seed=None):
        super(Simulate_Dataset, self).__init__()

        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        # 总专家数
        total_experts = int(n_moe_layer * n_e_per_layer)
        tokens = np.zeros((int(num_samples), total_experts, total_experts), dtype=int)

        # 只在相邻层之间生成token数量
        for layer in range(n_moe_layer - 1):  # 不包括最后一层
            start_id_current = int(layer * n_e_per_layer)
            end_id_current = int(start_id_current + n_e_per_layer)
            start_id_next = int(end_id_current)
            end_id_next = int(start_id_next + n_e_per_layer)
            # 为当前层和下一层之间的每对专家生成token数量
            for i in range(start_id_current, end_id_current):
                for j in range(start_id_next, end_id_next):
                    tokens[:, i, j] = np.random.randint(0, simu_tokens + 1, size=num_samples)
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
